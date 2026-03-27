namespace Cortex.Geo;

/// <summary>
/// A static R-tree spatial index using Sort-Tile-Recursive (STR) bulk loading.
/// Indexes 2D points for fast bounding-box and nearest-neighbor queries.
/// O(n log n) build time, O(log n + k) query time.
///
/// Usage:
///   var tree = RTree.Build(geoColumn);
///   var nearby = tree.Query(bbox);           // all points in bounding box
///   var (idx, dist) = tree.Nearest(point);   // closest point
/// </summary>
public class RTree
{
    private const int MaxLeafSize = 16;

    private readonly RTreeNode _root;
    private readonly GeoColumn _points;

    private RTree(RTreeNode root, GeoColumn points)
    {
        _root = root;
        _points = points;
    }

    /// <summary>Build an R-tree from a GeoColumn using STR bulk loading.</summary>
    public static RTree Build(GeoColumn points)
    {
        if (points.Length == 0)
            return new RTree(new RTreeLeaf(Array.Empty<int>(), new BoundingBox(0, 0, 0, 0)), points);

        var indices = Enumerable.Range(0, points.Length).ToArray();
        var root = BuildSTR(points, indices, 0);
        return new RTree(root, points);
    }

    /// <summary>Find all point indices within a bounding box.</summary>
    public List<int> Query(BoundingBox bbox)
    {
        var result = new List<int>();
        QueryNode(_root, bbox, result);
        return result;
    }

    /// <summary>Find all point indices within a given distance (km) of a target.</summary>
    public List<(int Index, double DistanceKm)> QueryRadius(GeoPoint target, double radiusKm)
    {
        var (latDeg, lonDeg) = GeoOps.KmToDegrees(radiusKm, target.Latitude);
        var bbox = BoundingBox.FromPoint(target, latDeg, lonDeg);
        var candidates = Query(bbox);

        var result = new List<(int, double)>();
        foreach (var idx in candidates)
        {
            double dist = GeoOps.HaversineKm(_points[idx], target);
            if (dist <= radiusKm)
                result.Add((idx, dist));
        }
        return result;
    }

    /// <summary>Find the nearest point to a target. Returns (index, distanceKm).</summary>
    public (int Index, double DistanceKm) Nearest(GeoPoint target)
    {
        double bestDist = double.MaxValue;
        int bestIdx = -1;
        NearestSearch(_root, target, ref bestDist, ref bestIdx);
        return (bestIdx, bestDist);
    }

    /// <summary>Find the k nearest points. Returns sorted by distance ascending.</summary>
    public List<(int Index, double DistanceKm)> KNearest(GeoPoint target, int k)
    {
        // Use expanding-radius search with the R-tree
        var result = new List<(int Index, double DistanceKm)>();

        // Start with a reasonable radius guess, expand if needed
        double radius = 10; // 10 km initial guess
        for (int attempt = 0; attempt < 20; attempt++)
        {
            var candidates = QueryRadius(target, radius);
            if (candidates.Count >= k)
            {
                candidates.Sort((a, b) => a.DistanceKm.CompareTo(b.DistanceKm));
                return candidates.Take(k).ToList();
            }
            radius *= 2;
        }

        // Fallback: brute force
        for (int i = 0; i < _points.Length; i++)
        {
            double dist = GeoOps.HaversineKm(_points[i], target);
            result.Add((i, dist));
        }
        result.Sort((a, b) => a.DistanceKm.CompareTo(b.DistanceKm));
        return result.Take(k).ToList();
    }

    private void QueryNode(RTreeNode node, BoundingBox bbox, List<int> result)
    {
        if (!node.Bounds.Intersects(bbox)) return;

        if (node is RTreeLeaf leaf)
        {
            foreach (var idx in leaf.Indices)
            {
                if (bbox.Contains(_points[idx]))
                    result.Add(idx);
            }
        }
        else if (node is RTreeBranch branch)
        {
            foreach (var child in branch.Children)
                QueryNode(child, bbox, result);
        }
    }

    private void NearestSearch(RTreeNode node, GeoPoint target, ref double bestDist, ref int bestIdx)
    {
        // Prune: if the node's bounding box is farther than current best, skip
        double minBoxDist = MinDistanceToBBox(target, node.Bounds);
        if (minBoxDist > bestDist) return;

        if (node is RTreeLeaf leaf)
        {
            foreach (var idx in leaf.Indices)
            {
                double dist = GeoOps.HaversineKm(_points[idx], target);
                if (dist < bestDist)
                {
                    bestDist = dist;
                    bestIdx = idx;
                }
            }
        }
        else if (node is RTreeBranch branch)
        {
            // Visit children sorted by distance to target (nearest first)
            var ordered = branch.Children
                .Select(c => (Child: c, Dist: MinDistanceToBBox(target, c.Bounds)))
                .OrderBy(x => x.Dist);

            foreach (var (child, _) in ordered)
                NearestSearch(child, target, ref bestDist, ref bestIdx);
        }
    }

    /// <summary>Minimum Haversine-approximated distance from a point to a bounding box.</summary>
    private static double MinDistanceToBBox(GeoPoint point, BoundingBox bbox)
    {
        double clampedLat = Math.Clamp(point.Latitude, bbox.MinLat, bbox.MaxLat);
        double clampedLon = Math.Clamp(point.Longitude, bbox.MinLon, bbox.MaxLon);
        if (clampedLat == point.Latitude && clampedLon == point.Longitude)
            return 0; // point is inside bbox
        return GeoOps.HaversineKm(point, new GeoPoint(clampedLat, clampedLon));
    }

    // ===== STR Bulk Loading =====

    private static RTreeNode BuildSTR(GeoColumn points, int[] indices, int depth)
    {
        if (indices.Length <= MaxLeafSize)
        {
            var bbox = ComputeBounds(points, indices);
            return new RTreeLeaf(indices, bbox);
        }

        // Sort by alternating axis (lat for even depth, lon for odd)
        bool sortByLat = depth % 2 == 0;
        if (sortByLat)
            Array.Sort(indices, (a, b) => points[a].Latitude.CompareTo(points[b].Latitude));
        else
            Array.Sort(indices, (a, b) => points[a].Longitude.CompareTo(points[b].Longitude));

        // Split into sqrt(n/leafSize) slices
        int numSlices = Math.Max(2, (int)Math.Ceiling(Math.Sqrt((double)indices.Length / MaxLeafSize)));
        int sliceSize = (indices.Length + numSlices - 1) / numSlices;

        var children = new List<RTreeNode>();
        for (int s = 0; s < numSlices; s++)
        {
            int start = s * sliceSize;
            int end = Math.Min(start + sliceSize, indices.Length);
            if (start >= indices.Length) break;

            var slice = indices[start..end];
            children.Add(BuildSTR(points, slice, depth + 1));
        }

        var branchBbox = children[0].Bounds;
        for (int i = 1; i < children.Count; i++)
            branchBbox = Union(branchBbox, children[i].Bounds);

        return new RTreeBranch(children.ToArray(), branchBbox);
    }

    private static BoundingBox ComputeBounds(GeoColumn points, int[] indices)
    {
        double minLat = double.MaxValue, maxLat = double.MinValue;
        double minLon = double.MaxValue, maxLon = double.MinValue;
        foreach (var idx in indices)
        {
            var p = points[idx];
            if (p.Latitude < minLat) minLat = p.Latitude;
            if (p.Latitude > maxLat) maxLat = p.Latitude;
            if (p.Longitude < minLon) minLon = p.Longitude;
            if (p.Longitude > maxLon) maxLon = p.Longitude;
        }
        return new BoundingBox(minLat, minLon, maxLat, maxLon);
    }

    private static BoundingBox Union(BoundingBox a, BoundingBox b) =>
        new(Math.Min(a.MinLat, b.MinLat), Math.Min(a.MinLon, b.MinLon),
            Math.Max(a.MaxLat, b.MaxLat), Math.Max(a.MaxLon, b.MaxLon));
}

internal abstract class RTreeNode
{
    public BoundingBox Bounds { get; }
    protected RTreeNode(BoundingBox bounds) => Bounds = bounds;
}

internal class RTreeLeaf : RTreeNode
{
    public int[] Indices { get; }
    public RTreeLeaf(int[] indices, BoundingBox bounds) : base(bounds) => Indices = indices;
}

internal class RTreeBranch : RTreeNode
{
    public RTreeNode[] Children { get; }
    public RTreeBranch(RTreeNode[] children, BoundingBox bounds) : base(bounds) => Children = children;
}
