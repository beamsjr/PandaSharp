using Cortex.Column;

namespace Cortex.Geo;

/// <summary>
/// A columnar geometry column storing 2D points as parallel lat/lon arrays.
/// Backed by Cortex Column&lt;double&gt; for zero-copy interop.
/// </summary>
public class GeoColumn
{
    // Primary storage: either owned arrays or column references (zero-copy)
    private readonly double[]? _latArray;
    private readonly double[]? _lonArray;
    private readonly Column<double>? _latColumn;
    private readonly Column<double>? _lonColumn;

    public string Name { get; }
    public int Length { get; }

    public GeoColumn(string name, double[] latitudes, double[] longitudes)
    {
        if (latitudes.Length != longitudes.Length)
            throw new ArgumentException("Latitude and longitude arrays must have the same length.");
        Name = name;
        Length = latitudes.Length;
        _latArray = latitudes;
        _lonArray = longitudes;
    }

    /// <summary>
    /// Create a GeoColumn backed directly by Column&lt;double&gt; references — no data copy.
    /// </summary>
    public GeoColumn(string name, Column<double> latColumn, Column<double> lonColumn)
    {
        if (latColumn.Length != lonColumn.Length)
            throw new ArgumentException("Latitude and longitude columns must have the same length.");
        Name = name;
        Length = latColumn.Length;
        _latColumn = latColumn;
        _lonColumn = lonColumn;
    }

    public GeoColumn(string name, GeoPoint[] points)
    {
        Name = name;
        Length = points.Length;
        _latArray = new double[points.Length];
        _lonArray = new double[points.Length];
        for (int i = 0; i < points.Length; i++)
        {
            _latArray[i] = points[i].Latitude;
            _lonArray[i] = points[i].Longitude;
        }
    }

    public GeoPoint this[int index]
    {
        get
        {
            var lats = Latitudes;
            var lons = Longitudes;
            return GeoPoint.FromProjected(lats[index], lons[index]);
        }
    }

    public ReadOnlySpan<double> Latitudes => _latColumn is not null ? _latColumn.Values : _latArray;
    public ReadOnlySpan<double> Longitudes => _lonColumn is not null ? _lonColumn.Values : _lonArray;

    /// <summary>
    /// Compute Haversine distance from every point to a target point. Returns km.
    /// </summary>
    public Column<double> DistanceTo(GeoPoint target)
    {
        var lats = Latitudes;
        var lons = Longitudes;
        var result = new double[Length];
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.HaversineKm(new GeoPoint(lats[i], lons[i]), target);
        return new Column<double>($"{Name}_dist_km", result);
    }

    /// <summary>
    /// Compute pairwise distance between this column and another (same length). Returns km.
    /// </summary>
    public Column<double> DistanceTo(GeoColumn other)
    {
        if (Length != other.Length)
            throw new ArgumentException("GeoColumns must have the same length.");
        var result = new double[Length];
        var lats = Latitudes;
        var lons = Longitudes;
        var oLats = other.Latitudes;
        var oLons = other.Longitudes;
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.HaversineKm(new GeoPoint(lats[i], lons[i]), new GeoPoint(oLats[i], oLons[i]));
        return new Column<double>($"dist_km", result);
    }

    /// <summary>
    /// Check which points fall within a bounding box. Returns a boolean mask.
    /// </summary>
    public bool[] Within(BoundingBox bbox)
    {
        var result = new bool[Length];
        var lats = Latitudes;
        var lons = Longitudes;
        for (int i = 0; i < Length; i++)
            result[i] = bbox.Contains(new GeoPoint(lats[i], lons[i]));
        return result;
    }

    /// <summary>
    /// Check which points are within a given distance (km) of a target point.
    /// Uses bounding box pre-filter + exact Haversine check.
    /// </summary>
    public bool[] WithinDistance(GeoPoint target, double radiusKm)
    {
        var result = new bool[Length];
        var (latDeg, lonDeg) = GeoOps.KmToDegrees(radiusKm, target.Latitude);
        var bbox = BoundingBox.FromPoint(target, latDeg, lonDeg);
        var lats = Latitudes;
        var lons = Longitudes;

        for (int i = 0; i < Length; i++)
        {
            var pt = new GeoPoint(lats[i], lons[i]);
            // Fast reject: bounding box check
            if (!bbox.Contains(pt))
            {
                result[i] = false;
                continue;
            }
            // Exact check
            result[i] = GeoOps.HaversineKm(pt, target) <= radiusKm;
        }
        return result;
    }

    /// <summary>
    /// Buffer each point by a radius in km, returning bounding boxes.
    /// </summary>
    public BoundingBox[] Buffer(double radiusKm)
    {
        var result = new BoundingBox[Length];
        var lats = Latitudes;
        var lons = Longitudes;
        for (int i = 0; i < Length; i++)
        {
            var (latD, lonD) = GeoOps.KmToDegrees(radiusKm, lats[i]);
            result[i] = BoundingBox.FromPoint(new GeoPoint(lats[i], lons[i]), latD, lonD);
        }
        return result;
    }

    /// <summary>
    /// Compute the bounding box of all points.
    /// </summary>
    public BoundingBox Bounds()
    {
        if (Length == 0)
            return new BoundingBox(0, 0, 0, 0);
        var lats = Latitudes;
        var lons = Longitudes;
        double minLat = lats[0], maxLat = lats[0];
        double minLon = lons[0], maxLon = lons[0];
        for (int i = 1; i < Length; i++)
        {
            if (lats[i] < minLat) minLat = lats[i];
            if (lats[i] > maxLat) maxLat = lats[i];
            if (lons[i] < minLon) minLon = lons[i];
            if (lons[i] > maxLon) maxLon = lons[i];
        }
        return new BoundingBox(minLat, minLon, maxLat, maxLon);
    }

    /// <summary>Return the centroid (mean lat/lon) of all points.</summary>
    public GeoPoint Centroid()
    {
        double sumLat = 0, sumLon = 0;
        var lats = Latitudes;
        var lons = Longitudes;
        for (int i = 0; i < Length; i++)
        {
            sumLat += lats[i];
            sumLon += lons[i];
        }
        return new GeoPoint(sumLat / Length, sumLon / Length);
    }

    /// <summary>Bearing from each point to a target, in degrees (0-360).</summary>
    public Column<double> BearingTo(GeoPoint target)
    {
        var result = new double[Length];
        var lats = Latitudes;
        var lons = Longitudes;
        for (int i = 0; i < Length; i++)
            result[i] = GeoOps.Bearing(new GeoPoint(lats[i], lons[i]), target);
        return new Column<double>($"{Name}_bearing", result);
    }

    /// <summary>
    /// Check which points fall within a polygon. Returns a boolean mask.
    /// Uses bounding box pre-filter + ray-casting.
    /// </summary>
    public bool[] Within(GeoPolygon polygon) => polygon.Contains(this);

    /// <summary>Filter to a subset of points by index mask.</summary>
    public GeoColumn Filter(bool[] mask)
    {
        if (mask.Length != Length)
            throw new ArgumentException("Mask length must match column length.");
        int count = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;
        var lats = new double[count];
        var lons = new double[count];
        var srcLats = Latitudes;
        var srcLons = Longitudes;
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                lats[j] = srcLats[i];
                lons[j] = srcLons[i];
                j++;
            }
        }
        return new GeoColumn(Name, lats, lons);
    }

    /// <summary>Take rows at given indices.</summary>
    public GeoColumn TakeRows(int[] indices)
    {
        var lats = new double[indices.Length];
        var lons = new double[indices.Length];
        var srcLats = Latitudes;
        var srcLons = Longitudes;
        for (int i = 0; i < indices.Length; i++)
        {
            lats[i] = srcLats[indices[i]];
            lons[i] = srcLons[indices[i]];
        }
        return new GeoColumn(Name, lats, lons);
    }
}
