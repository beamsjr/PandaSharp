using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.Geo;

/// <summary>
/// Spatial join operations: join two DataFrames based on geographic proximity.
/// </summary>
public static class SpatialJoin
{
    /// <summary>
    /// Nearest-neighbor spatial join: for each row in left, find the closest point in right.
    /// Uses R-tree index on the right side for O(n log m) performance.
    /// Returns a DataFrame with all left columns plus the nearest right row's columns + distance.
    /// </summary>
    public static DataFrame NearestJoin(
        DataFrame left, GeoColumn leftGeo,
        DataFrame right, GeoColumn rightGeo,
        double? maxDistanceKm = null,
        string distanceColumn = "distance_km",
        string suffix = "_right")
    {
        int leftRows = left.RowCount;
        var nearestIdx = new int[leftRows];
        var distances = new double[leftRows];

        // Build R-tree on right side for O(log m) nearest queries
        var tree = RTree.Build(rightGeo);

        for (int i = 0; i < leftRows; i++)
        {
            var (idx, dist) = tree.Nearest(leftGeo[i]);

            if (maxDistanceKm.HasValue && dist > maxDistanceKm.Value)
            {
                nearestIdx[i] = -1;
                distances[i] = double.NaN;
            }
            else
            {
                nearestIdx[i] = idx;
                distances[i] = dist;
            }
        }

        return BuildJoinResult(left, right, nearestIdx, distances, distanceColumn, suffix);
    }

    /// <summary>
    /// Within-distance join: for each left row, find all right rows within radiusKm.
    /// Uses R-tree index on right side for fast radius queries.
    /// Returns an expanded DataFrame (one row per match).
    /// </summary>
    public static DataFrame WithinJoin(
        DataFrame left, GeoColumn leftGeo,
        DataFrame right, GeoColumn rightGeo,
        double radiusKm,
        string distanceColumn = "distance_km",
        string suffix = "_right")
    {
        var leftIndices = new List<int>();
        var rightIndices = new List<int>();
        var dists = new List<double>();

        // Build R-tree on right side
        var tree = RTree.Build(rightGeo);

        for (int i = 0; i < left.RowCount; i++)
        {
            var matches = tree.QueryRadius(leftGeo[i], radiusKm);
            foreach (var (idx, dist) in matches)
            {
                leftIndices.Add(i);
                rightIndices.Add(idx);
                dists.Add(dist);
            }
        }

        return BuildExpandedJoinResult(left, right, leftIndices, rightIndices, dists, distanceColumn, suffix);
    }

    private static DataFrame BuildJoinResult(
        DataFrame left, DataFrame right,
        int[] nearestIdx, double[] distances,
        string distanceColumn, string suffix)
    {
        var columns = new List<IColumn>();
        var rightNames = new HashSet<string>(right.ColumnNames);
        var leftNames = new HashSet<string>(left.ColumnNames);

        // Left columns (all rows)
        foreach (var name in left.ColumnNames)
            columns.Add(left[name].Clone());

        // Right columns (nearest match per left row)
        foreach (var name in right.ColumnNames)
        {
            var col = right[name];
            string outputName = leftNames.Contains(name) ? name + suffix : name;

            var values = new object?[left.RowCount];
            for (int i = 0; i < left.RowCount; i++)
                values[i] = nearestIdx[i] >= 0 ? col.GetObject(nearestIdx[i]) : null;

            columns.Add(BuildColumnFromObjects(outputName, col.DataType, values));
        }

        // Distance column
        columns.Add(new Column<double>(distanceColumn, distances));

        return new DataFrame(columns);
    }

    private static DataFrame BuildExpandedJoinResult(
        DataFrame left, DataFrame right,
        List<int> leftIndices, List<int> rightIndices,
        List<double> distances,
        string distanceColumn, string suffix)
    {
        int resultRows = leftIndices.Count;
        if (resultRows == 0)
            return new DataFrame(new Column<double>(distanceColumn, Array.Empty<double>()));

        var columns = new List<IColumn>();
        var leftNames = new HashSet<string>(left.ColumnNames);
        var leftIdx = leftIndices.ToArray();
        var rightIdx = rightIndices.ToArray();

        foreach (var name in left.ColumnNames)
            columns.Add(left[name].TakeRows(leftIdx));

        foreach (var name in right.ColumnNames)
        {
            string outputName = leftNames.Contains(name) ? name + suffix : name;
            columns.Add(right[name].TakeRows(rightIdx).Clone(outputName));
        }

        columns.Add(new Column<double>(distanceColumn, distances.ToArray()));

        return new DataFrame(columns);
    }

    private static IColumn BuildColumnFromObjects(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(bool)) return BuildTyped<bool>(name, values);
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is T t ? t : null;
        return Column<T>.FromNullable(name, typed);
    }
}
