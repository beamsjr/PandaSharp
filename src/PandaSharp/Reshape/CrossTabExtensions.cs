#pragma warning disable CS8714 // nullable type as dictionary key
using PandaSharp.Column;

namespace PandaSharp.Reshape;

public static class CrossTabExtensions
{
    /// <summary>
    /// Compute a cross-tabulation (frequency table) of two columns.
    /// Returns a DataFrame where rows are unique values of col1 and columns are unique values of col2.
    /// </summary>
    public static DataFrame CrossTab(this DataFrame df, string rowColumn, string colColumn)
    {
        var rowCol = df[rowColumn];
        var colCol = df[colColumn];

        // Get unique values with O(1) lookup
        var rowValues = GetUnique(rowCol);
        var colValues = GetUnique(colCol);
        var rowMap = new Dictionary<object?, int>(new ObjComparer());
        for (int i = 0; i < rowValues.Count; i++) rowMap[rowValues[i]] = i;
        var colMap = new Dictionary<object?, int>(new ObjComparer());
        for (int i = 0; i < colValues.Count; i++) colMap[colValues[i]] = i;

        // Count occurrences
        var counts = new Dictionary<(int, int), int>();
        for (int i = 0; i < df.RowCount; i++)
        {
            var rv = rowCol.GetObject(i);
            var cv = colCol.GetObject(i);
            if (!rowMap.TryGetValue(rv, out int ri) || !colMap.TryGetValue(cv, out int ci)) continue;
            var key = (ri, ci);
            counts[key] = counts.GetValueOrDefault(key) + 1;
        }

        // Build result
        var columns = new List<IColumn>();
        columns.Add(new StringColumn(rowColumn, rowValues.Select(v => v?.ToString()).ToArray()));

        for (int c = 0; c < colValues.Count; c++)
        {
            var vals = new int[rowValues.Count];
            for (int r = 0; r < rowValues.Count; r++)
                vals[r] = counts.GetValueOrDefault((r, c));
            columns.Add(new Column<int>(colValues[c]?.ToString() ?? "null", vals));
        }

        return new DataFrame(columns);
    }

    private static List<object?> GetUnique(IColumn col)
    {
        var result = new List<object?>();
        var seen = new HashSet<object?>(new ObjComparer());
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            if (seen.Add(val)) result.Add(val);
        }
        return result;
    }

    private class ObjComparer : IEqualityComparer<object?>
    {
        public new bool Equals(object? x, object? y) => object.Equals(x, y);
        public int GetHashCode(object? obj) => obj?.GetHashCode() ?? 0;
    }
}
