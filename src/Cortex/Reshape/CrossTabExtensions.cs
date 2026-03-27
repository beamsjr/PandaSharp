#pragma warning disable CS8714 // nullable type as dictionary key
using Cortex.Column;

namespace Cortex.Reshape;

public static class CrossTabExtensions
{
    /// <summary>
    /// Compute a cross-tabulation (frequency table) of two columns.
    /// Returns a DataFrame where rows are unique values of col1 and columns are unique values of col2.
    /// Uses flat array indexing instead of dictionary for O(1) counting.
    /// </summary>
    public static DataFrame CrossTab(this DataFrame df, string rowColumn, string colColumn)
    {
        var rowCol = df[rowColumn];
        var colCol = df[colColumn];

        // Typed fast path: both StringColumns
        if (rowCol is StringColumn rowSc && colCol is StringColumn colSc)
            return CrossTabString(rowSc, colSc, rowColumn);

        // Fallback: generic path
        var rowValues = GetUnique(rowCol);
        var colValues = GetUnique(colCol);
        var rowMap = new Dictionary<object?, int>(new ObjComparer());
        for (int i = 0; i < rowValues.Count; i++) rowMap[rowValues[i]] = i;
        var colMap = new Dictionary<object?, int>(new ObjComparer());
        for (int i = 0; i < colValues.Count; i++) colMap[colValues[i]] = i;

        // Flat array counting: counts[ri * colCount + ci]
        int rowCount = rowValues.Count;
        int colCount = colValues.Count;
        var counts = new int[rowCount * colCount];

        for (int i = 0; i < df.RowCount; i++)
        {
            var rv = rowCol.GetObject(i);
            var cv = colCol.GetObject(i);
            if (rowMap.TryGetValue(rv, out int ri) && colMap.TryGetValue(cv, out int ci))
                counts[ri * colCount + ci]++;
        }

        var columns = new List<IColumn>();
        columns.Add(new StringColumn(rowColumn, rowValues.Select(v => v?.ToString()).ToArray()));

        for (int c = 0; c < colCount; c++)
        {
            var vals = new int[rowCount];
            for (int r = 0; r < rowCount; r++)
                vals[r] = counts[r * colCount + c];
            columns.Add(new Column<int>(colValues[c]?.ToString() ?? "null", vals));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Fast CrossTab for two StringColumns — avoids all boxing.
    /// </summary>
    private static DataFrame CrossTabString(StringColumn rowCol, StringColumn colCol, string rowColumnName)
    {
        var rowVals = rowCol.GetValues();
        var colVals = colCol.GetValues();
        int n = rowCol.Length;

        // Use a sentinel string for null so that null and "" are distinct
        const string NullSentinel = "\0__NULL__";

        // Build unique maps with ordinal comparison
        var rowMap = new Dictionary<string, int>(StringComparer.Ordinal);
        var rowOrder = new List<string>();
        var colMap = new Dictionary<string, int>(StringComparer.Ordinal);
        var colOrder = new List<string>();

        for (int i = 0; i < n; i++)
        {
            var rv = rowVals[i] ?? NullSentinel;
            var cv = colVals[i] ?? NullSentinel;
            if (!rowMap.ContainsKey(rv)) { rowMap[rv] = rowOrder.Count; rowOrder.Add(rv); }
            if (!colMap.ContainsKey(cv)) { colMap[cv] = colOrder.Count; colOrder.Add(cv); }
        }

        // Flat array counting
        int nRowUniques = rowOrder.Count;
        int nColUniques = colOrder.Count;
        var counts = new int[nRowUniques * nColUniques];

        for (int i = 0; i < n; i++)
        {
            var rv = rowVals[i] ?? NullSentinel;
            var cv = colVals[i] ?? NullSentinel;
            int ri = rowMap[rv];
            int ci = colMap[cv];
            counts[ri * nColUniques + ci]++;
        }

        var columns = new List<IColumn>();
        // Map sentinel back to a display name for the output
        columns.Add(new StringColumn(rowColumnName,
            rowOrder.Select(s => s == NullSentinel ? (string?)null : s).ToArray()));

        for (int c = 0; c < nColUniques; c++)
        {
            var vals = new int[nRowUniques];
            for (int r = 0; r < nRowUniques; r++)
                vals[r] = counts[r * nColUniques + c];
            var colName = colOrder[c] == NullSentinel ? "null" : colOrder[c];
            columns.Add(new Column<int>(colName, vals));
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
