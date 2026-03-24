using PandaSharp.Column;

namespace PandaSharp.Compare;

/// <summary>
/// DataFrame diff/comparison operations.
/// </summary>
public static class CompareExtensions
{
    /// <summary>
    /// Compare two DataFrames and return a DataFrame showing differences.
    /// Only compares columns present in both DataFrames.
    /// Returns rows where at least one value differs, with columns showing "self" and "other" values.
    /// </summary>
    public static DataFrame Compare(this DataFrame self, DataFrame other)
    {
        var commonCols = self.ColumnNames.Intersect(other.ColumnNames).ToList();
        int rowCount = Math.Min(self.RowCount, other.RowCount);

        var diffRows = new List<int>();
        for (int r = 0; r < rowCount; r++)
        {
            bool differs = false;
            foreach (var col in commonCols)
            {
                var selfVal = self[col].GetObject(r);
                var otherVal = other[col].GetObject(r);
                if (!Equals(selfVal, otherVal)) { differs = true; break; }
            }
            if (differs) diffRows.Add(r);
        }

        // Also flag rows only in one side
        for (int r = rowCount; r < self.RowCount; r++) diffRows.Add(r);

        var resultCols = new List<IColumn>();
        resultCols.Add(new Column<int>("row", diffRows.ToArray()));

        foreach (var colName in commonCols)
        {
            var selfVals = new string?[diffRows.Count];
            var otherVals = new string?[diffRows.Count];
            for (int i = 0; i < diffRows.Count; i++)
            {
                int r = diffRows[i];
                selfVals[i] = r < self.RowCount ? self[colName].GetObject(r)?.ToString() : null;
                otherVals[i] = r < other.RowCount ? other[colName].GetObject(r)?.ToString() : null;
            }
            resultCols.Add(new StringColumn($"{colName}_self", selfVals));
            resultCols.Add(new StringColumn($"{colName}_other", otherVals));
        }

        return new DataFrame(resultCols);
    }
}
