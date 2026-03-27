using System.Numerics;
using System.Runtime.InteropServices;
using Cortex.Column;

namespace Cortex;

/// <summary>
/// Extension providing Compare — returns a DataFrame showing only differing cells
/// between two DataFrames, with {col}_self and {col}_other output columns.
/// </summary>
public static class DataFrameCompareExtensions
{
    /// <summary>
    /// Compare two DataFrames element-wise. Returns a DataFrame with columns
    /// <c>{col}_self</c> and <c>{col}_other</c> for each column that has differences.
    /// Rows where all compared columns are equal are excluded.
    /// </summary>
    /// <param name="self">The first DataFrame.</param>
    /// <param name="other">The second DataFrame to compare against.</param>
    /// <param name="alignColumn">Optional column name to include as-is for row identification.</param>
    public static DataFrame Compare(this DataFrame self, DataFrame other, string? alignColumn = null)
    {
        int rowCount = Math.Min(self.RowCount, other.RowCount);

        // Find common columns (excluding alignColumn)
        var commonColumns = new List<string>();
        foreach (var name in self.ColumnNames)
        {
            if (name == alignColumn) continue;
            if (other.ColumnNames.Contains(name))
                commonColumns.Add(name);
        }

        // Build per-column diff masks and count differing rows in a single pass
        var columnDiffMasks = new Dictionary<string, bool[]>();
        int diffCount = 0;
        // Track which rows have at least one diff using a simple bool array built incrementally
        var rowHasDiff = new bool[rowCount];

        foreach (var name in commonColumns)
        {
            var leftCol = self[name];
            var rightCol = other[name];
            var diffMask = BuildDiffMask(leftCol, rightCol, rowCount);
            columnDiffMasks[name] = diffMask;

            for (int i = 0; i < rowCount; i++)
            {
                if (diffMask[i] && !rowHasDiff[i])
                {
                    rowHasDiff[i] = true;
                    diffCount++;
                }
            }
        }

        // Also flag extra rows in the longer DataFrame
        int maxRowCount = Math.Max(self.RowCount, other.RowCount);
        for (int i = rowCount; i < maxRowCount; i++)
        {
            diffCount++;
        }

        if (diffCount == 0)
            return new DataFrame(Array.Empty<IColumn>());

        // Build index of differing rows
        var diffIndices = new int[diffCount];
        int idx = 0;
        for (int i = 0; i < rowCount; i++)
            if (rowHasDiff[i]) diffIndices[idx++] = i;
        for (int i = rowCount; i < maxRowCount; i++)
            diffIndices[idx++] = i;

        // Build result columns
        var resultColumns = new List<IColumn>();

        // Always include row index column
        resultColumns.Add(new Column<int>("row", diffIndices));

        // Precompute filtered index arrays once before the column loop
        var selfIndices = diffIndices.Where(i => i < self.RowCount).ToArray();
        var otherIndices = diffIndices.Where(i => i < other.RowCount).ToArray();

        // Include align column if specified
        if (alignColumn is not null && self.ColumnNames.Contains(alignColumn))
        {
            resultColumns.Add(self[alignColumn].TakeRows(selfIndices));
        }

        foreach (var name in commonColumns)
        {
            // Only include columns that actually have differences
            var colDiff = columnDiffMasks[name];
            bool hasDiff = false;
            for (int i = 0; i < rowCount; i++)
                if (colDiff[i]) { hasDiff = true; break; }

            if (!hasDiff) continue;

            // Build string columns for maximum compatibility (handles extra rows gracefully)
            var selfVals = new string?[diffCount];
            var otherVals = new string?[diffCount];
            for (int i = 0; i < diffCount; i++)
            {
                int r = diffIndices[i];
                selfVals[i] = r < self.RowCount ? self[name].GetObject(r)?.ToString() : null;
                otherVals[i] = r < other.RowCount ? other[name].GetObject(r)?.ToString() : null;
            }
            IColumn selfCol = new StringColumn($"{name}_self", selfVals);
            IColumn otherCol = new StringColumn($"{name}_other", otherVals);
            resultColumns.Add(selfCol);
            resultColumns.Add(otherCol);
        }

        return new DataFrame(resultColumns);
    }

    private static bool[] BuildDiffMask(IColumn left, IColumn right, int rowCount)
    {
        // Try SIMD path for numeric types
        if (left is Column<int> li && right is Column<int> ri)
            return BuildNumericDiffMask(li, ri, rowCount);
        if (left is Column<long> ll && right is Column<long> rl)
            return BuildNumericDiffMask(ll, rl, rowCount);
        if (left is Column<double> ld && right is Column<double> rd)
            return BuildDoubleDiffMask(ld, rd, rowCount);
        if (left is Column<float> lf && right is Column<float> rf)
            return BuildFloatDiffMask(lf, rf, rowCount);

        // String path
        if (left is StringColumn ls && right is StringColumn rs)
            return BuildStringDiffMask(ls, rs, rowCount);

        // Generic fallback
        return BuildGenericDiffMask(left, right, rowCount);
    }

    private static bool[] BuildNumericDiffMask<T>(Column<T> left, Column<T> right, int rowCount) where T : struct
    {
        var mask = new bool[rowCount];
        var lSpan = left.Values;
        var rSpan = right.Values;

        for (int i = 0; i < rowCount; i++)
        {
            bool lNull = left.IsNull(i);
            bool rNull = right.IsNull(i);
            if (lNull != rNull)
                mask[i] = true;
            else if (!lNull && !EqualityComparer<T>.Default.Equals(lSpan[i], rSpan[i]))
                mask[i] = true;
        }

        return mask;
    }

    private static bool[] BuildDoubleDiffMask(Column<double> left, Column<double> right, int rowCount)
    {
        var mask = new bool[rowCount];
        var lSpan = left.Values;
        var rSpan = right.Values;

        // SIMD path when no nulls
        if (left.NullCount == 0 && right.NullCount == 0)
        {
            int vecSize = Vector<double>.Count;
            int i = 0;

            // Process SIMD-width chunks
            for (; i + vecSize <= rowCount; i += vecSize)
            {
                var lVec = new Vector<double>(lSpan.Slice(i, vecSize));
                var rVec = new Vector<double>(rSpan.Slice(i, vecSize));
                var eq = Vector.Equals(lVec, rVec);

                // If all equal, skip
                if (eq == Vector<long>.AllBitsSet) continue;

                // Mark differing elements
                for (int j = 0; j < vecSize; j++)
                    mask[i + j] = lSpan[i + j] != rSpan[i + j];
            }

            // Remainder
            for (; i < rowCount; i++)
                mask[i] = lSpan[i] != rSpan[i];

            return mask;
        }

        // Fallback with null checks
        for (int i = 0; i < rowCount; i++)
        {
            bool lNull = left.IsNull(i);
            bool rNull = right.IsNull(i);
            if (lNull != rNull)
                mask[i] = true;
            else if (!lNull && lSpan[i] != rSpan[i])
                mask[i] = true;
        }
        return mask;
    }

    private static bool[] BuildFloatDiffMask(Column<float> left, Column<float> right, int rowCount)
    {
        var mask = new bool[rowCount];
        var lSpan = left.Values;
        var rSpan = right.Values;

        if (left.NullCount == 0 && right.NullCount == 0)
        {
            int vecSize = Vector<float>.Count;
            int i = 0;

            for (; i + vecSize <= rowCount; i += vecSize)
            {
                var lVec = new Vector<float>(lSpan.Slice(i, vecSize));
                var rVec = new Vector<float>(rSpan.Slice(i, vecSize));
                var eq = Vector.Equals(lVec, rVec);

                if (eq == Vector<int>.AllBitsSet) continue;

                for (int j = 0; j < vecSize; j++)
                    mask[i + j] = lSpan[i + j] != rSpan[i + j];
            }

            for (; i < rowCount; i++)
                mask[i] = lSpan[i] != rSpan[i];

            return mask;
        }

        for (int i = 0; i < rowCount; i++)
        {
            bool lNull = left.IsNull(i);
            bool rNull = right.IsNull(i);
            if (lNull != rNull)
                mask[i] = true;
            else if (!lNull && lSpan[i] != rSpan[i])
                mask[i] = true;
        }
        return mask;
    }

    private static bool[] BuildStringDiffMask(StringColumn left, StringColumn right, int rowCount)
    {
        var mask = new bool[rowCount];
        var lVals = left.GetValues();
        var rVals = right.GetValues();

        for (int i = 0; i < rowCount; i++)
        {
            var l = lVals[i];
            var r = rVals[i];
            if (l is null && r is null) continue;
            if (l is null || r is null) { mask[i] = true; continue; }
            if (!MemoryExtensions.Equals(l.AsSpan(), r.AsSpan(), StringComparison.Ordinal))
                mask[i] = true;
        }

        return mask;
    }

    private static bool[] BuildGenericDiffMask(IColumn left, IColumn right, int rowCount)
    {
        var mask = new bool[rowCount];
        for (int i = 0; i < rowCount; i++)
        {
            var l = left.GetObject(i);
            var r = right.GetObject(i);
            if (!Equals(l, r)) mask[i] = true;
        }
        return mask;
    }
}
