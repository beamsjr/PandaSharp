using System.Runtime.InteropServices;
using PandaSharp.Column;
using PandaSharp.Storage;

namespace PandaSharp;

/// <summary>
/// Extension providing CombineFirst — fills nulls in the primary DataFrame
/// with values from a fallback DataFrame, matched by column name.
/// </summary>
public static class DataFrameCombineExtensions
{
    /// <summary>
    /// For each column matched by name: use <paramref name="primary"/> value if non-null,
    /// else <paramref name="fallback"/> value. Unmatched columns from fallback are included as-is.
    /// Both DataFrames must have the same number of rows.
    /// </summary>
    public static DataFrame CombineFirst(this DataFrame primary, DataFrame fallback)
    {
        if (primary.RowCount != fallback.RowCount)
            throw new ArgumentException("DataFrames must have the same number of rows.");

        int rowCount = primary.RowCount;
        var resultColumns = new List<IColumn>();
        var usedFallbackColumns = new HashSet<string>();

        // Process columns from primary
        foreach (var name in primary.ColumnNames)
        {
            var leftCol = primary[name];

            if (!fallback.ColumnNames.Contains(name))
            {
                resultColumns.Add(leftCol);
                continue;
            }

            usedFallbackColumns.Add(name);
            var rightCol = fallback[name];

            // No nulls in primary column — take it as-is
            if (leftCol.NullCount == 0)
            {
                resultColumns.Add(leftCol);
                continue;
            }

            resultColumns.Add(CombineColumn(name, leftCol, rightCol, rowCount));
        }

        // Add unmatched columns from fallback
        foreach (var name in fallback.ColumnNames)
        {
            if (!usedFallbackColumns.Contains(name))
                resultColumns.Add(fallback[name]);
        }

        return new DataFrame(resultColumns);
    }

    private static IColumn CombineColumn(string name, IColumn left, IColumn right, int rowCount)
    {
        // Try typed numeric combination to avoid boxing
        if (left is Column<int> li && right is Column<int> ri)
            return CombineNumeric(name, li, ri, rowCount);
        if (left is Column<long> ll && right is Column<long> rl)
            return CombineNumeric(name, ll, rl, rowCount);
        if (left is Column<double> ld && right is Column<double> rd)
            return CombineNumeric(name, ld, rd, rowCount);
        if (left is Column<float> lf && right is Column<float> rf)
            return CombineNumeric(name, lf, rf, rowCount);
        if (left is Column<bool> lb && right is Column<bool> rb)
            return CombineNumeric(name, lb, rb, rowCount);
        if (left is Column<DateTime> ldt && right is Column<DateTime> rdt)
            return CombineNumeric(name, ldt, rdt, rowCount);

        // StringColumn path
        if (left is StringColumn ls && right is StringColumn rs)
            return CombineString(name, ls, rs, rowCount);

        // Generic fallback using boxing
        return CombineGeneric(name, left, right, rowCount);
    }

    /// <summary>
    /// For numeric Column&lt;T&gt;, operate on spans and null bitmaps directly — no boxing.
    /// </summary>
    private static Column<T> CombineNumeric<T>(string name, Column<T> left, Column<T> right, int rowCount) where T : struct
    {
        var leftSpan = left.Values;
        var rightSpan = right.Values;
        var result = new T?[rowCount];

        for (int i = 0; i < rowCount; i++)
        {
            if (!left.IsNull(i))
                result[i] = leftSpan[i];
            else if (!right.IsNull(i))
                result[i] = rightSpan[i];
            // else both null → result[i] stays null
        }

        return Column<T>.FromNullable(name, result);
    }

    private static StringColumn CombineString(string name, StringColumn left, StringColumn right, int rowCount)
    {
        var leftVals = left.GetValues();
        var rightVals = right.GetValues();
        var result = new string?[rowCount];

        for (int i = 0; i < rowCount; i++)
            result[i] = leftVals[i] ?? rightVals[i];

        return new StringColumn(name, result);
    }

    private static IColumn CombineGeneric(string name, IColumn left, IColumn right, int rowCount)
    {
        // Fallback: use GetObject (boxing) for unknown column types
        var result = new object?[rowCount];
        for (int i = 0; i < rowCount; i++)
            result[i] = left.IsNull(i) ? right.GetObject(i) : left.GetObject(i);

        // Reconstruct as double column (best generic option)
        var doubles = new double?[rowCount];
        for (int i = 0; i < rowCount; i++)
            doubles[i] = result[i] is not null ? Convert.ToDouble(result[i]) : null;
        return Column<double>.FromNullable(name, doubles);
    }
}
