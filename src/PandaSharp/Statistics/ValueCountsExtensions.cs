using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class ValueCountsExtensions
{
    /// <summary>
    /// Returns a DataFrame with unique values and their counts, sorted by count descending.
    /// </summary>
    public static DataFrame ValueCounts(this IColumn column)
    {
        var counts = new Dictionary<object, int>();
        int nullCount = 0;

        for (int i = 0; i < column.Length; i++)
        {
            var val = column.GetObject(i);
            if (val is null) { nullCount++; continue; }
            counts[val] = counts.GetValueOrDefault(val) + 1;
        }

        var sorted = counts.OrderByDescending(kv => kv.Value).ToList();

        var valueStrings = sorted.Select(kv => kv.Key.ToString()).ToArray();
        var countValues = sorted.Select(kv => kv.Value).ToArray();

        if (nullCount > 0)
        {
            valueStrings = [.. valueStrings, null];
            countValues = [.. countValues, nullCount];
        }

        return new DataFrame(
            new StringColumn(column.Name, valueStrings),
            new Column<int>("count", countValues)
        );
    }

    /// <summary>
    /// Returns the number of unique non-null values.
    /// </summary>
    public static int NUnique(this IColumn column)
    {
        var unique = new HashSet<object>();
        for (int i = 0; i < column.Length; i++)
        {
            var val = column.GetObject(i);
            if (val is not null) unique.Add(val);
        }
        return unique.Count;
    }
}
