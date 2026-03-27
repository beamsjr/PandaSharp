using Cortex.Column;

namespace Cortex.Statistics;

public static class ValueCountsExtensions
{
    /// <summary>
    /// Returns a DataFrame with unique values and their counts, sorted by count descending.
    /// </summary>
    public static DataFrame ValueCounts(this IColumn column)
    {
        // Fast path: StringColumn with dict encoding
        if (column is StringColumn sc)
        {
            var (codes, uniques) = sc.GetDictCodes();
            var countArr = new int[uniques.Length];
            int nullCount2 = 0;
            for (int i = 0; i < codes.Length; i++)
            {
                if (codes[i] < 0) nullCount2++;
                else countArr[codes[i]]++;
            }

            // Sort by count descending
            var order = Enumerable.Range(0, uniques.Length).ToArray();
            Array.Sort(order, (a, b) => countArr[b].CompareTo(countArr[a]));

            int totalEntries = uniques.Length + (nullCount2 > 0 ? 1 : 0);
            var sortedNames = new string?[totalEntries];
            var sortedCounts = new int[totalEntries];
            for (int i = 0; i < order.Length; i++)
            {
                sortedNames[i] = uniques[order[i]];
                sortedCounts[i] = countArr[order[i]];
            }

            if (nullCount2 > 0)
            {
                sortedNames[uniques.Length] = null;
                sortedCounts[uniques.Length] = nullCount2;
            }

            return new DataFrame(
                new StringColumn(column.Name, sortedNames),
                new Column<int>("count", sortedCounts));
        }

        // Generic path
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
            new Column<int>("count", countValues));
    }

    /// <summary>
    /// Returns the number of unique non-null values.
    /// </summary>
    public static int NUnique(this IColumn column)
    {
        // Fast path: StringColumn with dict encoding
        if (column is StringColumn sc)
        {
            var (_, uniques) = sc.GetDictCodes();
            // Nulls use sentinel code -1 and are not in Uniques array
            return uniques.Length;
        }

        var unique = new HashSet<object>();
        for (int i = 0; i < column.Length; i++)
        {
            var val = column.GetObject(i);
            if (val is not null) unique.Add(val);
        }
        return unique.Count;
    }
}
