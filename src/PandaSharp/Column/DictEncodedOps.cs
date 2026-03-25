namespace PandaSharp.Column;

/// <summary>
/// Dictionary encoding for string columns — stores K unique strings + N int codes.
/// Enables O(K) string operations instead of O(N) where K &lt;&lt; N.
/// For 14.7M rows with 6K unique tickers: 2000x less string processing.
///
/// This is the same optimization pandas/Arrow use for categorical/dictionary types.
/// </summary>
internal class DictEncoding
{
    public string[] Uniques { get; }
    public int[] Codes { get; }

    private DictEncoding(string[] uniques, int[] codes)
    {
        Uniques = uniques;
        Codes = codes;
    }

    /// <summary>Build dictionary encoding from a string column.</summary>
    public static DictEncoding Encode(StringColumn col)
    {
        var vals = col.GetValues();
        int n = col.Length;
        var codes = new int[n];

        // Use capacity hint based on expected cardinality
        // For stock data: Ticker ~6K uniques, Date ~5K uniques from 14.7M rows
        int initialCapacity = Math.Min(n, 65536);
        var uniqueMap = new Dictionary<string, int>(initialCapacity, StringComparer.Ordinal);
        var uniqueList = new List<string>(initialCapacity);

        for (int i = 0; i < n; i++)
        {
            var s = vals[i] ?? "";
            ref var slot = ref System.Runtime.InteropServices.CollectionsMarshal.GetValueRefOrAddDefault(uniqueMap, s, out bool exists);
            if (!exists)
            {
                slot = uniqueList.Count;
                uniqueList.Add(s);
            }
            codes[i] = slot;
        }

        return new DictEncoding(uniqueList.ToArray(), codes);
    }

    /// <summary>Apply a string transform to unique values, then rebuild column.</summary>
    public StringColumn TransformUniques(string name, Func<string, string> transform)
    {
        var transformedUniques = new string[Uniques.Length];
        for (int i = 0; i < Uniques.Length; i++)
            transformedUniques[i] = transform(Uniques[i]);

        // Rebuild string array from transformed uniques + codes (parallel for large data)
        var result = RebuildFromCodes(transformedUniques);
        return StringColumn.CreateOwnedNoNulls(name, result!);
    }

    /// <summary>Rebuild an array from lookup table + codes, parallelized for large data.</summary>
    private T[] RebuildFromCodes<T>(T[] uniqueValues)
    {
        int n = Codes.Length;
        var result = new T[n];
        var codes = Codes;
        if (n > 1_000_000)
        {
            int nThreads = Math.Min(Environment.ProcessorCount, 4);
            Parallel.For(0, nThreads, t =>
            {
                int start = (int)((long)n * t / nThreads);
                int end = (int)((long)n * (t + 1) / nThreads);
                for (int j = start; j < end; j++)
                    result[j] = uniqueValues[codes[j]];
            });
        }
        else
        {
            for (int j = 0; j < n; j++)
                result[j] = uniqueValues[codes[j]];
        }
        return result;
    }

    /// <summary>Check which uniques contain substring, map back to rows.</summary>
    public bool[] Contains(string substring)
    {
        var uniqueMatch = new bool[Uniques.Length];
        for (int i = 0; i < Uniques.Length; i++)
            uniqueMatch[i] = Uniques[i].Contains(substring);
        return RebuildFromCodes(uniqueMatch);
    }

    /// <summary>Check which uniques start with prefix, map back to rows.</summary>
    public bool[] StartsWith(string prefix)
    {
        var uniqueMatch = new bool[Uniques.Length];
        for (int i = 0; i < Uniques.Length; i++)
            uniqueMatch[i] = Uniques[i].StartsWith(prefix, StringComparison.Ordinal);
        return RebuildFromCodes(uniqueMatch);
    }

    /// <summary>Get string length per row (compute K lengths, map to N rows).</summary>
    public Column<int> Len(string name)
    {
        var uniqueLens = new int[Uniques.Length];
        for (int i = 0; i < Uniques.Length; i++)
            uniqueLens[i] = Uniques[i].Length;
        return new Column<int>(name, RebuildFromCodes(uniqueLens));
    }

    /// <summary>Replace in unique values, rebuild column.</summary>
    public StringColumn Replace(string name, string oldValue, string newValue)
    {
        return TransformUniques(name, s => s.Replace(oldValue, newValue));
    }

    /// <summary>Slice unique values, rebuild column.</summary>
    public StringColumn Slice(string name, int start, int length)
    {
        return TransformUniques(name, s =>
            start < s.Length ? s.Substring(start, Math.Min(length, s.Length - start)) : "");
    }
}
