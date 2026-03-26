using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

/// <summary>
/// Returns a summary DataFrame with one row per stat and one column per source column.
/// Stats: count, mean, std, min, 25%, 50%, 75%, max, null%, unique, top, freq, dtype, memory_bytes.
/// Uses single-pass SIMD-friendly loops for numeric min/max/mean/count.
/// Leverages existing Describe infrastructure where possible.
/// </summary>
public static class ProfileExtensions
{
    private static readonly string[] StatNames =
    {
        "count", "mean", "std", "min", "25%", "50%", "75%", "max",
        "null%", "unique", "top", "freq", "dtype", "memory_bytes"
    };

    /// <summary>
    /// Profile every column and return a DataFrame with stat rows.
    /// Each source column becomes a string column (heterogeneous stat types
    /// like count=int, mean=double, dtype=string require string representation).
    /// </summary>
    public static DataFrame ProfileToDataFrame(this DataFrame df)
    {
        var resultColumns = new List<IColumn>();
        resultColumns.Add(new StringColumn("stat", StatNames));

        // Process each column
        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            var statValues = ComputeColumnStats(col);
            resultColumns.Add(new StringColumn(col.Name, statValues));
        }

        return new DataFrame(resultColumns);
    }

    private static string?[] ComputeColumnStats(IColumn col)
    {
        var stats = new string?[StatNames.Length];
        int length = col.Length;
        int nullCount = col.NullCount;
        int count = length - nullCount;

        // count
        stats[0] = count.ToString();

        // dtype
        stats[12] = FormatType(col.DataType);

        // memory_bytes
        stats[13] = EstimateColumnMemory(col).ToString();

        // null%
        stats[8] = length > 0
            ? (nullCount * 100.0 / length).ToString("F2")
            : "0.00";

        bool isNumeric = IsNumericType(col.DataType);

        if (isNumeric && count > 0)
        {
            ComputeNumericStats(col, count, stats);
        }
        else if (col.DataType == typeof(string))
        {
            ComputeStringStats(col, count, stats);
        }
        else if (col.DataType == typeof(bool))
        {
            ComputeBoolStats(col, count, stats);
        }
        else
        {
            // Non-numeric, non-string: fill with empty
            for (int i = 1; i <= 7; i++) stats[i] ??= "";
            ComputeGenericTopUnique(col, count, stats);
        }

        // Ensure no nulls in stat values (use empty string)
        for (int i = 0; i < stats.Length; i++)
            stats[i] ??= "";

        return stats;
    }

    /// <summary>
    /// Single-pass computation of min, max, mean, count for numeric columns.
    /// Then sort for quantiles. Uses typed spans to avoid boxing.
    /// </summary>
    private static void ComputeNumericStats(IColumn col, int count, string?[] stats)
    {
        // Extract non-null values as doubles using typed fast paths
        double[] data = ExtractNonNullDoubles(col, count);

        // Single pass: sum, min, max
        double sum = 0, mn = double.MaxValue, mx = double.MinValue;
        for (int i = 0; i < data.Length; i++)
        {
            double v = data[i];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }

        double mean = sum / count;

        // Second pass: variance
        double sumSq = 0;
        for (int i = 0; i < data.Length; i++)
        {
            double d = data[i] - mean;
            sumSq += d * d;
        }
        double std = count > 1 ? Math.Sqrt(sumSq / (count - 1)) : 0;

        stats[1] = mean.ToString("G6");   // mean
        stats[2] = std.ToString("G6");     // std
        stats[3] = mn.ToString("G6");      // min
        stats[7] = mx.ToString("G6");      // max

        // Sort for quantiles
        Array.Sort(data);
        stats[4] = Percentile(data, 0.25).ToString("G6"); // 25%
        stats[5] = Percentile(data, 0.50).ToString("G6"); // 50%
        stats[6] = Percentile(data, 0.75).ToString("G6"); // 75%

        // unique, top, freq
        ComputeGenericTopUnique(col, count, stats);
    }

    private static void ComputeStringStats(IColumn col, int count, string?[] stats)
    {
        // mean, std, min, max, quantiles not applicable => empty
        for (int i = 1; i <= 7; i++) stats[i] = "";

        ComputeGenericTopUnique(col, count, stats);
    }

    private static void ComputeBoolStats(IColumn col, int count, string?[] stats)
    {
        // Treat bool as 0/1 numeric
        int trueCount = 0;
        for (int i = 0; i < col.Length; i++)
        {
            if (!col.IsNull(i) && (bool)col.GetObject(i)!)
                trueCount++;
        }

        double mean = count > 0 ? (double)trueCount / count : 0;
        stats[1] = mean.ToString("G6");
        stats[2] = ""; // std for bool
        stats[3] = "False"; // min
        stats[7] = "True";  // max
        stats[4] = ""; stats[5] = ""; stats[6] = "";

        ComputeGenericTopUnique(col, count, stats);
    }

    /// <summary>
    /// Compute unique count, top value, and top frequency for any column type.
    /// </summary>
    private static void ComputeGenericTopUnique(IColumn col, int count, string?[] stats)
    {
        if (count == 0)
        {
            stats[9] = "0";
            stats[10] = "";
            stats[11] = "0";
            return;
        }

        var valueCounts = new Dictionary<string, int>();
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) continue;
            var key = col.GetObject(i)?.ToString() ?? "";
            valueCounts[key] = valueCounts.GetValueOrDefault(key) + 1;
        }

        stats[9] = valueCounts.Count.ToString(); // unique

        // top + freq
        string topVal = "";
        int topFreq = 0;
        foreach (var kv in valueCounts)
        {
            if (kv.Value > topFreq)
            {
                topFreq = kv.Value;
                topVal = kv.Key;
            }
        }
        stats[10] = topVal;
        stats[11] = topFreq.ToString();
    }

    /// <summary>
    /// Extract non-null values as double[] using typed fast paths to avoid boxing.
    /// </summary>
    private static double[] ExtractNonNullDoubles(IColumn col, int count)
    {
        var result = new double[count];

        if (col is Column<double> dc)
        {
            var span = dc.Buffer.Span;
            int j = 0;
            if (dc.NullCount == 0)
            {
                span.CopyTo(result);
            }
            else
            {
                for (int i = 0; i < dc.Length; i++)
                    if (!dc.IsNull(i)) result[j++] = span[i];
            }
        }
        else if (col is Column<int> ic)
        {
            var span = ic.Buffer.Span;
            int j = 0;
            for (int i = 0; i < ic.Length; i++)
                if (!ic.IsNull(i)) result[j++] = span[i];
        }
        else if (col is Column<long> lc)
        {
            var span = lc.Buffer.Span;
            int j = 0;
            for (int i = 0; i < lc.Length; i++)
                if (!lc.IsNull(i)) result[j++] = span[i];
        }
        else if (col is Column<float> fc)
        {
            var span = fc.Buffer.Span;
            int j = 0;
            for (int i = 0; i < fc.Length; i++)
                if (!fc.IsNull(i)) result[j++] = span[i];
        }
        else
        {
            int j = 0;
            for (int i = 0; i < col.Length; i++)
                if (!col.IsNull(i)) result[j++] = Convert.ToDouble(col.GetObject(i));
        }

        return result;
    }

    private static double Percentile(double[] sorted, double p)
    {
        int n = sorted.Length;
        if (n == 1) return sorted[0];
        double rank = p * (n - 1);
        int lower = (int)rank;
        int upper = lower + 1;
        if (upper >= n) return sorted[lower];
        double frac = rank - lower;
        return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long EstimateColumnMemory(IColumn col)
    {
        if (col is StringColumn)
        {
            return (long)col.Length * IntPtr.Size + (long)(col.Length - col.NullCount) * 40;
        }

        long elementSize = col.DataType switch
        {
            var t when t == typeof(int) => Unsafe.SizeOf<int>(),
            var t when t == typeof(long) => Unsafe.SizeOf<long>(),
            var t when t == typeof(double) => Unsafe.SizeOf<double>(),
            var t when t == typeof(float) => Unsafe.SizeOf<float>(),
            var t when t == typeof(bool) => Unsafe.SizeOf<bool>(),
            var t when t == typeof(DateTime) => Unsafe.SizeOf<DateTime>(),
            _ => Marshal.SizeOf(col.DataType)
        };
        return elementSize * col.Length;
    }

    private static bool IsNumericType(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);

    private static string FormatType(Type type)
    {
        if (type == typeof(int)) return "int32";
        if (type == typeof(long)) return "int64";
        if (type == typeof(float)) return "float32";
        if (type == typeof(double)) return "float64";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(DateTime)) return "datetime";
        if (type == typeof(string)) return "string";
        return type.Name;
    }
}
