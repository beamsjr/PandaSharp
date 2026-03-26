using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class DescribeExtensions
{
    /// <summary>
    /// Returns a summary DataFrame with count, mean, std, min, 25%, 50%, 75%, max per numeric column.
    /// Uses single-sort optimization: sorts once per column, computes all quantiles from sorted data.
    /// </summary>
    public static DataFrame Describe(this DataFrame df)
    {
        var statNames = new string[] { "count", "mean", "std", "min", "25%", "50%", "75%", "max" };
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("stat", statNames));

        // Process columns in parallel for wide DataFrames
        var numericCols = new List<(string Name, IColumn Col)>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            if (col is Column<double> || col is Column<int> || col is Column<long> || col is Column<float>)
                numericCols.Add((name, col));
        }

        var results = new double[numericCols.Count][];
        Parallel.For(0, numericCols.Count, ci =>
            results[ci] = DescribeFast(numericCols[ci].Col));

        for (int ci = 0; ci < numericCols.Count; ci++)
            columns.Add(new Column<double>(numericCols[ci].Name, results[ci]));

        return new DataFrame(columns);
    }

    private static double[] DescribeFast(IColumn col)
    {
        if (col is Column<double> dc) return DescribeDouble(dc);
        if (col is Column<int> ic) return DescribeViaDouble(ic);
        if (col is Column<long> lc) return DescribeViaDouble(lc);
        if (col is Column<float> fc) return DescribeViaDouble(fc);
        return [0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];
    }

    /// <summary>
    /// Fast describe for Column<double>: single sort, compute all stats from sorted array.
    /// count + mean + std in one pass, then sort once for min/quantiles/max.
    /// </summary>
    private static double[] DescribeDouble(Column<double> col)
    {
        int n = col.Length;
        int count = n - col.NullCount;
        if (count == 0) return [0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];

        // Compact non-null values (null positions have default 0.0 which would corrupt stats)
        var span = col.Buffer.Span;
        double[] data;
        if (col.NullCount == 0)
        {
            data = span.ToArray();
        }
        else
        {
            data = new double[count];
            int j = 0;
            for (int i = 0; i < n; i++)
                if (!col.IsNull(i)) data[j++] = span[i];
        }

        // Pass 1: mean + std + min + max from compacted data
        double sum = 0, mn = double.MaxValue, mx = double.MinValue;
        bool allNaN = true;
        for (int i = 0; i < count; i++)
        {
            double v = data[i];
            if (double.IsNaN(v)) continue;
            allNaN = false;
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (allNaN) return [count, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];
        double mean = sum / count;
        double sumSq = 0;
        for (int i = 0; i < count; i++) { double d = data[i] - mean; sumSq += d * d; }
        double std = count > 1 ? Math.Sqrt(sumSq / (count - 1)) : 0;

        // O(n) quantiles using successive quickselect on compacted data
        int k25 = Math.Min((int)(0.25 * (count - 1)), count - 1);
        int k50 = Math.Min((int)(0.50 * (count - 1)), count - 1);
        int k75 = Math.Min((int)(0.75 * (count - 1)), count - 1);
        QuickSelect(data, 0, count - 1, k25);
        double q25 = data[k25];
        QuickSelect(data, k25, count - 1, k50);
        double q50 = data[k50];
        QuickSelect(data, k50, count - 1, k75);
        double q75 = data[k75];

        return [count, mean, std, mn, q25, q50, q75, mx];
    }

    private static void QuickSelect(double[] arr, int lo, int hi, int k)
    {
        while (lo < hi)
        {
            // Median-of-3 pivot
            int mid = lo + (hi - lo) / 2;
            if (arr[mid] < arr[lo]) (arr[lo], arr[mid]) = (arr[mid], arr[lo]);
            if (arr[hi] < arr[lo]) (arr[lo], arr[hi]) = (arr[hi], arr[lo]);
            if (arr[mid] < arr[hi]) (arr[mid], arr[hi]) = (arr[hi], arr[mid]);
            double pivot = arr[hi];

            int store = lo;
            for (int i = lo; i < hi; i++)
                if (arr[i] < pivot) { (arr[store], arr[i]) = (arr[i], arr[store]); store++; }
            (arr[store], arr[hi]) = (arr[hi], arr[store]);

            if (store == k) return;
            if (store < k) lo = store + 1;
            else hi = store - 1;
        }
    }

    private static double[] DescribeViaDouble<T>(Column<T> col) where T : struct, INumber<T>
    {
        int n = col.Length;
        int count = n - col.NullCount;
        if (count == 0) return [0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];

        var span = col.Buffer.Span;
        double[] doubles;
        if (col.NullCount == 0)
        {
            doubles = new double[n];
            for (int i = 0; i < n; i++) doubles[i] = double.CreateChecked(span[i]);
        }
        else
        {
            doubles = new double[count];
            int j = 0;
            for (int i = 0; i < n; i++)
                if (!col.IsNull(i)) doubles[j++] = double.CreateChecked(span[i]);
        }

        double sum = 0, mn = double.MaxValue, mx = double.MinValue;
        for (int i = 0; i < count; i++) { sum += doubles[i]; if (doubles[i] < mn) mn = doubles[i]; if (doubles[i] > mx) mx = doubles[i]; }
        double mean = sum / count;
        double sumSq = 0;
        for (int i = 0; i < count; i++) { double d = doubles[i] - mean; sumSq += d * d; }
        double std = count > 1 ? Math.Sqrt(sumSq / (count - 1)) : 0;

        int k25 = (int)(0.25 * (count - 1));
        int k50 = (int)(0.50 * (count - 1));
        int k75 = (int)(0.75 * (count - 1));
        QuickSelect(doubles, 0, count - 1, k25);
        double q25 = doubles[k25];
        QuickSelect(doubles, k25, count - 1, k50);
        double q50 = doubles[k50];
        QuickSelect(doubles, k50, count - 1, k75);
        double q75 = doubles[k75];

        return [count, mean, std, mn, q25, q50, q75, mx];
    }

    private static double Percentile(double[] sorted, double p)
    {
        int n = sorted.Length;
        double rank = p * (n - 1);
        int lower = (int)rank;
        int upper = lower + 1;
        double frac = rank - lower;
        if (upper >= n) return sorted[lower];
        return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
    }
}
