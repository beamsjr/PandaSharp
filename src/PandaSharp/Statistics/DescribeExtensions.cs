using System.Numerics;
using System.Runtime.InteropServices;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class DescribeExtensions
{
    // Native C++ nth_element-based quantile computation (~2n comparisons)
    private const string QuantileLib = "libpandasharp_quantile";
    private static bool _quantileChecked;
    private static bool _quantileAvailable;

    [DllImport(QuantileLib)]
    private static extern void describe_quantiles(IntPtr data, int n, IntPtr output);

    private static bool NativeQuantileAvailable
    {
        get
        {
            if (!_quantileChecked)
            {
                _quantileChecked = true;
                try
                {
                    _quantileAvailable = NativeLibrary.TryLoad(QuantileLib, typeof(DescribeExtensions).Assembly, null, out _);
                    if (!_quantileAvailable)
                    {
                        var asmDir = Path.GetDirectoryName(typeof(DescribeExtensions).Assembly.Location);
                        if (asmDir is not null)
                        {
                            foreach (var name in new[] { "libpandasharp_quantile.dylib", "libpandasharp_quantile.so", "pandasharp_quantile.dll" })
                            {
                                var path = Path.Combine(asmDir, "Native", name);
                                if (File.Exists(path) && NativeLibrary.TryLoad(path, out _))
                                { _quantileAvailable = true; break; }
                                path = Path.Combine(asmDir, name);
                                if (File.Exists(path) && NativeLibrary.TryLoad(path, out _))
                                { _quantileAvailable = true; break; }
                            }
                        }
                    }
                }
                catch { _quantileAvailable = false; }
            }
            return _quantileAvailable;
        }
    }

    private static unsafe (double Q25, double Q50, double Q75) NativeDescribeQuantiles(double[] data, int count)
    {
        var output = new double[3];
        fixed (double* pData = data, pOut = output)
            describe_quantiles((IntPtr)pData, count, (IntPtr)pOut);
        return (output[0], output[1], output[2]);
    }

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
        int validCount;
        if (col.NullCount == 0)
        {
            // Quick NaN scan: only do NaN compaction if NaN actually exists
            bool hasNaN = false;
            for (int i = 0; i < n; i++)
                if (double.IsNaN(span[i])) { hasNaN = true; break; }

            if (!hasNaN)
            {
                data = span.ToArray();
                validCount = n;
            }
            else
            {
                data = new double[n];
                validCount = 0;
                for (int i = 0; i < n; i++)
                    if (!double.IsNaN(span[i]))
                        data[validCount++] = span[i];
            }
        }
        else
        {
            data = new double[count];
            int j = 0;
            for (int i = 0; i < n; i++)
                if (!col.IsNull(i)) data[j++] = span[i];

            // Further compact: remove NaN values from the data array
            validCount = 0;
            for (int i = 0; i < count; i++)
                if (!double.IsNaN(data[i]))
                    data[validCount++] = data[i];
        }

        if (validCount == 0) return [0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];

        // Pass 1: mean + std + min + max from NaN-free compacted data
        double sum = 0, mn = double.MaxValue, mx = double.MinValue;
        for (int i = 0; i < validCount; i++)
        {
            double v = data[i];
            sum += v;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        double mean = sum / validCount;
        double sumSq = 0;
        for (int i = 0; i < validCount; i++) { double d = data[i] - mean; sumSq += d * d; }
        double std = validCount > 1 ? Math.Sqrt(sumSq / (validCount - 1)) : 0;

        // Quantiles: try native O(n) cascaded nth_element, fallback to O(n log n) sort
        double q25, q50, q75;
        if (NativeQuantileAvailable && validCount > 100)
        {
            var (nq25, nq50, nq75) = NativeDescribeQuantiles(data, validCount);
            q25 = nq25; q50 = nq50; q75 = nq75;
        }
        else
        {
            Array.Sort(data, 0, validCount);
            q25 = SortedQuantile(data, validCount, 0.25);
            q50 = SortedQuantile(data, validCount, 0.50);
            q75 = SortedQuantile(data, validCount, 0.75);
        }

        return [validCount, mean, std, mn, q25, q50, q75, mx];
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

        bool isFloating = typeof(T) == typeof(float) || typeof(T) == typeof(double);
        var span = col.Buffer.Span;
        double[] doubles;
        int validCount;
        if (col.NullCount == 0)
        {
            doubles = new double[n];
            for (int i = 0; i < n; i++) doubles[i] = double.CreateChecked(span[i]);

            // Quick NaN scan for floating-point types
            if (isFloating)
            {
                bool hasNaN = false;
                for (int i = 0; i < n; i++)
                    if (double.IsNaN(doubles[i])) { hasNaN = true; break; }

                if (hasNaN)
                {
                    validCount = 0;
                    for (int i = 0; i < n; i++)
                        if (!double.IsNaN(doubles[i]))
                            doubles[validCount++] = doubles[i];
                }
                else
                {
                    validCount = n;
                }
            }
            else
            {
                validCount = n;
            }
        }
        else
        {
            doubles = new double[count];
            int j = 0;
            for (int i = 0; i < n; i++)
                if (!col.IsNull(i)) doubles[j++] = double.CreateChecked(span[i]);

            // Filter out NaN values for floating-point types
            validCount = count;
            if (isFloating)
            {
                validCount = 0;
                for (int i = 0; i < count; i++)
                    if (!double.IsNaN(doubles[i]))
                        doubles[validCount++] = doubles[i];
            }
        }

        if (validCount == 0) return [0, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN, double.NaN];

        double sum = 0, mn = double.MaxValue, mx = double.MinValue;
        for (int i = 0; i < validCount; i++) { sum += doubles[i]; if (doubles[i] < mn) mn = doubles[i]; if (doubles[i] > mx) mx = doubles[i]; }
        double mean = sum / validCount;
        double sumSq = 0;
        for (int i = 0; i < validCount; i++) { double d = doubles[i] - mean; sumSq += d * d; }
        double std = validCount > 1 ? Math.Sqrt(sumSq / (validCount - 1)) : 0;

        double q25, q50, q75;
        if (NativeQuantileAvailable && validCount > 100)
        {
            var (nq25, nq50, nq75) = NativeDescribeQuantiles(doubles, validCount);
            q25 = nq25; q50 = nq50; q75 = nq75;
        }
        else
        {
            Array.Sort(doubles, 0, validCount);
            q25 = SortedQuantile(doubles, validCount, 0.25);
            q50 = SortedQuantile(doubles, validCount, 0.50);
            q75 = SortedQuantile(doubles, validCount, 0.75);
        }

        return [validCount, mean, std, mn, q25, q50, q75, mx];
    }

    /// <summary>
    /// Compute an interpolated quantile from pre-sorted data.
    /// O(1) — just index lookup with linear interpolation between adjacent elements.
    /// Matches pandas default "linear" interpolation method.
    /// </summary>
    private static double SortedQuantile(double[] sortedData, int count, double p)
    {
        if (count == 1) return sortedData[0];
        double idx = p * (count - 1);
        int lower = (int)idx;
        double frac = idx - lower;
        if (lower + 1 >= count || frac == 0.0)
            return sortedData[Math.Min(lower, count - 1)];
        return sortedData[lower] + frac * (sortedData[lower + 1] - sortedData[lower]);
    }

    /// <summary>
    /// Compute an interpolated quantile using QuickSelect (for use outside Describe).
    /// </summary>
    internal static double InterpolatedQuantile(double[] data, int count, double p, int lo)
    {
        if (count == 1) return data[lo];
        double idx = p * (count - 1);
        int lower = (int)idx;
        int upper = lower + 1;
        double frac = idx - lower;

        int loIdx = lo + lower;
        QuickSelect(data, lo, lo + count - 1, loIdx);
        double loVal = data[loIdx];

        if (upper >= count || frac == 0.0)
            return loVal;

        int hiIdx = loIdx + 1;
        double hiVal = data[hiIdx];
        for (int i = hiIdx + 1; i < lo + count; i++)
            if (data[i] < hiVal) hiVal = data[i];

        return loVal + frac * (hiVal - loVal);
    }
}
