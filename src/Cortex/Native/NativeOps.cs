using System.Numerics;
using System.Runtime.InteropServices;
using Cortex.Column;

namespace Cortex.Native;

/// <summary>
/// P/Invoke bindings to native C accelerators.
/// Falls back to managed implementations if the native library is unavailable.
/// </summary>
public static class NativeOps
{
    private const string LibName = "libpandasharp";
    private static readonly bool _available;

    static NativeOps()
    {
        try
        {
            _available = NativeLibrary.TryLoad(LibName, typeof(NativeOps).Assembly, null, out _);
            if (!_available)
            {
                // Also try Native/ subdirectory next to the assembly
                var asmDir = Path.GetDirectoryName(typeof(NativeOps).Assembly.Location);
                if (asmDir is not null)
                {
                    var nativeDir = Path.Combine(asmDir, "Native");
                    var candidates = new[] { "libpandasharp.dylib", "libpandasharp.so", "pandasharp.dll" };
                    foreach (var name in candidates)
                    {
                        var path = Path.Combine(nativeDir, name);
                        if (File.Exists(path) && NativeLibrary.TryLoad(path, out _))
                        {
                            _available = true;
                            break;
                        }
                    }
                }
            }
        }
        catch { _available = false; }
    }

    /// <summary>Whether the native library is loaded.</summary>
    public static bool IsAvailable => _available;

    // ── P/Invoke declarations ──

    [DllImport(LibName)] private static extern void gram_matrix_upper(IntPtr X, int n, int k, IntPtr C);
    [DllImport(LibName)] private static extern void rolling_mean(IntPtr input, IntPtr output, int length, int window);
    [DllImport(LibName)] private static extern void expanding_mean(IntPtr input, IntPtr output, int length);
    [DllImport(LibName)] private static extern double sum_double(IntPtr data, int length);
    [DllImport(LibName)] private static extern int filter_gt_double(IntPtr data, int length, double threshold, IntPtr mask);
    [DllImport(LibName)] private static extern int filter_abs_gt_double(IntPtr data, int length, double threshold, IntPtr mask);
    [DllImport(LibName)] private static extern void add_arrays(IntPtr a, IntPtr b, IntPtr output, int length);
    [DllImport(LibName)] private static extern void sub_arrays(IntPtr a, IntPtr b, IntPtr output, int length);
    [DllImport(LibName)] private static extern void mul_arrays(IntPtr a, IntPtr b, IntPtr output, int length);
    [DllImport(LibName)] private static extern void mul_scalar(IntPtr a, double scalar, IntPtr output, int length);
    [DllImport(LibName)] private static extern void eval_daily_return(IntPtr close, IntPtr open, IntPtr output, int length);
    [DllImport(LibName)] private static extern void eval_spread(IntPtr high, IntPtr low, IntPtr close, IntPtr output, int length);
    [DllImport(LibName)] private static extern int filter_complex(IntPtr ret, IntPtr vol, IntPtr close, int length, double minRet, double minVol, double minClose, IntPtr mask);
    [DllImport(LibName)] private static extern void str_upper(IntPtr input, IntPtr output, int totalBytes);
    [DllImport(LibName)] private static extern void str_lower(IntPtr input, IntPtr output, int totalBytes);
    [DllImport(LibName)] private static extern void agg_sum_by_group(IntPtr data, IntPtr groupIds, int nRows, IntPtr groupSums, int nGroups);
    [DllImport(LibName)] private static extern void agg_count_by_group(IntPtr groupIds, int nRows, IntPtr groupCounts, int nGroups);
    [DllImport(LibName)] private static extern void agg_min_by_group(IntPtr data, IntPtr groupIds, int nRows, IntPtr groupMins, int nGroups);
    [DllImport(LibName)] private static extern void agg_max_by_group(IntPtr data, IntPtr groupIds, int nRows, IntPtr groupMaxs, int nGroups);
    [DllImport(LibName)] private static extern void cast_int_to_double(IntPtr input, IntPtr output, int length);
    [DllImport(LibName)] private static extern void multi_agg_double(IntPtr data, IntPtr groupIds, int nRows, int nGroups, IntPtr sums, IntPtr counts, IntPtr mins, IntPtr maxs, IntPtr means, IntPtr stds);
    [DllImport(LibName)] private static extern void dedup_hash_2str(IntPtr hash1, IntPtr hash2, int nRows, IntPtr combined);
    [DllImport(LibName)] private static extern void map_column_double(IntPtr source, IntPtr rowMap, IntPtr output, int nRows);
    [DllImport(LibName)] private static extern void map_column_int(IntPtr source, IntPtr rowMap, IntPtr output, int nRows);

    // ── High-level typed wrappers ──

    /// <summary>Native O(n) rolling mean — ~3-5x faster than managed for large arrays.</summary>
    public static double[] RollingMean(ReadOnlySpan<double> input, int window)
    {
        var output = new double[input.Length];
        if (!_available) return ManagedRollingMean(input, output, window);

        unsafe
        {
            fixed (double* pIn = input, pOut = output)
            {
                rolling_mean((IntPtr)pIn, (IntPtr)pOut, input.Length, window);
            }
        }
        return output;
    }

    /// <summary>Native expanding mean.</summary>
    public static double[] ExpandingMean(ReadOnlySpan<double> input)
    {
        var output = new double[input.Length];
        if (!_available) return ManagedExpandingMean(input, output);

        unsafe
        {
            fixed (double* pIn = input, pOut = output)
            {
                expanding_mean((IntPtr)pIn, (IntPtr)pOut, input.Length);
            }
        }
        return output;
    }

    /// <summary>Native Kahan-compensated sum.</summary>
    public static double Sum(ReadOnlySpan<double> data)
    {
        if (!_available)
        {
            double s = 0;
            for (int i = 0; i < data.Length; i++) s += data[i];
            return s;
        }

        unsafe
        {
            fixed (double* p = data)
            {
                return sum_double((IntPtr)p, data.Length);
            }
        }
    }

    /// <summary>Native filter: data[i] > threshold.</summary>
    public static bool[] FilterGt(ReadOnlySpan<double> data, double threshold)
    {
        var mask = new bool[data.Length];
        if (!_available)
        {
            for (int i = 0; i < data.Length; i++) mask[i] = data[i] > threshold;
            return mask;
        }

        var byteMask = new byte[data.Length];
        unsafe
        {
            fixed (double* pData = data)
            fixed (byte* pMask = byteMask)
            {
                filter_gt_double((IntPtr)pData, data.Length, threshold, (IntPtr)pMask);
            }
        }
        for (int i = 0; i < data.Length; i++) mask[i] = byteMask[i] != 0;
        return mask;
    }

    /// <summary>Native filter: abs(data[i]) > threshold.</summary>
    public static bool[] FilterAbsGt(ReadOnlySpan<double> data, double threshold)
    {
        var mask = new bool[data.Length];
        if (!_available)
        {
            for (int i = 0; i < data.Length; i++) mask[i] = Math.Abs(data[i]) > threshold;
            return mask;
        }

        var byteMask = new byte[data.Length];
        unsafe
        {
            fixed (double* pData = data)
            fixed (byte* pMask = byteMask)
            {
                filter_abs_gt_double((IntPtr)pData, data.Length, threshold, (IntPtr)pMask);
            }
        }
        for (int i = 0; i < data.Length; i++) mask[i] = byteMask[i] != 0;
        return mask;
    }

    /// <summary>Fused (close - open) / open * 100 — native C when raw bytes available, managed otherwise.</summary>
    public static Column<double> EvalDailyReturn(Column<double> closeCol, Column<double> openCol, string name)
    {
        int n = closeCol.Length;
        var outBytes = new byte[n * sizeof(double)];

        // Try true zero-copy native: pin Arrow byte buffers directly
        var cRaw = closeCol.RawBytes;
        var oRaw = openCol.RawBytes;
        if (_available && cRaw is not null && oRaw is not null)
        {
            unsafe
            {
                fixed (byte* pC = cRaw, pO = oRaw, pOut = outBytes)
                    eval_daily_return((IntPtr)pC, (IntPtr)pO, (IntPtr)pOut, n);
            }
            return Column<double>.WrapResult(name, outBytes, n);
        }

        // Managed fallback
        var output = MemoryMarshal.Cast<byte, double>(outBytes.AsSpan());
        var cs = closeCol.Buffer.Span;
        var os = openCol.Buffer.Span;
        for (int i = 0; i < n; i++)
            output[i] = (cs[i] - os[i]) / os[i] * 100;
        return Column<double>.WrapResult(name, outBytes, n);
    }

    /// <summary>Fused (high - low) / close * 100 — native C when raw bytes available, managed otherwise.</summary>
    public static Column<double> EvalSpread(Column<double> highCol, Column<double> lowCol, Column<double> closeCol, string name)
    {
        int n = highCol.Length;
        var outBytes = new byte[n * sizeof(double)];

        var hRaw = highCol.RawBytes;
        var lRaw = lowCol.RawBytes;
        var cRaw = closeCol.RawBytes;
        if (_available && hRaw is not null && lRaw is not null && cRaw is not null)
        {
            unsafe
            {
                fixed (byte* pH = hRaw, pL = lRaw, pC = cRaw, pOut = outBytes)
                    eval_spread((IntPtr)pH, (IntPtr)pL, (IntPtr)pC, (IntPtr)pOut, n);
            }
            return Column<double>.WrapResult(name, outBytes, n);
        }

        var output = MemoryMarshal.Cast<byte, double>(outBytes.AsSpan());
        var hs = highCol.Buffer.Span; var ls = lowCol.Buffer.Span; var cs = closeCol.Buffer.Span;
        for (int i = 0; i < n; i++)
            output[i] = (hs[i] - ls[i]) / cs[i] * 100;
        return Column<double>.WrapResult(name, outBytes, n);
    }

    /// <summary>Fused 3-predicate filter — managed branchless (faster than native due to byte→bool conversion overhead).</summary>
    public static bool[] FilterComplex(Column<double> retCol, Column<double> volCol, Column<double> closeCol,
        double minRet, double minVol, double minClose)
    {
        int n = retCol.Length;
        var mask = new bool[n];
        var rs = retCol.Buffer.Span; var vs = volCol.Buffer.Span; var cs = closeCol.Buffer.Span;
        for (int i = 0; i < n; i++)
            mask[i] = rs[i] > minRet & vs[i] > minVol & cs[i] > minClose;
        return mask;
    }

    /// <summary>Native string upper — uses packed byte buffer, one C call for ALL strings.</summary>
    public static StringColumn StringUpper(StringColumn col)
    {
        if (!_available) return col.Str.Upper();
        return col.ApplyPackedTransform((pIn, pOut, len) => str_upper(pIn, pOut, len));
    }

    /// <summary>Native string lower — uses packed byte buffer, one C call for ALL strings.</summary>
    public static StringColumn StringLower(StringColumn col)
    {
        if (!_available) return col.Str.Lower();
        return col.ApplyPackedTransform((pIn, pOut, len) => str_lower(pIn, pOut, len));
    }

    /// <summary>Native int→double cast — single C loop, no per-element overhead.</summary>
    public static Column<double> CastIntToDouble(Column<int> col)
    {
        var output = new double[col.Length];
        if (!_available)
        {
            var span = col.Values;
            for (int i = 0; i < col.Length; i++) output[i] = span[i];
            return new Column<double>(col.Name, output);
        }
        unsafe
        {
            var input = col.Values.ToArray();
            fixed (int* pIn = input)
            fixed (double* pOut = output)
                cast_int_to_double((IntPtr)pIn, (IntPtr)pOut, col.Length);
        }
        return new Column<double>(col.Name, output);
    }

    /// <summary>
    /// Native multi-aggregation: compute sum+count+min+max+mean+std in ONE C pass.
    /// Eliminates 7 separate group-index iterations.
    /// </summary>
    /// <summary>
    /// Multi-aggregation: compute sum+count+min+max+mean+std.
    /// Uses native C when available (auto-vectorized), managed span fallback otherwise.
    /// </summary>
    public static (double[] Sums, int[] Counts, double[] Mins, double[] Maxs, double[] Means, double[] Stds)
        MultiAggDouble(ReadOnlySpan<double> data, int[] groupIds, int nRows, int nGroups)
        => MultiAggDouble(data, groupIds, nRows, nGroups, rawBytes: null);

    /// <summary>Overload accepting Column directly — uses RawBytes for zero-copy parallel.</summary>
    public static (double[] Sums, int[] Counts, double[] Mins, double[] Maxs, double[] Means, double[] Stds)
        MultiAggDouble(Column<double> col, int[] groupIds, int nGroups)
        => MultiAggDouble(col.Values, groupIds, col.Length, nGroups, col.RawBytes);

    private static (double[] Sums, int[] Counts, double[] Mins, double[] Maxs, double[] Means, double[] Stds)
        MultiAggDouble(ReadOnlySpan<double> data, int[] groupIds, int nRows, int nGroups, byte[]? rawBytes)
    {
        var sums = new double[nGroups];
        var counts = new int[nGroups];
        var mins = new double[nGroups];
        var maxs = new double[nGroups];
        var means = new double[nGroups];
        var stds = new double[nGroups];

        // For large datasets, use parallel chunk aggregation with per-thread accumulators
        int nThreads = Math.Min(Environment.ProcessorCount, nRows > 1_000_000 ? 4 : 1);
        if (nThreads > 1)
        {
            // Use RawBytes directly if available (zero-copy, thread-safe byte[] access)
            // Otherwise copy span to array for thread safety
            var dataArr = rawBytes is null ? data.ToArray() : null;
            var rawRef = rawBytes; // capture for lambda

            // Per-thread partial accumulators
            var tSums = new double[nThreads][];
            var tCounts = new int[nThreads][];
            var tMins = new double[nThreads][];
            var tMaxs = new double[nThreads][];
            var tSumSq = new double[nThreads][]; // for std

            Parallel.For(0, nThreads, t =>
            {
                int start = (int)((long)nRows * t / nThreads);
                int end = (int)((long)nRows * (t + 1) / nThreads);
                var ls = new double[nGroups]; var lc = new int[nGroups];
                var ln = new double[nGroups]; var lx = new double[nGroups];
                for (int g = 0; g < nGroups; g++) { ln[g] = double.MaxValue; lx[g] = double.MinValue; }

                // Read from rawRef (zero-copy) or dataArr (copied)
                if (rawRef is not null)
                {
                    // Access raw bytes as doubles via unsafe cast (thread-safe read)
                    unsafe
                    {
                        fixed (byte* pRaw = rawRef)
                        {
                            var dPtr = (double*)pRaw;
                            for (int i = start; i < end; i++)
                            {
                                int g = groupIds[i]; double v = dPtr[i];
                                ls[g] += v; lc[g]++;
                                if (v < ln[g]) ln[g] = v;
                                if (v > lx[g]) lx[g] = v;
                            }
                        }
                    }
                }
                else
                {
                    for (int i = start; i < end; i++)
                    {
                        int g = groupIds[i]; double v = dataArr![i];
                        ls[g] += v; lc[g]++;
                        if (v < ln[g]) ln[g] = v;
                        if (v > lx[g]) lx[g] = v;
                    }
                }
                tSums[t] = ls; tCounts[t] = lc; tMins[t] = ln; tMaxs[t] = lx;
            });

            // Merge partial results
            for (int g = 0; g < nGroups; g++) { mins[g] = double.MaxValue; maxs[g] = double.MinValue; }
            for (int t = 0; t < nThreads; t++)
            {
                for (int g = 0; g < nGroups; g++)
                {
                    sums[g] += tSums[t][g]; counts[g] += tCounts[t][g];
                    if (tMins[t][g] < mins[g]) mins[g] = tMins[t][g];
                    if (tMaxs[t][g] > maxs[g]) maxs[g] = tMaxs[t][g];
                }
            }
            for (int g = 0; g < nGroups; g++) means[g] = counts[g] > 0 ? sums[g] / counts[g] : 0;

            // Parallel pass 2: std (also zero-copy when rawRef available)
            var tStds = new double[nThreads][];
            Parallel.For(0, nThreads, t =>
            {
                int start = (int)((long)nRows * t / nThreads);
                int end = (int)((long)nRows * (t + 1) / nThreads);
                var ls = new double[nGroups];
                if (rawRef is not null)
                {
                    unsafe
                    {
                        fixed (byte* pRaw = rawRef)
                        {
                            var dPtr = (double*)pRaw;
                            for (int i = start; i < end; i++)
                            { int g = groupIds[i]; double d = dPtr[i] - means[g]; ls[g] += d * d; }
                        }
                    }
                }
                else
                {
                    for (int i = start; i < end; i++)
                    { int g = groupIds[i]; double d = dataArr![i] - means[g]; ls[g] += d * d; }
                }
                tStds[t] = ls;
            });
            for (int t = 0; t < nThreads; t++)
                for (int g = 0; g < nGroups; g++) stds[g] += tStds[t][g];
            for (int g = 0; g < nGroups; g++) stds[g] = counts[g] > 1 ? Math.Sqrt(stds[g] / (counts[g] - 1)) : 0;

            return (sums, counts, mins, maxs, means, stds);
        }

        // Sequential fallback for small data
        for (int g = 0; g < nGroups; g++) { mins[g] = double.MaxValue; maxs[g] = double.MinValue; }
        for (int i = 0; i < nRows; i++)
        {
            int g = groupIds[i]; double v = data[i];
            sums[g] += v; counts[g]++;
            if (v < mins[g]) mins[g] = v;
            if (v > maxs[g]) maxs[g] = v;
        }
        for (int g = 0; g < nGroups; g++) means[g] = counts[g] > 0 ? sums[g] / counts[g] : 0;
        for (int i = 0; i < nRows; i++)
        { int g = groupIds[i]; double d = data[i] - means[g]; stds[g] += d * d; }
        for (int g = 0; g < nGroups; g++) stds[g] = counts[g] > 1 ? Math.Sqrt(stds[g] / (counts[g] - 1)) : 0;

        return (sums, counts, mins, maxs, means, stds);
    }

    /// <summary>Array overload for backward compatibility.</summary>
    public static (double[] Sums, int[] Counts, double[] Mins, double[] Maxs, double[] Means, double[] Stds)
        MultiAggDouble(double[] data, int[] groupIds, int nRows, int nGroups)
        => MultiAggDouble(data.AsSpan(), groupIds, nRows, nGroups);

    // Apple Accelerate BLAS — dsyrk computes C = alpha * X^T * X + beta * C
    [DllImport("/System/Library/Frameworks/Accelerate.framework/Accelerate")]
    private static extern void cblas_dsyrk(int order, int uplo, int trans,
        int n, int k, double alpha, IntPtr A, int lda,
        double beta, IntPtr C, int ldc);

    private const int CblasRowMajor = 101;
    private const int CblasUpper = 121;
    private const int CblasTrans = 112;

    private static bool _accelerateChecked;
    private static bool _accelerateAvailable;

    /// <summary>Gram matrix C = X^T * X using Apple Accelerate BLAS when available.</summary>
    public static void GramMatrixUpper(double[] X, int n, int k, double[] C)
    {
        // Use cached Accelerate availability to avoid repeated try/catch
        if (!_accelerateChecked)
        {
            try
            {
                Array.Clear(C);
                unsafe
                {
                    fixed (double* pX = X, pC = C)
                    {
                        cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                            k, n, 1.0, (IntPtr)pX, k, 0.0, (IntPtr)pC, k);
                    }
                }
                _accelerateAvailable = true;
                _accelerateChecked = true;
                return;
            }
            catch (DllNotFoundException)
            {
                _accelerateAvailable = false;
                _accelerateChecked = true;
            }
        }
        else if (_accelerateAvailable)
        {
            Array.Clear(C);
            unsafe
            {
                fixed (double* pX = X, pC = C)
                {
                    cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans,
                        k, n, 1.0, (IntPtr)pX, k, 0.0, (IntPtr)pC, k);
                }
            }
            return;
        }

        if (_available)
        {
            unsafe
            {
                fixed (double* pX = X, pC = C)
                    gram_matrix_upper((IntPtr)pX, n, k, (IntPtr)pC);
            }
        }
        else
        {
            Array.Clear(C);
            for (int r = 0; r < n; r++)
            {
                int rowOff = r * k;
                for (int ci = 0; ci < k; ci++)
                {
                    double xi = X[rowOff + ci];
                    for (int cj = ci; cj < k; cj++)
                        C[ci * k + cj] += xi * X[rowOff + cj];
                }
            }
        }
    }

    /// <summary>
    /// Compute Q25, Q50, Q75 using native Floyd-Rivest selection (~2n comparisons).
    /// Data array is modified in-place (partially sorted). NaN must be pre-filtered.
    /// Returns (q25, q50, q75).
    /// </summary>
    /// <summary>Native dedup hash for 2 string columns.</summary>
    public static long[] DedupHash2Str(int[] hash1, int[] hash2, int nRows)
    {
        var combined = new long[nRows];
        if (!_available)
        {
            for (int i = 0; i < nRows; i++)
                combined[i] = ((long)hash1[i] << 32) | (uint)hash2[i];
            return combined;
        }
        unsafe
        {
            fixed (int* pH1 = hash1, pH2 = hash2)
            fixed (long* pOut = combined)
                dedup_hash_2str((IntPtr)pH1, (IntPtr)pH2, nRows, (IntPtr)pOut);
        }
        return combined;
    }

    /// <summary>Native column mapping for JoinMany: output[i] = source[rowMap[i]].</summary>
    public static void MapColumnDouble(double[] source, int[] rowMap, double[] output, int nRows)
    {
        if (!_available) { for (int i = 0; i < nRows; i++) output[i] = rowMap[i] >= 0 ? source[rowMap[i]] : 0; return; }
        unsafe { fixed (double* pS = source, pO = output) fixed (int* pM = rowMap) map_column_double((IntPtr)pS, (IntPtr)pM, (IntPtr)pO, nRows); }
    }

    public static void MapColumnInt(int[] source, int[] rowMap, int[] output, int nRows)
    {
        if (!_available) { for (int i = 0; i < nRows; i++) output[i] = rowMap[i] >= 0 ? source[rowMap[i]] : 0; return; }
        unsafe { fixed (int* pS = source, pO = output, pM = rowMap) map_column_int((IntPtr)pS, (IntPtr)pM, (IntPtr)pO, nRows); }
    }

    // ── Managed fallbacks ──

    private static double[] ManagedRollingMean(ReadOnlySpan<double> input, double[] output, int window)
    {
        double sum = 0; int count = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (!double.IsNaN(input[i])) { sum += input[i]; count++; }
            if (i >= window && !double.IsNaN(input[i - window])) { sum -= input[i - window]; count--; }
            output[i] = (i < window - 1 || count == 0) ? double.NaN : sum / count;
        }
        return output;
    }

    private static double[] ManagedExpandingMean(ReadOnlySpan<double> input, double[] output)
    {
        double sum = 0; int count = 0;
        for (int i = 0; i < input.Length; i++)
        {
            if (!double.IsNaN(input[i])) { sum += input[i]; count++; }
            output[i] = count > 0 ? sum / count : double.NaN;
        }
        return output;
    }
}
