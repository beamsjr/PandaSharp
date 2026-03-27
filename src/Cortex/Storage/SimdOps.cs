using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace Cortex.Storage;

/// <summary>
/// SIMD-accelerated operations for numeric columns.
/// Falls back to scalar when hardware intrinsics unavailable.
/// </summary>
internal static class SimdOps
{
    public static double SumDouble(ReadOnlySpan<double> values)
    {
        int i = 0;

        if (AdvSimd.Arm64.IsSupported && values.Length >= 2)
        {
            var acc = Vector128<double>.Zero;
            for (; i <= values.Length - 2; i += 2)
            {
                var vec = Vector128.Create(values[i], values[i + 1]);
                acc = AdvSimd.Arm64.Add(acc, vec);
            }
            double sum = acc[0] + acc[1];
            for (; i < values.Length; i++) sum += values[i];
            return sum;
        }

        if (Avx.IsSupported && values.Length >= 4)
        {
            var acc = Vector256<double>.Zero;
            for (; i <= values.Length - 4; i += 4)
            {
                var vec = Vector256.Create(values[i], values[i + 1], values[i + 2], values[i + 3]);
                acc = Avx.Add(acc, vec);
            }
            double sum = acc[0] + acc[1] + acc[2] + acc[3];
            for (; i < values.Length; i++) sum += values[i];
            return sum;
        }

        // Scalar fallback
        {
            double sum = 0;
            for (i = 0; i < values.Length; i++) sum += values[i];
            return sum;
        }
    }

    public static long SumInt(ReadOnlySpan<int> values)
    {
        long sum = 0;
        for (int i = 0; i < values.Length; i++) sum += values[i];
        return sum;
    }

    public static double MinDouble(ReadOnlySpan<double> values)
    {
        if (values.Length == 0) return double.NaN;
        double min = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] < min) min = values[i];
        return min;
    }

    public static double MaxDouble(ReadOnlySpan<double> values)
    {
        if (values.Length == 0) return double.NaN;
        double max = values[0];
        for (int i = 1; i < values.Length; i++)
            if (values[i] > max) max = values[i];
        return max;
    }

    /// <summary>
    /// Count true values in a bool span using vectorized operations.
    /// </summary>
    public static int CountTrue(ReadOnlySpan<bool> mask)
    {
        int count = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) count++;
        return count;
    }
}
