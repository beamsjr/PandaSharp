using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Algorithms.ScanReduceOperations;
using ILGPU.Runtime;

namespace Cortex.GPU.Kernels;

/// <summary>
/// GPU kernels for reduction operations: sum, min, max, mean, variance.
/// Uses ILGPU.Algorithms built-in reduction for optimal performance.
/// </summary>
internal static class ReductionKernels
{
    /// <summary>
    /// Compute sum, min, max in a single kernel launch using atomics.
    /// </summary>
    internal static void SumMinMaxKernel(
        Index1D index,
        ArrayView<double> data,
        ArrayView<double> partialSums,
        ArrayView<double> partialMins,
        ArrayView<double> partialMaxs)
    {
        var val = data[index];
        Atomic.Add(ref partialSums[0], val);
        Atomic.Min(ref partialMins[0], val);
        Atomic.Max(ref partialMaxs[0], val);
    }

    /// <summary>
    /// Compute sum of squares (for variance/std computation).
    /// meanVal is subtracted first: result += (x - mean)^2
    /// </summary>
    internal static void SumSquaredDiffKernel(
        Index1D index,
        ArrayView<double> data,
        double mean,
        ArrayView<double> result)
    {
        double diff = data[index] - mean;
        Atomic.Add(ref result[0], diff * diff);
    }

    /// <summary>
    /// Squared Euclidean distance between two rows.
    /// dist = sum((a[i] - b[i])^2) for i in 0..d
    /// </summary>
    internal static void PairwiseDistanceKernel(
        Index2D index,
        ArrayView<double> X,     // nX × d (flattened)
        ArrayView<double> Y,     // nY × d (flattened)
        ArrayView<double> dist,  // nX × nY (flattened)
        int d,
        int nY)
    {
        int i = index.X; // query index
        int j = index.Y; // train index
        double sum = 0;
        int offI = i * d;
        int offJ = j * d;
        for (int k = 0; k < d; k++)
        {
            double diff = X[offI + k] - Y[offJ + k];
            sum += diff * diff;
        }
        dist[i * nY + j] = sum;
    }
}
