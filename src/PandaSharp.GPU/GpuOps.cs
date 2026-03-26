using ILGPU;
using ILGPU.Runtime;
using PandaSharp.GPU.Kernels;

namespace PandaSharp.GPU;

/// <summary>
/// High-level GPU operations for PandaSharp. Manages kernel compilation, memory transfers,
/// and provides a simple API for DataFrame/Column operations.
/// Automatically falls back to CPU when data is too small to benefit from GPU transfer overhead.
/// </summary>
public sealed class GpuOps
{
    /// <summary>
    /// Minimum array size to justify GPU transfer overhead.
    /// Below this, CPU is faster due to PCIe/memory copy latency.
    /// </summary>
    public const int MinGpuSize = 50_000;

    private readonly Accelerator _acc;

    // Cached compiled kernels
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _addKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _subKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _mulKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _divKernel;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _addScalarKernel;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _mulScalarKernel;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _subScalarKernel;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _divScalarKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>>? _negateKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>>? _absKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>>? _sqrtKernel;
    private readonly Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>? _sumMinMaxKernel;
    private readonly Action<Index1D, ArrayView<double>, double, ArrayView<double>>? _sumSqDiffKernel;
    private readonly Action<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int>? _matMulKernel;
    private readonly Action<Index2D, ArrayView<double>, ArrayView<double>, int, int>? _gramKernel;
    private readonly Action<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int>? _distKernel;

    /// <summary>Whether this instance is backed by a real GPU.</summary>
    public bool HasGpu => _acc.AcceleratorType != AcceleratorType.CPU;

    /// <summary>Device description.</summary>
    public string DeviceInfo => $"{_acc.AcceleratorType}: {_acc.Name}";

    public GpuOps() : this(GpuAccelerator.Instance.Device) { }

    public GpuOps(Accelerator accelerator)
    {
        _acc = accelerator;

        // Compile all kernels upfront
        _addKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.AddKernel);
        _subKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.SubtractKernel);
        _mulKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.MultiplyKernel);
        _divKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.DivideKernel);
        _addScalarKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, double, ArrayView<double>>(ElementWiseKernels.AddScalarKernel);
        _mulScalarKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, double, ArrayView<double>>(ElementWiseKernels.MultiplyScalarKernel);
        _subScalarKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, double, ArrayView<double>>(ElementWiseKernels.SubtractScalarKernel);
        _divScalarKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, double, ArrayView<double>>(ElementWiseKernels.DivideScalarKernel);
        _negateKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.NegateKernel);
        _absKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.AbsKernel);
        _sqrtKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>>(ElementWiseKernels.SqrtKernel);
        _sumMinMaxKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>, ArrayView<double>>(ReductionKernels.SumMinMaxKernel);
        _sumSqDiffKernel = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<double>, double, ArrayView<double>>(ReductionKernels.SumSquaredDiffKernel);
        _matMulKernel = _acc.LoadAutoGroupedStreamKernel<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int, int>(MatrixKernels.MatMulKernel);
        _gramKernel = _acc.LoadAutoGroupedStreamKernel<Index2D, ArrayView<double>, ArrayView<double>, int, int>(MatrixKernels.GramMatrixKernel);
        _distKernel = _acc.LoadAutoGroupedStreamKernel<Index2D, ArrayView<double>, ArrayView<double>, ArrayView<double>, int, int>(ReductionKernels.PairwiseDistanceKernel);
    }

    // ═══════════════════════════════════════════════════════
    // Element-wise operations
    // ═══════════════════════════════════════════════════════

    /// <summary>Element-wise add: result = a + b</summary>
    public double[] Add(double[] a, double[] b)
    {
        ValidateBinaryArgs(a, b);
        if (a.Length == 0) return [];
        return BinaryOp(a, b, _addKernel!);
    }

    /// <summary>Element-wise subtract: result = a - b</summary>
    public double[] Subtract(double[] a, double[] b)
    {
        ValidateBinaryArgs(a, b);
        if (a.Length == 0) return [];
        return BinaryOp(a, b, _subKernel!);
    }

    /// <summary>Element-wise multiply: result = a * b</summary>
    public double[] Multiply(double[] a, double[] b)
    {
        ValidateBinaryArgs(a, b);
        if (a.Length == 0) return [];
        return BinaryOp(a, b, _mulKernel!);
    }

    /// <summary>Element-wise divide: result = a / b</summary>
    public double[] Divide(double[] a, double[] b)
    {
        ValidateBinaryArgs(a, b);
        if (a.Length == 0) return [];
        return BinaryOp(a, b, _divKernel!);
    }

    /// <summary>Scalar add: result = a + scalar</summary>
    public double[] AddScalar(double[] a, double scalar)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return ScalarOp(a, scalar, _addScalarKernel!);
    }

    /// <summary>Scalar multiply: result = a * scalar</summary>
    public double[] MultiplyScalar(double[] a, double scalar)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return ScalarOp(a, scalar, _mulScalarKernel!);
    }

    /// <summary>Scalar subtract: result = a - scalar</summary>
    public double[] SubtractScalar(double[] a, double scalar)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return ScalarOp(a, scalar, _subScalarKernel!);
    }

    /// <summary>Scalar divide: result = a / scalar</summary>
    public double[] DivideScalar(double[] a, double scalar)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return ScalarOp(a, scalar, _divScalarKernel!);
    }

    /// <summary>Negate: result = -a</summary>
    public double[] Negate(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return UnaryOp(a, _negateKernel!);
    }

    /// <summary>Absolute value: result = |a|</summary>
    public double[] Abs(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return UnaryOp(a, _absKernel!);
    }

    /// <summary>Square root: result = sqrt(a)</summary>
    public double[] Sqrt(double[] a)
    {
        ArgumentNullException.ThrowIfNull(a);
        if (a.Length == 0) return [];
        return UnaryOp(a, _sqrtKernel!);
    }

    private static void ValidateBinaryArgs(double[] a, double[] b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length != b.Length)
            throw new ArgumentException("Arrays must have equal length.");
    }

    // ═══════════════════════════════════════════════════════
    // Reductions
    // ═══════════════════════════════════════════════════════

    /// <summary>Compute sum, min, max in one GPU launch.</summary>
    public (double Sum, double Min, double Max) SumMinMax(double[] data)
    {
        int n = data.Length;
        using var dData = _acc.Allocate1D(data);
        using var dSum = _acc.Allocate1D<double>(1);
        using var dMin = _acc.Allocate1D<double>(1);
        using var dMax = _acc.Allocate1D<double>(1);

        // Initialize
        dSum.MemSetToZero();
        dMin.CopyFromCPU(new[] { double.MaxValue });
        dMax.CopyFromCPU(new[] { double.MinValue });

        _sumMinMaxKernel!(n, dData.View, dSum.View, dMin.View, dMax.View);
        _acc.Synchronize();

        var sum = new double[1];
        var min = new double[1];
        var max = new double[1];
        dSum.CopyToCPU(sum);
        dMin.CopyToCPU(min);
        dMax.CopyToCPU(max);

        return (sum[0], min[0], max[0]);
    }

    /// <summary>Compute variance: mean of squared differences from mean.</summary>
    public double Variance(double[] data, double mean)
    {
        int n = data.Length;
        using var dData = _acc.Allocate1D(data);
        using var dResult = _acc.Allocate1D<double>(1);
        dResult.MemSetToZero();

        _sumSqDiffKernel!(n, dData.View, mean, dResult.View);
        _acc.Synchronize();

        var result = new double[1];
        dResult.CopyToCPU(result);
        return result[0] / n;
    }

    // ═══════════════════════════════════════════════════════
    // Matrix operations
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// Matrix multiply: C = A × B.
    /// A is m×k, B is k×n, C is m×n. All row-major.
    /// </summary>
    public double[] MatMul(double[] A, double[] B, int m, int k, int n)
    {
        var C = new double[m * n];
        using var dA = _acc.Allocate1D(A);
        using var dB = _acc.Allocate1D(B);
        using var dC = _acc.Allocate1D<double>(m * n);
        dC.MemSetToZero();

        _matMulKernel!((m, n), dA.View, dB.View, dC.View, m, k, n);
        _acc.Synchronize();

        dC.CopyToCPU(C);
        return C;
    }

    /// <summary>
    /// Gram matrix: C = A^T × A. A is n×d, C is d×d (symmetric).
    /// </summary>
    public double[] GramMatrix(double[] A, int n, int d)
    {
        var C = new double[d * d];
        using var dA = _acc.Allocate1D(A);
        using var dC = _acc.Allocate1D<double>(d * d);
        dC.MemSetToZero();

        _gramKernel!((d, d), dA.View, dC.View, n, d);
        _acc.Synchronize();

        dC.CopyToCPU(C);
        return C;
    }

    /// <summary>
    /// Pairwise squared Euclidean distances.
    /// X is nX×d, Y is nY×d. Result is nX×nY.
    /// </summary>
    public double[] PairwiseDistances(double[] X, double[] Y, int nX, int nY, int d)
    {
        var dist = new double[nX * nY];
        using var dX = _acc.Allocate1D(X);
        using var dY = _acc.Allocate1D(Y);
        using var dDist = _acc.Allocate1D<double>(nX * nY);
        dDist.MemSetToZero();

        _distKernel!((nX, nY), dX.View, dY.View, dDist.View, d, nY);
        _acc.Synchronize();

        dDist.CopyToCPU(dist);
        return dist;
    }

    // ═══════════════════════════════════════════════════════
    // Private helpers
    // ═══════════════════════════════════════════════════════

    private double[] BinaryOp(double[] a, double[] b,
        Action<Index1D, ArrayView<double>, ArrayView<double>, ArrayView<double>> kernel)
    {
        int n = a.Length;
        var result = new double[n];
        using var dA = _acc.Allocate1D(a);
        using var dB = _acc.Allocate1D(b);
        using var dR = _acc.Allocate1D<double>(n);

        kernel(n, dA.View, dB.View, dR.View);
        _acc.Synchronize();

        dR.CopyToCPU(result);
        return result;
    }

    private double[] ScalarOp(double[] a, double scalar,
        Action<Index1D, ArrayView<double>, double, ArrayView<double>> kernel)
    {
        int n = a.Length;
        var result = new double[n];
        using var dA = _acc.Allocate1D(a);
        using var dR = _acc.Allocate1D<double>(n);

        kernel(n, dA.View, scalar, dR.View);
        _acc.Synchronize();

        dR.CopyToCPU(result);
        return result;
    }

    private double[] UnaryOp(double[] a,
        Action<Index1D, ArrayView<double>, ArrayView<double>> kernel)
    {
        int n = a.Length;
        var result = new double[n];
        using var dA = _acc.Allocate1D(a);
        using var dR = _acc.Allocate1D<double>(n);

        kernel(n, dA.View, dR.View);
        _acc.Synchronize();

        dR.CopyToCPU(result);
        return result;
    }
}
