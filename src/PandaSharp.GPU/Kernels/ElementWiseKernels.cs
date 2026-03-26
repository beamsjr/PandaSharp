using ILGPU;
using ILGPU.Algorithms;
using ILGPU.Runtime;

namespace PandaSharp.GPU.Kernels;

/// <summary>
/// GPU kernels for element-wise column operations: add, subtract, multiply, divide, scalar ops.
/// </summary>
internal static class ElementWiseKernels
{
    // ═══ Column-Column operations ═══

    internal static void AddKernel(Index1D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> result)
    {
        result[index] = a[index] + b[index];
    }

    internal static void SubtractKernel(Index1D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> result)
    {
        result[index] = a[index] - b[index];
    }

    internal static void MultiplyKernel(Index1D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> result)
    {
        result[index] = a[index] * b[index];
    }

    internal static void DivideKernel(Index1D index, ArrayView<double> a, ArrayView<double> b, ArrayView<double> result)
    {
        result[index] = a[index] / b[index];
    }

    // ═══ Column-Scalar operations ═══

    internal static void AddScalarKernel(Index1D index, ArrayView<double> a, double scalar, ArrayView<double> result)
    {
        result[index] = a[index] + scalar;
    }

    internal static void MultiplyScalarKernel(Index1D index, ArrayView<double> a, double scalar, ArrayView<double> result)
    {
        result[index] = a[index] * scalar;
    }

    internal static void SubtractScalarKernel(Index1D index, ArrayView<double> a, double scalar, ArrayView<double> result)
    {
        result[index] = a[index] - scalar;
    }

    internal static void DivideScalarKernel(Index1D index, ArrayView<double> a, double scalar, ArrayView<double> result)
    {
        result[index] = a[index] / scalar;
    }

    // ═══ Unary operations ═══

    internal static void NegateKernel(Index1D index, ArrayView<double> a, ArrayView<double> result)
    {
        result[index] = -a[index];
    }

    internal static void AbsKernel(Index1D index, ArrayView<double> a, ArrayView<double> result)
    {
        result[index] = a[index] >= 0 ? a[index] : -a[index];
    }

    internal static void SqrtKernel(Index1D index, ArrayView<double> a, ArrayView<double> result)
    {
        result[index] = XMath.Sqrt(a[index]);
    }

    // ═══ Float versions for Vision/ML ═══

    internal static void AddKernelF(Index1D index, ArrayView<float> a, ArrayView<float> b, ArrayView<float> result)
    {
        result[index] = a[index] + b[index];
    }

    internal static void MultiplyScalarKernelF(Index1D index, ArrayView<float> a, float scalar, ArrayView<float> result)
    {
        result[index] = a[index] * scalar;
    }

    internal static void FmaKernelF(Index1D index, ArrayView<float> a, float scale, float offset, ArrayView<float> result)
    {
        result[index] = a[index] * scale + offset;
    }
}
