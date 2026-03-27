using ILGPU;
using ILGPU.Runtime;

namespace Cortex.GPU.Kernels;

/// <summary>
/// GPU kernels for matrix operations: multiply, transpose, gram matrix.
/// </summary>
internal static class MatrixKernels
{
    /// <summary>
    /// Matrix multiply: C[i,j] = sum_k A[i,k] * B[k,j]
    /// A is m×k, B is k×n, C is m×n.
    /// Uses tiled approach for better memory coalescing.
    /// </summary>
    internal static void MatMulKernel(
        Index2D index,
        ArrayView<double> A,
        ArrayView<double> B,
        ArrayView<double> C,
        int m, int k, int n)
    {
        int row = index.X;
        int col = index.Y;

        if (row >= m || col >= n) return;

        double sum = 0;
        int aBase = row * k;
        for (int i = 0; i < k; i++)
            sum += A[aBase + i] * B[i * n + col];

        C[row * n + col] = sum;
    }

    /// <summary>
    /// Gram matrix (symmetric): C = A^T * A
    /// A is n×d, C is d×d. Only computes upper triangle + diagonal.
    /// </summary>
    internal static void GramMatrixKernel(
        Index2D index,
        ArrayView<double> A,
        ArrayView<double> C,
        int n, int d)
    {
        int i = index.X;
        int j = index.Y;

        // Only compute upper triangle
        if (i > j || i >= d || j >= d) return;

        double sum = 0;
        for (int r = 0; r < n; r++)
            sum += A[r * d + i] * A[r * d + j];

        C[i * d + j] = sum;
        if (i != j) C[j * d + i] = sum; // mirror
    }

    /// <summary>
    /// Matrix transpose: B[j,i] = A[i,j]
    /// A is m×n, B is n×m.
    /// </summary>
    internal static void TransposeKernel(
        Index2D index,
        ArrayView<double> A,
        ArrayView<double> B,
        int m, int n)
    {
        int i = index.X;
        int j = index.Y;
        if (i >= m || j >= n) return;
        B[j * m + i] = A[i * n + j];
    }

    /// <summary>
    /// Normalize rows: each row[i] /= norm[i]. A is n×d.
    /// </summary>
    internal static void NormalizeRowsKernel(
        Index2D index,
        ArrayView<double> A,
        ArrayView<double> norms,
        int d)
    {
        int row = index.X;
        int col = index.Y;
        if (col >= d) return;

        double norm = norms[row];
        if (norm > 0)
            A[row * d + col] /= norm;
    }
}
