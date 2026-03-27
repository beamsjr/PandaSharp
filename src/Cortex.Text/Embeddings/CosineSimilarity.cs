using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.Text.Embeddings;

/// <summary>
/// Cosine similarity computations for embedding vectors.
/// Supports single-pair, pairwise matrix, and labeled DataFrame output.
/// </summary>
public static class CosineSimilarity
{
    // Apple Accelerate BLAS for fast matrix multiply
    [DllImport("/System/Library/Frameworks/Accelerate.framework/Accelerate")]
    private static extern void cblas_dsyrk(int order, int uplo, int trans,
        int n, int k, double alpha, IntPtr A, int lda,
        double beta, IntPtr C, int ldc);

    private static volatile bool _blasChecked;
    private static volatile bool _blasAvailable;

    /// <summary>
    /// Compute cosine similarity between two vectors.
    /// </summary>
    public static double Compute(double[] a, double[] b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector length mismatch: {a.Length} vs {b.Length}", nameof(b));

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom == 0 ? 0.0 : dot / denom;
    }

    /// <summary>
    /// Compute cosine similarity between two spans for performance-sensitive code.
    /// </summary>
    public static double Compute(ReadOnlySpan<double> a, ReadOnlySpan<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Vector length mismatch: {a.Length} vs {b.Length}");

        double dot = 0, normA = 0, normB = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += a[i] * b[i];
            normA += a[i] * a[i];
            normB += b[i] * b[i];
        }

        double denom = Math.Sqrt(normA) * Math.Sqrt(normB);
        return denom == 0 ? 0.0 : dot / denom;
    }

    /// <summary>
    /// Compute pairwise cosine similarity matrix from a 2D embedding tensor.
    /// </summary>
    /// <param name="embeddings">2D tensor of shape [N, D] where N is number of vectors.</param>
    /// <returns>N x N similarity matrix.</returns>
    public static double[,] PairwiseMatrix(Tensor<double> embeddings)
    {
        if (embeddings.Rank != 2)
            throw new ArgumentException("Embeddings must be a 2D tensor [N, D].");

        int n = embeddings.Shape[0];
        int d = embeddings.Shape[1];
        var data = embeddings.Span.ToArray();

        // Normalize rows in-place: each row /= its L2 norm
        var norms = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            int offset = i * d;
            for (int k = 0; k < d; k++) sum += data[offset + k] * data[offset + k];
            norms[i] = Math.Sqrt(sum);
            if (norms[i] > 0)
            {
                double inv = 1.0 / norms[i];
                for (int k = 0; k < d; k++) data[offset + k] *= inv;
            }
        }

        // Compute gram matrix: S = normalized @ normalized.T
        var matrix = new double[n, n];

        if (TryBlasSyrk(data, matrix, n, d, norms))
            return matrix;

        // Fallback: parallel dot products (only upper triangle + copy)
        Parallel.For(0, n, i =>
        {
            int offI = i * d;
            matrix[i, i] = norms[i] > 0 ? 1.0 : 0.0;
            for (int j = i + 1; j < n; j++)
            {
                double dot = 0;
                int offJ = j * d;
                for (int k = 0; k < d; k++) dot += data[offI + k] * data[offJ + k];
                matrix[i, j] = dot;
                matrix[j, i] = dot;
            }
        });

        return matrix;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe bool TryBlasSyrk(double[] data, double[,] matrix, int n, int d, double[] norms)
    {
        if (_blasChecked && !_blasAvailable)
            return false;

        try
        {
            // Use flat array for BLAS, then copy into double[,] via pinning
            var flat = new double[n * n];
            fixed (double* pData = data, pFlat = flat)
            {
                cblas_dsyrk(101 /*RowMajor*/, 121 /*Upper*/, 111 /*NoTrans*/,
                    n, d, 1.0, (IntPtr)pData, d, 0.0, (IntPtr)pFlat, n);
            }

            // Fill lower triangle and copy to matrix simultaneously
            fixed (double* pFlat = flat, pMatrix = matrix)
            {
                for (int i = 0; i < n; i++)
                {
                    int rowOff = i * n;
                    // Copy upper triangle (already computed by BLAS)
                    for (int j = i; j < n; j++)
                    {
                        double v = pFlat[rowOff + j];
                        pMatrix[rowOff + j] = v;
                        pMatrix[j * n + i] = v; // symmetric
                    }
                }
            }

            _blasChecked = true;
            _blasAvailable = true;
            return true;
        }
        catch (Exception)
        {
            _blasChecked = true;
            _blasAvailable = false;
            return false;
        }
    }

    /// <summary>
    /// Compute pairwise cosine similarity and return as a labeled DataFrame.
    /// </summary>
    public static DataFrame PairwiseDataFrame(Tensor<double> embeddings, string[] labels)
    {
        if (embeddings.Rank != 2)
            throw new ArgumentException("Embeddings must be a 2D tensor [N, D].");
        if (labels.Length != embeddings.Shape[0])
            throw new ArgumentException("Labels length must match number of embedding rows.");

        var matrix = PairwiseMatrix(embeddings);
        int n = labels.Length;

        var columns = new List<IColumn>();
        columns.Add(new StringColumn("label", labels));

        for (int j = 0; j < n; j++)
        {
            var colValues = new double[n];
            for (int i = 0; i < n; i++)
                colValues[i] = matrix[i, j];
            columns.Add(new Column<double>(labels[j], colValues));
        }

        return new DataFrame(columns);
    }
}
