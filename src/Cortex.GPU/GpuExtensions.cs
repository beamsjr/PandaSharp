using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.GPU;

/// <summary>
/// Extension methods that add GPU-accelerated operations to DataFrame and Column types.
/// Operations automatically fall back to CPU for small data or when no GPU is available.
/// </summary>
public static class GpuExtensions
{
    private static readonly Lazy<GpuOps> _ops = new(() => new GpuOps());
    private static GpuOps Ops => _ops.Value;

    // ═══════════════════════════════════════════════════════
    // Column GPU operations
    // ═══════════════════════════════════════════════════════

    /// <summary>GPU-accelerated element-wise add.</summary>
    public static Column<double> GpuAdd(this Column<double> a, Column<double> b)
    {
        var result = Ops.Add(a.Values.ToArray(), b.Values.ToArray());
        return new Column<double>(a.Name, result);
    }

    /// <summary>GPU-accelerated element-wise multiply.</summary>
    public static Column<double> GpuMultiply(this Column<double> a, Column<double> b)
    {
        var result = Ops.Multiply(a.Values.ToArray(), b.Values.ToArray());
        return new Column<double>(a.Name, result);
    }

    /// <summary>GPU-accelerated scalar multiply.</summary>
    public static Column<double> GpuMultiply(this Column<double> a, double scalar)
    {
        var result = Ops.MultiplyScalar(a.Values.ToArray(), scalar);
        return new Column<double>(a.Name, result);
    }

    /// <summary>GPU-accelerated element-wise subtract.</summary>
    public static Column<double> GpuSubtract(this Column<double> a, Column<double> b)
    {
        var result = Ops.Subtract(a.Values.ToArray(), b.Values.ToArray());
        return new Column<double>(a.Name, result);
    }

    /// <summary>GPU-accelerated element-wise divide.</summary>
    public static Column<double> GpuDivide(this Column<double> a, Column<double> b)
    {
        var result = Ops.Divide(a.Values.ToArray(), b.Values.ToArray());
        return new Column<double>(a.Name, result);
    }

    /// <summary>GPU-accelerated sum, min, max in one pass.</summary>
    public static (double Sum, double Min, double Max) GpuSumMinMax(this Column<double> col)
    {
        return Ops.SumMinMax(col.Values.ToArray());
    }

    /// <summary>GPU-accelerated variance.</summary>
    public static double GpuVariance(this Column<double> col)
    {
        var data = col.Values.ToArray();
        double mean = 0;
        for (int i = 0; i < data.Length; i++) mean += data[i];
        mean /= data.Length;
        return Ops.Variance(data, mean);
    }

    // ═══════════════════════════════════════════════════════
    // Matrix / Tensor GPU operations
    // ═══════════════════════════════════════════════════════

    /// <summary>GPU-accelerated matrix multiply for Tensor<double>.</summary>
    public static Tensor<double> GpuMatMul(this Tensor<double> A, Tensor<double> B)
    {
        if (A.Rank != 2 || B.Rank != 2)
            throw new ArgumentException("Both tensors must be 2D for matrix multiplication.");
        if (A.Shape[1] != B.Shape[0])
            throw new ArgumentException($"Shape mismatch: ({A.Shape[0]}×{A.Shape[1]}) × ({B.Shape[0]}×{B.Shape[1]})");

        int m = A.Shape[0], k = A.Shape[1], n = B.Shape[1];
        var result = Ops.MatMul(A.ToArray(), B.ToArray(), m, k, n);
        return new Tensor<double>(result, m, n);
    }

    /// <summary>GPU-accelerated Gram matrix: C = A^T × A.</summary>
    public static Tensor<double> GpuGramMatrix(this Tensor<double> A)
    {
        if (A.Rank != 2)
            throw new ArgumentException("Tensor must be 2D.");
        int n = A.Shape[0], d = A.Shape[1];
        var result = Ops.GramMatrix(A.ToArray(), n, d);
        return new Tensor<double>(result, d, d);
    }

    /// <summary>GPU-accelerated pairwise distance matrix.</summary>
    public static Tensor<double> GpuPairwiseDistances(this Tensor<double> X, Tensor<double> Y)
    {
        if (X.Rank != 2 || Y.Rank != 2)
            throw new ArgumentException("Both tensors must be 2D.");
        if (X.Shape[1] != Y.Shape[1])
            throw new ArgumentException("Feature dimensions must match.");

        int nX = X.Shape[0], nY = Y.Shape[0], d = X.Shape[1];
        var result = Ops.PairwiseDistances(X.ToArray(), Y.ToArray(), nX, nY, d);
        return new Tensor<double>(result, nX, nY);
    }

    // ═══════════════════════════════════════════════════════
    // DataFrame GPU operations
    // ═══════════════════════════════════════════════════════

    /// <summary>
    /// GPU-accelerated correlation matrix for all numeric columns.
    /// Uses GPU Gram matrix for the core computation.
    /// </summary>
    public static DataFrame GpuCorr(this DataFrame df)
    {
        var numericCols = new List<(string Name, double[] Data)>();
        foreach (var name in df.ColumnNames)
        {
            if (df[name] is Column<double> colD)
                numericCols.Add((name, colD.Values.ToArray()));
            else if (df[name] is Column<int> colI)
                numericCols.Add((name, colI.Values.ToArray().Select(v => (double)v).ToArray()));
            else if (df[name] is Column<float> colF)
                numericCols.Add((name, colF.Values.ToArray().Select(v => (double)v).ToArray()));
        }

        if (numericCols.Count == 0)
            return new DataFrame(Array.Empty<IColumn>());

        int n = df.RowCount;
        int k = numericCols.Count;
        var data = new double[n * k];

        // Build matrix + center columns
        var means = new double[k];
        var stds = new double[k];

        for (int c = 0; c < k; c++)
        {
            var col = numericCols[c].Data;
            double sum = 0;
            for (int i = 0; i < n; i++) sum += col[i];
            means[c] = sum / n;

            double ss = 0;
            for (int i = 0; i < n; i++)
            {
                double centered = col[i] - means[c];
                data[i * k + c] = centered;
                ss += centered * centered;
            }
            stds[c] = n > 1 ? Math.Sqrt(ss / (n - 1)) : double.NaN;
        }

        // GPU Gram matrix: X^T @ X
        var gram = Ops.GramMatrix(data, n, k);

        // Normalize to correlation
        var corrMatrix = new double[k * k];
        for (int i = 0; i < k; i++)
            for (int j = 0; j < k; j++)
            {
                if (n <= 1 || double.IsNaN(stds[i]) || double.IsNaN(stds[j]))
                {
                    corrMatrix[i * k + j] = double.NaN;
                }
                else
                {
                    double denom = stds[i] * stds[j] * (n - 1);
                    corrMatrix[i * k + j] = denom > 0 ? gram[i * k + j] / denom : 0;
                }
            }

        // Build DataFrame
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", numericCols.Select(c => c.Name).ToArray()));
        for (int j = 0; j < k; j++)
        {
            var colData = new double[k];
            for (int i = 0; i < k; i++) colData[i] = corrMatrix[i * k + j];
            columns.Add(new Column<double>(numericCols[j].Name, colData));
        }

        return new DataFrame(columns);
    }

    /// <summary>Get information about the current GPU device.</summary>
    public static string GpuInfo() => Ops.DeviceInfo;

    /// <summary>Whether a real GPU (non-CPU fallback) is available.</summary>
    public static bool GpuAvailable() => Ops.HasGpu;
}
