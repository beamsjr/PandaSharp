using PandaSharp.ML.Native;
using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Ordinary least squares linear regression with optional L2 (Ridge) penalty.
/// Solves β = (X^T X + λI)^(-1) X^T y via the normal equations.
/// A bias (intercept) column is added automatically.
/// Uses BLAS/LAPACK via Apple Accelerate when available, with managed fallbacks.
/// </summary>
public class LinearRegression : IModel
{
    /// <inheritdoc />
    public string Name => "LinearRegression";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>L2 regularization strength. 0 = plain OLS, &gt;0 = Ridge regression.</summary>
    public double L2Penalty { get; set; }

    /// <summary>Fitted weight vector (one per feature, excluding intercept).</summary>
    public Tensor<double>? Weights { get; private set; }

    /// <summary>Fitted intercept (bias) term.</summary>
    public double Intercept { get; private set; }

    /// <summary>Create a linear regression model.</summary>
    /// <param name="l2Penalty">L2 penalty strength (0 for OLS).</param>
    public LinearRegression(double l2Penalty = 0.0)
    {
        L2Penalty = l2Penalty;
    }

    /// <inheritdoc />
    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));
        if (y.Length != X.Shape[0])
            throw new ArgumentException("y length must match number of samples in X.", nameof(y));

        int n = X.Shape[0];
        int p = X.Shape[1];
        int pAug = p + 1;

        // Build augmented matrix with bias column: [1 | X] as flat array
        var xAug = BuildAugmentedArray(X, n, p);

        // Compute X^T X  (pAug x pAug) using BLAS dgemm: XtX = X_aug^T @ X_aug
        var xtx = new double[pAug * pAug];
        BlasOps.Dgemm(xAug, xAug, xtx, pAug, pAug, n, alpha: 1.0, beta: 0.0);
        // The above computes xAug^T(pAug×n) @ xAug(n×pAug) but Dgemm expects row-major A(m×k)*B(k×n)
        // We need xtx = xAug^T @ xAug. xAug is n×pAug.
        // Use: treat A = xAug^T (pAug×n), B = xAug (n×pAug) → but we don't have xAug^T stored.
        // Instead: xtx[i,j] = sum_s xAug[s,i]*xAug[s,j] which is the same as computing
        // with transposed xAug. Let's build xAugT explicitly.
        var xAugT = new double[pAug * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < pAug; j++)
                xAugT[j * n + i] = xAug[i * pAug + j];

        // XtX = xAugT(pAug×n) @ xAug(n×pAug) = pAug×pAug
        Array.Clear(xtx);
        BlasOps.Dgemm(xAugT, xAug, xtx, pAug, pAug, n);

        // Add L2 penalty to diagonal (skip intercept at index 0)
        if (L2Penalty > 0)
        {
            for (int i = 1; i < pAug; i++)
                xtx[i * pAug + i] += L2Penalty;
        }

        // Compute X^T y using BLAS dgemv: xty = xAugT(pAug×n) @ y(n)
        var yArr = y.ToArray();
        var xty = new double[pAug];
        BlasOps.Dgemv(xAugT, yArr, xty, pAug, n);

        // Try LAPACK dgesv first, fall back to Gauss-Jordan
        var beta = SolveNormalEquation(xtx, xty, pAug);

        Intercept = beta[0];
        var weights = new double[p];
        Array.Copy(beta, 1, weights, 0, p);
        Weights = new Tensor<double>(weights, p);
        IsFitted = true;
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int n = X.Shape[0];
        int p = X.Shape[1];
        var w = Weights!.ToArray();
        var xArr = X.ToArray();
        var result = new double[n];

        // y = X @ w using BLAS dgemv
        BlasOps.Dgemv(xArr, w, result, n, p);

        // Add intercept
        for (int i = 0; i < n; i++)
            result[i] += Intercept;

        return new Tensor<double>(result, n);
    }

    /// <summary>
    /// Computes R² = 1 - SS_res / SS_tot.
    /// </summary>
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        EnsureFitted();
        if (X.Shape[0] == 0) return 0.0;
        var pred = Predict(X);
        var ySpan = y.Span;
        var pSpan = pred.Span;
        int n = y.Length;

        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += ySpan[i];
        yMean /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = ySpan[i] - pSpan[i];
            ssRes += diff * diff;
            double diffMean = ySpan[i] - yMean;
            ssTot += diffMean * diffMean;
        }

        return ssTot == 0 ? (ssRes == 0 ? 1.0 : 0.0) : 1.0 - ssRes / ssTot;
    }

    // -- Private helpers --

    private static double[] BuildAugmentedArray(Tensor<double> X, int n, int p)
    {
        int pAug = p + 1;
        var data = new double[n * pAug];
        var xSpan = X.Span;
        for (int i = 0; i < n; i++)
        {
            data[i * pAug] = 1.0; // bias
            int srcOff = i * p;
            int dstOff = i * pAug + 1;
            for (int j = 0; j < p; j++)
                data[dstOff + j] = xSpan[srcOff + j];
        }
        return data;
    }

    /// <summary>
    /// Attempts LAPACK dgesv, falls back to Gauss-Jordan.
    /// </summary>
    private static double[] SolveNormalEquation(double[] xtx, double[] xty, int size)
    {
        // Try LAPACK LU solve
        var aCopy = new double[size * size];
        Array.Copy(xtx, aCopy, xtx.Length);
        var bCopy = new double[size];
        Array.Copy(xty, bCopy, xty.Length);

        if (BlasOps.Dgesv(aCopy, bCopy, size))
            return bCopy;

        // Fallback: Gauss-Jordan
        return SolveGaussJordan(xtx, xty, size);
    }

    /// <summary>
    /// Solves A * x = b via Gauss-Jordan elimination with partial pivoting.
    /// A is (size x size), b is (size). Returns x as double[].
    /// </summary>
    private static double[] SolveGaussJordan(double[] A, double[] b, int size)
    {
        // Build augmented matrix [A | b]
        var aug = new double[size * (size + 1)];
        int cols = size + 1;

        for (int i = 0; i < size; i++)
        {
            int srcRow = i * size;
            int dstRow = i * cols;
            for (int j = 0; j < size; j++)
                aug[dstRow + j] = A[srcRow + j];
            aug[dstRow + size] = b[i];
        }

        // Forward elimination with partial pivoting
        for (int col = 0; col < size; col++)
        {
            // Find pivot
            int maxRow = col;
            double maxVal = Math.Abs(aug[col * cols + col]);
            for (int row = col + 1; row < size; row++)
            {
                double val = Math.Abs(aug[row * cols + col]);
                if (val > maxVal) { maxVal = val; maxRow = row; }
            }

            // Swap rows
            if (maxRow != col)
            {
                for (int j = 0; j < cols; j++)
                {
                    int a1 = col * cols + j, a2 = maxRow * cols + j;
                    (aug[a1], aug[a2]) = (aug[a2], aug[a1]);
                }
            }

            double pivot = aug[col * cols + col];
            // Use relative tolerance scaled by max absolute value in the column
            double maxAbsInColumn = maxVal; // maxVal from partial pivoting search above
            if (maxAbsInColumn == 0 || Math.Abs(pivot) < 1e-12 * maxAbsInColumn)
                throw new InvalidOperationException("Singular matrix encountered in normal equations. Consider adding L2 regularization.");

            // Scale pivot row
            double invPivot = 1.0 / pivot;
            for (int j = col; j < cols; j++)
                aug[col * cols + j] *= invPivot;

            // Eliminate column
            for (int row = 0; row < size; row++)
            {
                if (row == col) continue;
                double factor = aug[row * cols + col];
                if (factor == 0) continue;
                for (int j = col; j < cols; j++)
                    aug[row * cols + j] -= factor * aug[col * cols + j];
            }
        }

        var result = new double[size];
        for (int i = 0; i < size; i++)
            result[i] = aug[i * cols + size];
        return result;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
