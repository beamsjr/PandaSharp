using Cortex.ML.Native;
using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Principal Component Analysis via SVD (preferred) or eigendecomposition of the covariance matrix.
/// Reduces dimensionality by projecting data onto the top principal components.
/// Uses LAPACK SVD via Apple Accelerate when available, falling back to Jacobi eigenvalue algorithm.
/// </summary>
public class PCA
{
    /// <summary>Number of principal components to keep.</summary>
    public int NComponents { get; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Principal components (eigenvectors) of shape (n_components, n_features). Available after fitting.</summary>
    public Tensor<double>? Components { get; private set; }

    /// <summary>Variance explained by each component. Available after fitting.</summary>
    public double[]? ExplainedVariance { get; private set; }

    /// <summary>Proportion of variance explained by each component. Available after fitting.</summary>
    public double[]? ExplainedVarianceRatio { get; private set; }

    /// <summary>Per-feature mean computed during fitting.</summary>
    public double[]? Mean { get; private set; }

    /// <summary>Create a PCA model.</summary>
    /// <param name="nComponents">Number of components to retain.</param>
    public PCA(int nComponents)
    {
        if (nComponents < 1)
            throw new ArgumentOutOfRangeException(nameof(nComponents), "nComponents must be >= 1.");
        NComponents = nComponents;
    }

    /// <summary>Fit the PCA model: compute mean, center data, and decompose via SVD or eigendecomposition.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public PCA Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];
        if (NComponents > d)
            throw new ArgumentException($"nComponents ({NComponents}) cannot exceed n_features ({d}).");

        var xData = X.Span;

        // Compute mean
        var mean = new double[d];
        for (int i = 0; i < n; i++)
        {
            int off = i * d;
            for (int j = 0; j < d; j++)
                mean[j] += xData[off + j];
        }
        for (int j = 0; j < d; j++)
            mean[j] /= n;

        // Center data
        var centered = new double[n * d];
        for (int i = 0; i < n; i++)
        {
            int off = i * d;
            for (int j = 0; j < d; j++)
                centered[off + j] = xData[off + j] - mean[j];
        }

        // Try SVD approach: X_c = U @ diag(S) @ VT
        // Principal components = rows of VT (top k)
        // Explained variance = S^2 / (n-1)
        if (FitViaSvd(centered, n, d, mean))
            return this;

        // Fallback: covariance matrix + Jacobi eigendecomposition
        FitViaJacobi(centered, n, d, mean);
        return this;
    }

    /// <summary>
    /// Attempt PCA via LAPACK SVD of the centered data matrix.
    /// Returns true on success.
    /// </summary>
    private bool FitViaSvd(double[] centered, int n, int d, double[] mean)
    {
        int mn = Math.Min(n, d);
        var A = new double[n * d];
        Array.Copy(centered, A, n * d);
        var S = new double[mn];
        var U = new double[n * mn];
        var VT = new double[mn * d];

        if (!BlasOps.Dgesvd(A, S, U, VT, n, d))
            return false;

        // Extract top NComponents
        double scale = n > 1 ? 1.0 / (n - 1) : 1.0;
        var components = new double[NComponents * d];
        var explainedVar = new double[NComponents];

        double totalVar = 0;
        for (int i = 0; i < mn; i++)
            totalVar += S[i] * S[i] * scale;

        for (int c = 0; c < NComponents; c++)
        {
            explainedVar[c] = S[c] * S[c] * scale;
            // VT row c = principal component c
            Array.Copy(VT, c * d, components, c * d, d);
        }

        var explainedVarRatio = new double[NComponents];
        if (totalVar > 0)
        {
            for (int c = 0; c < NComponents; c++)
                explainedVarRatio[c] = explainedVar[c] / totalVar;
        }

        Mean = mean;
        Components = new Tensor<double>(components, NComponents, d);
        ExplainedVariance = explainedVar;
        ExplainedVarianceRatio = explainedVarRatio;
        IsFitted = true;
        return true;
    }

    /// <summary>
    /// PCA via covariance matrix eigendecomposition (Jacobi method).
    /// Uses BLAS dgemm for covariance computation when available.
    /// </summary>
    private void FitViaJacobi(double[] centered, int n, int d, double[] mean)
    {
        // Compute covariance matrix: (1/(n-1)) * X_c^T * X_c using BLAS
        // Build X_c^T (d×n)
        var centeredT = new double[d * n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                centeredT[j * n + i] = centered[i * d + j];

        var cov = new double[d * d];
        BlasOps.Dgemm(centeredT, centered, cov, d, d, n);

        double scale = n > 1 ? 1.0 / (n - 1) : 1.0;
        for (int i = 0; i < d * d; i++)
            cov[i] *= scale;

        // Eigendecompose using Jacobi method
        var (eigenvalues, eigenvectors) = JacobiEigen(cov, d);

        // Sort by descending eigenvalue
        var indices = Enumerable.Range(0, d).ToArray();
        Array.Sort(indices, (a, b) => eigenvalues[b].CompareTo(eigenvalues[a]));

        // Extract top NComponents
        var components = new double[NComponents * d];
        var explainedVar = new double[NComponents];
        double totalVar = 0;
        for (int i = 0; i < d; i++)
            totalVar += Math.Max(0, eigenvalues[i]);

        for (int c = 0; c < NComponents; c++)
        {
            int srcIdx = indices[c];
            explainedVar[c] = Math.Max(0, eigenvalues[srcIdx]);
            for (int j = 0; j < d; j++)
                components[c * d + j] = eigenvectors[j * d + srcIdx]; // column srcIdx of eigenvectors
        }

        var explainedVarRatio = new double[NComponents];
        if (totalVar > 0)
        {
            for (int c = 0; c < NComponents; c++)
                explainedVarRatio[c] = explainedVar[c] / totalVar;
        }

        Mean = mean;
        Components = new Tensor<double>(components, NComponents, d);
        ExplainedVariance = explainedVar;
        ExplainedVarianceRatio = explainedVarRatio;
        IsFitted = true;
    }

    /// <summary>Project data onto the principal components.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Transformed data of shape (n_samples, n_components).</returns>
    public Tensor<double> Transform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int n = X.Shape[0];
        int d = X.Shape[1];

        // Center the data
        var xData = X.Span;
        var centered = new double[n * d];
        for (int i = 0; i < n; i++)
        {
            int off = i * d;
            for (int j = 0; j < d; j++)
                centered[off + j] = xData[off + j] - Mean![j];
        }

        // Project: result = X_centered @ Components^T using BLAS
        // Components is NComponents×d, Components^T is d×NComponents
        var compArr = Components!.ToArray();
        var compT = new double[d * NComponents];
        for (int i = 0; i < NComponents; i++)
            for (int j = 0; j < d; j++)
                compT[j * NComponents + i] = compArr[i * d + j];

        var result = new double[n * NComponents];
        BlasOps.Dgemm(centered, compT, result, n, NComponents, d);

        return new Tensor<double>(result, n, NComponents);
    }

    /// <summary>Fit the model and transform the data in one step.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Transformed data of shape (n_samples, n_components).</returns>
    public Tensor<double> FitTransform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        Fit(X);
        return Transform(X);
    }

    // -- Jacobi eigenvalue algorithm for symmetric matrices --

    /// <summary>
    /// Jacobi eigenvalue algorithm for a real symmetric matrix.
    /// Returns eigenvalues and eigenvectors (stored column-wise).
    /// </summary>
    private static (double[] Eigenvalues, double[] Eigenvectors) JacobiEigen(double[] matrix, int n,
        int maxIter = 1000, double tolerance = 1e-12)
    {
        // Copy matrix (will be modified in place)
        var a = new double[n * n];
        Array.Copy(matrix, a, n * n);

        // Initialize eigenvectors to identity
        var v = new double[n * n];
        for (int i = 0; i < n; i++)
            v[i * n + i] = 1.0;

        var newA = new double[n * n];

        for (int iter = 0; iter < maxIter; iter++)
        {
            // Find largest off-diagonal element
            double maxOffDiag = 0;
            int p = 0, q = 1;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double val = Math.Abs(a[i * n + j]);
                    if (val > maxOffDiag)
                    {
                        maxOffDiag = val;
                        p = i;
                        q = j;
                    }
                }
            }

            if (maxOffDiag < tolerance) break;

            // Compute rotation
            double app = a[p * n + p];
            double aqq = a[q * n + q];
            double apq = a[p * n + q];

            double theta;
            if (Math.Abs(app - aqq) < 1e-30)
            {
                theta = Math.PI / 4.0;
            }
            else
            {
                theta = 0.5 * Math.Atan2(2.0 * apq, app - aqq);
            }

            double cosT = Math.Cos(theta);
            double sinT = Math.Sin(theta);

            // Apply Jacobi rotation to matrix A
            Array.Copy(a, newA, n * n);

            for (int i = 0; i < n; i++)
            {
                if (i == p || i == q) continue;

                double aip = a[i * n + p];
                double aiq = a[i * n + q];

                newA[i * n + p] = cosT * aip + sinT * aiq;
                newA[p * n + i] = newA[i * n + p];
                newA[i * n + q] = -sinT * aip + cosT * aiq;
                newA[q * n + i] = newA[i * n + q];
            }

            newA[p * n + p] = cosT * cosT * app + 2.0 * sinT * cosT * apq + sinT * sinT * aqq;
            newA[q * n + q] = sinT * sinT * app - 2.0 * sinT * cosT * apq + cosT * cosT * aqq;
            newA[p * n + q] = 0;
            newA[q * n + p] = 0;

            Array.Copy(newA, a, n * n);

            // Update eigenvectors
            for (int i = 0; i < n; i++)
            {
                double vip = v[i * n + p];
                double viq = v[i * n + q];
                v[i * n + p] = cosT * vip + sinT * viq;
                v[i * n + q] = -sinT * vip + cosT * viq;
            }
        }

        // Extract eigenvalues from diagonal
        var eigenvalues = new double[n];
        for (int i = 0; i < n; i++)
            eigenvalues[i] = a[i * n + i];

        return (eigenvalues, v);
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("PCA has not been fitted. Call Fit() first.");
    }
}
