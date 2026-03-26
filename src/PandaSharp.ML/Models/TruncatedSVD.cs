using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Truncated Singular Value Decomposition (SVD) via randomized power iteration.
/// Unlike PCA, this does not center the data, making it suitable for sparse matrices.
/// Finds the top-k singular vectors to project data into a lower-dimensional space.
/// </summary>
public class TruncatedSVD
{
    /// <summary>Number of singular vectors (components) to extract.</summary>
    public int NComponents { get; }

    /// <summary>Number of power iterations for improving approximation accuracy.</summary>
    public int NIter { get; }

    /// <summary>Random seed for reproducibility.</summary>
    public int? Seed { get; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Right singular vectors of shape (n_components, n_features). Available after fitting.</summary>
    public Tensor<double>? Components { get; private set; }

    /// <summary>Variance explained by each component. Available after fitting.</summary>
    public double[]? ExplainedVariance { get; private set; }

    /// <summary>Proportion of total variance explained by each component. Available after fitting.</summary>
    public double[]? ExplainedVarianceRatio { get; private set; }

    /// <summary>Singular values corresponding to each component. Available after fitting.</summary>
    public double[]? SingularValues { get; private set; }

    /// <summary>Create a TruncatedSVD model.</summary>
    /// <param name="nComponents">Number of components to retain.</param>
    /// <param name="nIter">Number of power iterations (default 5).</param>
    /// <param name="seed">Random seed (default null).</param>
    public TruncatedSVD(int nComponents, int nIter = 5, int? seed = null)
    {
        if (nComponents < 1)
            throw new ArgumentOutOfRangeException(nameof(nComponents), "nComponents must be >= 1.");
        if (nIter < 0)
            throw new ArgumentOutOfRangeException(nameof(nIter), "nIter must be >= 0.");

        NComponents = nComponents;
        NIter = nIter;
        Seed = seed;
    }

    /// <summary>Fit the TruncatedSVD model to the feature matrix X (no centering).</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public TruncatedSVD Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];

        if (NComponents > Math.Min(n, d))
            throw new ArgumentException($"nComponents ({NComponents}) cannot exceed min(n_samples, n_features) = {Math.Min(n, d)}.");

        // Randomized SVD via power iteration
        // 1. Generate random matrix Omega of shape (d, NComponents)
        var rng = Seed.HasValue ? new Random(Seed.Value) : new Random();
        var omega = new double[d * NComponents];
        for (int i = 0; i < omega.Length; i++)
            omega[i] = rng.NextGaussian();

        var omegaTensor = new Tensor<double>(omega, d, NComponents);

        // 2. Form Y = X * Omega  (n x NComponents)
        var Y = X.MatMul(omegaTensor);

        // 3. Power iteration: repeatedly form Y = X * (X^T * Y) to sharpen singular vectors
        for (int iter = 0; iter < NIter; iter++)
        {
            // Q = QR(Y), then Y = X * X^T * Q
            Y = QROrthogonalize(Y);
            var XtY = X.Transpose().MatMul(Y);   // d x k
            Y = X.MatMul(XtY);                    // n x k
        }

        // 4. Orthogonalize Y via QR
        Y = QROrthogonalize(Y); // Q: n x k

        // 5. Form B = Q^T * X  (k x d)
        var B = Y.Transpose().MatMul(X);

        // 6. SVD of small matrix B via eigendecompose B * B^T
        var BBt = B.MatMul(B.Transpose()); // k x k
        var bbtData = BBt.ToArray();
        int k = NComponents;
        var (eigenvalues, eigenvectors) = JacobiEigen(bbtData, k);

        // Sort by descending eigenvalue
        var indices = Enumerable.Range(0, k).ToArray();
        Array.Sort(indices, (a, b) => eigenvalues[b].CompareTo(eigenvalues[a]));

        // Singular values = sqrt(eigenvalues of B*B^T)
        var singularValues = new double[k];
        for (int i = 0; i < k; i++)
            singularValues[i] = Math.Sqrt(Math.Max(0, eigenvalues[indices[i]]));

        // Right singular vectors: V = B^T * U * Sigma^{-1}
        // First extract U from eigenvectors (columns)
        var uSmall = new double[k * k]; // U columns from eigenvectors
        for (int c = 0; c < k; c++)
        {
            int srcCol = indices[c];
            for (int r = 0; r < k; r++)
                uSmall[r * k + c] = eigenvectors[r * k + srcCol];
        }

        // Components (V^T) = diag(1/sigma) * U^T * B, shape (k x d)
        var components = new double[k * d];
        var bData = B.ToArray();
        for (int i = 0; i < k; i++)
        {
            double invSigma = singularValues[i] > 1e-15 ? 1.0 / singularValues[i] : 0.0;
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int m = 0; m < k; m++)
                    sum += uSmall[m * k + i] * bData[m * d + j]; // U^T row i = U col i transposed
                components[i * d + j] = sum * invSigma;
            }
        }

        // Compute explained variance
        var xData = X.ToArray();
        double totalVariance = 0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
            {
                double v = xData[i * d + j];
                totalVariance += v * v;
            }
        totalVariance /= n > 1 ? n - 1 : 1;

        var explainedVar = new double[k];
        var explainedVarRatio = new double[k];
        // Project data and compute per-component variance
        var componentsTensor = new Tensor<double>(components, k, d);
        var projected = X.MatMul(componentsTensor.Transpose()); // n x k
        var projData = projected.ToArray();

        for (int c = 0; c < k; c++)
        {
            double mean = 0;
            for (int i = 0; i < n; i++)
                mean += projData[i * k + c];
            mean /= n;

            double var_ = 0;
            for (int i = 0; i < n; i++)
            {
                double diff = projData[i * k + c] - mean;
                var_ += diff * diff;
            }
            explainedVar[c] = var_ / (n > 1 ? n - 1 : 1);
            explainedVarRatio[c] = totalVariance > 0 ? explainedVar[c] / totalVariance : 0;
        }

        Components = componentsTensor;
        SingularValues = singularValues;
        ExplainedVariance = explainedVar;
        ExplainedVarianceRatio = explainedVarRatio;
        IsFitted = true;
        return this;
    }

    /// <summary>Project data onto the truncated singular vectors (no centering).</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Transformed data of shape (n_samples, n_components).</returns>
    public Tensor<double> Transform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        return X.MatMul(Components!.Transpose());
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

    // -- Helpers --

    /// <summary>Modified Gram-Schmidt QR orthogonalization, returning Q.</summary>
    private static Tensor<double> QROrthogonalize(Tensor<double> A)
    {
        int m = A.Shape[0];
        int n = A.Shape[1];
        var q = A.ToArray();

        for (int j = 0; j < n; j++)
        {
            // Normalize column j
            double norm = 0;
            for (int i = 0; i < m; i++)
                norm += q[i * n + j] * q[i * n + j];
            norm = Math.Sqrt(norm);

            if (norm > 1e-15)
            {
                for (int i = 0; i < m; i++)
                    q[i * n + j] /= norm;
            }

            // Orthogonalize subsequent columns against column j
            for (int k = j + 1; k < n; k++)
            {
                double dot = 0;
                for (int i = 0; i < m; i++)
                    dot += q[i * n + j] * q[i * n + k];
                for (int i = 0; i < m; i++)
                    q[i * n + k] -= dot * q[i * n + j];
            }
        }

        return new Tensor<double>(q, m, n);
    }

    /// <summary>
    /// Jacobi eigenvalue algorithm for a real symmetric matrix.
    /// Returns eigenvalues and eigenvectors (stored column-wise).
    /// </summary>
    private static (double[] Eigenvalues, double[] Eigenvectors) JacobiEigen(double[] matrix, int n,
        int maxIter = 1000, double tolerance = 1e-12)
    {
        var a = new double[n * n];
        Array.Copy(matrix, a, n * n);

        var v = new double[n * n];
        for (int i = 0; i < n; i++)
            v[i * n + i] = 1.0;

        var newA = new double[n * n];

        for (int iter = 0; iter < maxIter; iter++)
        {
            double maxOffDiag = 0;
            int p = 0, q2 = 1;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double val = Math.Abs(a[i * n + j]);
                    if (val > maxOffDiag)
                    {
                        maxOffDiag = val;
                        p = i;
                        q2 = j;
                    }
                }
            }

            if (maxOffDiag < tolerance) break;

            double app = a[p * n + p];
            double aqq = a[q2 * n + q2];
            double apq = a[p * n + q2];

            double theta = Math.Abs(app - aqq) < 1e-30
                ? Math.PI / 4.0
                : 0.5 * Math.Atan2(2.0 * apq, app - aqq);

            double cosT = Math.Cos(theta);
            double sinT = Math.Sin(theta);

            Array.Copy(a, newA, n * n);

            for (int i = 0; i < n; i++)
            {
                if (i == p || i == q2) continue;
                double aip = a[i * n + p];
                double aiq = a[i * n + q2];
                newA[i * n + p] = cosT * aip + sinT * aiq;
                newA[p * n + i] = newA[i * n + p];
                newA[i * n + q2] = -sinT * aip + cosT * aiq;
                newA[q2 * n + i] = newA[i * n + q2];
            }

            newA[p * n + p] = cosT * cosT * app + 2.0 * sinT * cosT * apq + sinT * sinT * aqq;
            newA[q2 * n + q2] = sinT * sinT * app - 2.0 * sinT * cosT * apq + cosT * cosT * aqq;
            newA[p * n + q2] = 0;
            newA[q2 * n + p] = 0;

            Array.Copy(newA, a, n * n);

            for (int i = 0; i < n; i++)
            {
                double vip = v[i * n + p];
                double viq = v[i * n + q2];
                v[i * n + p] = cosT * vip + sinT * viq;
                v[i * n + q2] = -sinT * vip + cosT * viq;
            }
        }

        var eigenvalues = new double[n];
        for (int i = 0; i < n; i++)
            eigenvalues[i] = a[i * n + i];

        return (eigenvalues, v);
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("TruncatedSVD has not been fitted. Call Fit() first.");
    }
}

/// <summary>
/// Extension methods for Random to support Gaussian sampling.
/// </summary>
internal static class RandomExtensions
{
    /// <summary>Generates a sample from the standard normal distribution using Box-Muller.</summary>
    public static double NextGaussian(this Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
