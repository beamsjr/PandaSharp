using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Linear regression with combined L1 and L2 regularization (Elastic Net).
/// Uses coordinate descent to solve the penalised least squares problem:
/// min (1/2n) ||y - Xw||² + Alpha * L1Ratio * ||w||₁ + (Alpha * (1-L1Ratio) / 2) * ||w||²
/// </summary>
public class ElasticNet : IModel
{
    /// <inheritdoc />
    public string Name => "ElasticNet";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>Overall regularization strength.</summary>
    public double Alpha { get; set; }

    /// <summary>Balance between L1 and L2 penalty. 0 = pure Ridge, 1 = pure Lasso.</summary>
    public double L1Ratio { get; set; }

    /// <summary>Maximum number of coordinate descent iterations.</summary>
    public int MaxIterations { get; set; }

    /// <summary>Convergence tolerance on coefficient change.</summary>
    public double Tolerance { get; set; }

    /// <summary>Fitted weight vector (one per feature, excluding intercept).</summary>
    public Tensor<double>? Weights { get; private set; }

    /// <summary>Fitted intercept (bias) term.</summary>
    public double Intercept { get; private set; }

    /// <summary>Create an Elastic Net model.</summary>
    /// <param name="alpha">Overall penalty strength.</param>
    /// <param name="l1Ratio">Mixing parameter: 0=Ridge, 1=Lasso, between=ElasticNet.</param>
    /// <param name="maxIterations">Maximum coordinate descent iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    public ElasticNet(double alpha = 1.0, double l1Ratio = 0.5, int maxIterations = 1000, double tolerance = 1e-6)
    {
        Alpha = alpha;
        L1Ratio = l1Ratio;
        MaxIterations = maxIterations;
        Tolerance = tolerance;
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
        var xSpan = X.Span;
        var ySpan = y.Span;

        // Centre y to handle intercept separately
        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += ySpan[i];
        yMean /= n;

        // Column means for X
        var xMeans = new double[p];
        for (int j = 0; j < p; j++)
        {
            double sum = 0;
            for (int i = 0; i < n; i++)
                sum += xSpan[i * p + j];
            xMeans[j] = sum / n;
        }

        // Precompute column norms (sum of squares of centred columns)
        var colNormSq = new double[p];
        for (int j = 0; j < p; j++)
        {
            double ss = 0;
            for (int i = 0; i < n; i++)
            {
                double v = xSpan[i * p + j] - xMeans[j];
                ss += v * v;
            }
            colNormSq[j] = ss;
        }

        double l1 = Alpha * L1Ratio;
        double l2 = Alpha * (1.0 - L1Ratio);

        var w = new double[p];
        var residual = new double[n];
        for (int i = 0; i < n; i++)
            residual[i] = ySpan[i] - yMean;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            double maxChange = 0;

            for (int j = 0; j < p; j++)
            {
                double oldW = w[j];

                // Compute partial residual correlation
                double rho = 0;
                for (int i = 0; i < n; i++)
                {
                    double xij = xSpan[i * p + j] - xMeans[j];
                    rho += xij * (residual[i] + oldW * xij);
                }

                // Soft thresholding
                double denom = colNormSq[j] + n * l2;
                double newW;
                if (denom == 0)
                {
                    newW = 0;
                }
                else
                {
                    newW = SoftThreshold(rho, n * l1) / denom;
                }

                // Update residual
                if (newW != oldW)
                {
                    double diff = newW - oldW;
                    for (int i = 0; i < n; i++)
                    {
                        double xij = xSpan[i * p + j] - xMeans[j];
                        residual[i] -= diff * xij;
                    }

                    double change = Math.Abs(diff);
                    if (change > maxChange) maxChange = change;
                }

                w[j] = newW;
            }

            if (maxChange < Tolerance) break;
        }

        // Compute intercept from means
        double intercept = yMean;
        for (int j = 0; j < p; j++)
            intercept -= w[j] * xMeans[j];

        Intercept = intercept;
        Weights = new Tensor<double>((double[])w.Clone(), p);
        IsFitted = true;
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        int n = X.Shape[0];
        int p = X.Shape[1];
        var xSpan = X.Span;
        var w = Weights!.Span;
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            double sum = Intercept;
            int off = i * p;
            for (int j = 0; j < p; j++)
                sum += xSpan[off + j] * w[j];
            result[i] = sum;
        }

        return new Tensor<double>(result, n);
    }

    /// <summary>Returns R² score.</summary>
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        EnsureFitted();
        if (X.Shape[0] == 0) return 0.0;
        var pred = Predict(X);
        var pSpan = pred.Span;
        var ySpan = y.Span;
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

    /// <summary>Soft-thresholding operator: sign(x) * max(|x| - threshold, 0).</summary>
    internal static double SoftThreshold(double x, double threshold)
    {
        if (x > threshold) return x - threshold;
        if (x < -threshold) return x + threshold;
        return 0.0;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
