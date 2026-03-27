using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// L1-regularized linear regression (Lasso). Thin wrapper around <see cref="ElasticNet"/> with L1Ratio = 1.0.
/// Uses coordinate descent with soft-thresholding to promote sparse solutions.
/// </summary>
public class Lasso : IModel
{
    private readonly ElasticNet _inner;

    /// <inheritdoc />
    public string Name => "Lasso";

    /// <inheritdoc />
    public bool IsFitted => _inner.IsFitted;

    /// <summary>L1 penalty strength.</summary>
    public double Alpha
    {
        get => _inner.Alpha;
        set => _inner.Alpha = value;
    }

    /// <summary>Maximum number of coordinate descent iterations.</summary>
    public int MaxIterations
    {
        get => _inner.MaxIterations;
        set => _inner.MaxIterations = value;
    }

    /// <summary>Convergence tolerance on coefficient change.</summary>
    public double Tolerance
    {
        get => _inner.Tolerance;
        set => _inner.Tolerance = value;
    }

    /// <summary>Fitted weight vector (one per feature, excluding intercept).</summary>
    public Tensor<double>? Weights => _inner.Weights;

    /// <summary>Fitted intercept (bias) term.</summary>
    public double Intercept => _inner.Intercept;

    /// <summary>Create a Lasso regression model.</summary>
    /// <param name="alpha">L1 penalty strength.</param>
    /// <param name="maxIterations">Maximum coordinate descent iterations.</param>
    /// <param name="tolerance">Convergence tolerance.</param>
    public Lasso(double alpha = 1.0, int maxIterations = 1000, double tolerance = 1e-6)
    {
        _inner = new ElasticNet(alpha: alpha, l1Ratio: 1.0, maxIterations: maxIterations, tolerance: tolerance);
    }

    /// <inheritdoc />
    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        _inner.Fit(X, y);
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        return _inner.Predict(X);
    }

    /// <summary>Returns R² score.</summary>
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Shape[0] == 0) return 0.0;
        return _inner.Score(X, y);
    }
}
