using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>Loss function for SGD regression.</summary>
public enum RegressionLoss
{
    /// <summary>Standard squared error loss.</summary>
    SquaredError,
    /// <summary>Huber loss: squared for small errors, linear for large.</summary>
    Huber,
    /// <summary>Epsilon-insensitive loss (SVR-style): no penalty within epsilon band.</summary>
    EpsilonInsensitive
}

/// <summary>
/// Stochastic gradient descent regressor with configurable loss function (SquaredError, Huber, EpsilonInsensitive),
/// learning rate schedule, and mini-batch support.
/// </summary>
public class SGDRegressor : IModel
{
    /// <inheritdoc />
    public string Name => "SGDRegressor";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>Loss function to optimise.</summary>
    public RegressionLoss Loss { get; set; }

    /// <summary>Learning rate schedule.</summary>
    public LearningRateSchedule Schedule { get; set; }

    /// <summary>Initial learning rate.</summary>
    public double Eta0 { get; set; }

    /// <summary>Exponent for inverse scaling schedule.</summary>
    public double PowerT { get; set; }

    /// <summary>Maximum number of epochs over the training data.</summary>
    public int MaxEpochs { get; set; }

    /// <summary>Mini-batch size (1 = pure SGD).</summary>
    public int BatchSize { get; set; }

    /// <summary>L2 regularization strength.</summary>
    public double Alpha { get; set; }

    /// <summary>Convergence tolerance on loss change between epochs.</summary>
    public double Tolerance { get; set; }

    /// <summary>Epsilon threshold for Huber and EpsilonInsensitive losses.</summary>
    public double Epsilon { get; set; }

    /// <summary>Random seed for shuffling.</summary>
    public int Seed { get; set; }

    /// <summary>Fitted weight vector (one per feature, excluding intercept).</summary>
    public Tensor<double>? Weights { get; private set; }

    /// <summary>Fitted intercept (bias) term.</summary>
    public double Intercept { get; private set; }

    private double[]? _w;

    /// <summary>Create an SGD regressor.</summary>
    public SGDRegressor(
        RegressionLoss loss = RegressionLoss.SquaredError,
        LearningRateSchedule schedule = LearningRateSchedule.InvScaling,
        double eta0 = 0.01,
        double powerT = 0.25,
        int maxEpochs = 1000,
        int batchSize = 1,
        double alpha = 0.0001,
        double tolerance = 1e-6,
        double epsilon = 0.1,
        int seed = 42)
    {
        Loss = loss;
        Schedule = schedule;
        Eta0 = eta0;
        PowerT = powerT;
        MaxEpochs = maxEpochs;
        BatchSize = batchSize;
        Alpha = alpha;
        Tolerance = tolerance;
        Epsilon = epsilon;
        Seed = seed;
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
        var xAug = BuildAugmented(X, n, p);
        var ySpan = y.Span;

        _w = new double[pAug];
        var rng = new Random(Seed);
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        double prevLoss = double.MaxValue;
        int t = 1;
        var grad = new double[pAug];

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Shuffle indices
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            double epochLoss = 0;

            for (int bStart = 0; bStart < n; bStart += BatchSize)
            {
                int bEnd = Math.Min(bStart + BatchSize, n);
                int bSize = bEnd - bStart;
                double eta = GetLearningRate(t);
                t++;

                Array.Clear(grad);

                for (int bi = bStart; bi < bEnd; bi++)
                {
                    int idx = indices[bi];
                    double pred = 0;
                    int off = idx * pAug;
                    for (int j = 0; j < pAug; j++)
                        pred += xAug[off + j] * _w[j];

                    double residual = pred - ySpan[idx];
                    ComputeGradient(xAug, off, pAug, residual, grad, ref epochLoss);
                }

                double invB = 1.0 / bSize;
                for (int j = 0; j < pAug; j++)
                {
                    double g = grad[j] * invB;
                    if (Alpha > 0 && j > 0) // skip bias
                        g += Alpha * _w[j];
                    _w[j] -= eta * g;
                }
            }

            epochLoss /= n;
            if (Math.Abs(prevLoss - epochLoss) < Tolerance) break;
            prevLoss = epochLoss;
        }

        Intercept = _w[0];
        var weights = new double[p];
        Array.Copy(_w, 1, weights, 0, p);
        Weights = new Tensor<double>(weights, p);
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
        var result = new double[n];

        for (int i = 0; i < n; i++)
        {
            double sum = Intercept;
            int off = i * p;
            var w = Weights!.Span;
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

        return ssTot == 0 ? 1.0 : 1.0 - ssRes / ssTot;
    }

    // -- Private --

    private void ComputeGradient(double[] xAug, int off, int pAug, double residual, double[] grad, ref double loss)
    {
        switch (Loss)
        {
            case RegressionLoss.SquaredError:
                loss += 0.5 * residual * residual;
                for (int j = 0; j < pAug; j++)
                    grad[j] += residual * xAug[off + j];
                break;

            case RegressionLoss.Huber:
            {
                double absR = Math.Abs(residual);
                if (absR <= Epsilon)
                {
                    loss += 0.5 * residual * residual;
                    for (int j = 0; j < pAug; j++)
                        grad[j] += residual * xAug[off + j];
                }
                else
                {
                    loss += Epsilon * (absR - 0.5 * Epsilon);
                    double sign = residual > 0 ? 1.0 : -1.0;
                    for (int j = 0; j < pAug; j++)
                        grad[j] += Epsilon * sign * xAug[off + j];
                }
                break;
            }

            case RegressionLoss.EpsilonInsensitive:
            {
                double absR = Math.Abs(residual);
                if (absR > Epsilon)
                {
                    loss += absR - Epsilon;
                    double sign = residual > 0 ? 1.0 : -1.0;
                    for (int j = 0; j < pAug; j++)
                        grad[j] += sign * xAug[off + j];
                }
                break;
            }
        }
    }

    private double GetLearningRate(int t)
    {
        return Schedule switch
        {
            LearningRateSchedule.Constant => Eta0,
            LearningRateSchedule.InvScaling => Eta0 / Math.Pow(t, PowerT),
            _ => Eta0
        };
    }

    private static double[] BuildAugmented(Tensor<double> X, int n, int p)
    {
        int pAug = p + 1;
        var data = new double[n * pAug];
        var xSpan = X.Span;
        for (int i = 0; i < n; i++)
        {
            data[i * pAug] = 1.0;
            int srcOff = i * p;
            int dstOff = i * pAug + 1;
            for (int j = 0; j < p; j++)
                data[dstOff + j] = xSpan[srcOff + j];
        }
        return data;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
