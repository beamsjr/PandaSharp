using PandaSharp.ML.Native;
using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Logistic regression classifier supporting binary and multi-class (one-vs-rest) classification.
/// Uses gradient descent optimisation with configurable learning rate, iteration limit, and convergence tolerance.
/// Uses BLAS for matrix-vector products when available via Apple Accelerate.
/// </summary>
public class LogisticRegression : IClassifier
{
    /// <inheritdoc />
    public string Name => "LogisticRegression";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <inheritdoc />
    public int NumClasses { get; private set; }

    /// <summary>Learning rate for gradient descent.</summary>
    public double LearningRate { get; set; }

    /// <summary>Maximum number of gradient descent iterations.</summary>
    public int MaxIterations { get; set; }

    /// <summary>Convergence tolerance on loss change.</summary>
    public double Tolerance { get; set; }

    /// <summary>L2 regularization strength (0 = none).</summary>
    public double L2Penalty { get; set; }

    // Internal state: for binary, _weights[0] is shape (nFeatures+1).
    // For multi-class OVR, _weights[c] is shape (nFeatures+1) per class.
    private double[][]? _weights;
    private double[]? _classes;

    /// <summary>Create a logistic regression classifier.</summary>
    /// <param name="learningRate">Step size for gradient descent.</param>
    /// <param name="maxIterations">Maximum optimisation iterations.</param>
    /// <param name="tolerance">Early stop when loss improvement is below this.</param>
    /// <param name="l2Penalty">L2 regularization strength.</param>
    public LogisticRegression(double learningRate = 0.01, int maxIterations = 1000, double tolerance = 1e-6, double l2Penalty = 0.0)
    {
        LearningRate = learningRate;
        MaxIterations = maxIterations;
        Tolerance = tolerance;
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
        var ySpan = y.Span;

        // Discover classes
        var classSet = new SortedSet<double>();
        for (int i = 0; i < y.Length; i++) classSet.Add(ySpan[i]);
        _classes = [.. classSet];
        NumClasses = _classes.Length;

        // Build augmented X with bias
        var xAug = BuildAugmented(X, n, p);
        int pAug = p + 1;

        if (NumClasses == 2)
        {
            // Binary logistic regression
            _weights = new double[1][];
            _weights[0] = FitBinary(xAug, y, n, pAug, _classes[1]);
        }
        else
        {
            // One-vs-rest
            _weights = new double[NumClasses][];
            for (int c = 0; c < NumClasses; c++)
                _weights[c] = FitBinary(xAug, y, n, pAug, _classes[c]);
        }

        IsFitted = true;
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> PredictProba(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        int n = X.Shape[0];
        int p = X.Shape[1];
        var xAug = BuildAugmented(X, n, p);
        int pAug = p + 1;

        if (NumClasses == 2)
        {
            var proba = new double[n * 2];
            // Compute z = xAug @ w using BLAS
            var z = new double[n];
            BlasOps.Dgemv(xAug, _weights![0], z, n, pAug);
            for (int i = 0; i < n; i++)
            {
                double p1 = Sigmoid(z[i]);
                proba[i * 2] = 1.0 - p1;
                proba[i * 2 + 1] = p1;
            }
            return new Tensor<double>(proba, n, 2);
        }
        else
        {
            // OVR: raw sigmoid per class, then normalise
            var proba = new double[n * NumClasses];
            var z = new double[n];
            for (int c = 0; c < NumClasses; c++)
            {
                Array.Clear(z);
                BlasOps.Dgemv(xAug, _weights![c], z, n, pAug);
                for (int i = 0; i < n; i++)
                    proba[i * NumClasses + c] = Sigmoid(z[i]);
            }
            // Normalise to sum to 1
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int c = 0; c < NumClasses; c++)
                    sum += proba[i * NumClasses + c];
                if (sum > 0)
                    for (int c = 0; c < NumClasses; c++)
                        proba[i * NumClasses + c] /= sum;
            }
            return new Tensor<double>(proba, n, NumClasses);
        }
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        var proba = PredictProba(X);
        int n = proba.Shape[0];
        int nc = proba.Shape[1];
        var result = new double[n];
        var pSpan = proba.Span;

        for (int i = 0; i < n; i++)
        {
            int bestIdx = 0;
            double bestVal = pSpan[i * nc];
            for (int c = 1; c < nc; c++)
            {
                double v = pSpan[i * nc + c];
                if (v > bestVal) { bestVal = v; bestIdx = c; }
            }
            result[i] = _classes![bestIdx];
        }

        return new Tensor<double>(result, n);
    }

    /// <summary>Returns classification accuracy.</summary>
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
        int correct = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(pSpan[i] - ySpan[i]) < 1e-9) correct++;
        return (double)correct / n;
    }

    // -- Private helpers --

    private double[] FitBinary(double[] xAug, Tensor<double> y, int n, int pAug, double positiveClass)
    {
        var ySpan = y.Span;
        var w = new double[pAug];
        var grad = new double[pAug];
        var z = new double[n];
        var err = new double[n];
        double prevLoss = double.MaxValue;

        // Precompute binary labels
        var yi = new double[n];
        for (int i = 0; i < n; i++)
            yi[i] = (Math.Abs(ySpan[i] - positiveClass) < 1e-9) ? 1.0 : 0.0;

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            // z = X_aug @ w using BLAS
            Array.Clear(z);
            BlasOps.Dgemv(xAug, w, z, n, pAug);

            // Compute sigmoid, error, and loss (scalar ops)
            double loss = 0;
            for (int i = 0; i < n; i++)
            {
                double prob = Math.Clamp(Sigmoid(z[i]), 1e-15, 1.0 - 1e-15);
                err[i] = prob - yi[i];
                loss += -yi[i] * Math.Log(prob) - (1.0 - yi[i]) * Math.Log(1.0 - prob);
            }

            // grad = X_aug^T @ err using BLAS
            Array.Clear(grad);
            BlasOps.DgemvT(xAug, err, grad, n, pAug);

            // Average and apply L2
            double invN = 1.0 / n;
            loss *= invN;
            for (int j = 0; j < pAug; j++)
            {
                grad[j] *= invN;
                if (L2Penalty > 0 && j > 0) // skip bias
                {
                    grad[j] += L2Penalty * w[j];
                    loss += 0.5 * L2Penalty * w[j] * w[j];
                }
            }

            // Update weights
            for (int j = 0; j < pAug; j++)
                w[j] -= LearningRate * grad[j];

            if (Math.Abs(prevLoss - loss) < Tolerance) break;
            prevLoss = loss;
        }

        return w;
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

    private static double Sigmoid(double z)
    {
        if (z >= 0)
        {
            double ez = Math.Exp(-z);
            return 1.0 / (1.0 + ez);
        }
        else
        {
            double ez = Math.Exp(z);
            return ez / (1.0 + ez);
        }
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
