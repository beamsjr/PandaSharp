using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>Loss function for SGD classification.</summary>
public enum ClassificationLoss
{
    /// <summary>Hinge loss (linear SVM).</summary>
    Hinge,
    /// <summary>Logistic (log) loss.</summary>
    Log,
    /// <summary>Modified Huber loss (smooth hinge).</summary>
    ModifiedHuber
}

/// <summary>Learning rate schedule for SGD models.</summary>
public enum LearningRateSchedule
{
    /// <summary>Constant learning rate.</summary>
    Constant,
    /// <summary>Inverse scaling: eta = eta0 / t^power_t.</summary>
    InvScaling
}

/// <summary>
/// Stochastic gradient descent classifier with configurable loss function (Hinge, Log, ModifiedHuber),
/// learning rate schedule, and mini-batch support. Supports binary and multi-class (one-vs-rest).
/// </summary>
public class SGDClassifier : IClassifier
{
    /// <inheritdoc />
    public string Name => "SGDClassifier";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <inheritdoc />
    public int NumClasses { get; private set; }

    /// <summary>Loss function to optimise.</summary>
    public ClassificationLoss Loss { get; set; }

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

    /// <summary>Random seed for shuffling.</summary>
    public int Seed { get; set; }

    private double[][]? _weights;
    private double[]? _classes;

    /// <summary>Create an SGD classifier.</summary>
    public SGDClassifier(
        ClassificationLoss loss = ClassificationLoss.Hinge,
        LearningRateSchedule schedule = LearningRateSchedule.InvScaling,
        double eta0 = 0.01,
        double powerT = 0.25,
        int maxEpochs = 1000,
        int batchSize = 1,
        double alpha = 0.0001,
        double tolerance = 1e-6,
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
        var ySpan = y.Span;

        var classSet = new SortedSet<double>();
        for (int i = 0; i < y.Length; i++) classSet.Add(ySpan[i]);
        _classes = [.. classSet];
        NumClasses = _classes.Length;

        var xAug = BuildAugmented(X, n, p);
        int pAug = p + 1;

        if (NumClasses == 2)
        {
            _weights = [FitBinary(xAug, y, n, pAug, _classes[1])];
        }
        else
        {
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
            for (int i = 0; i < n; i++)
            {
                double z = DotRow(xAug, i, pAug, _weights![0]);
                double p1 = ToProba(z);
                proba[i * 2] = 1.0 - p1;
                proba[i * 2 + 1] = p1;
            }
            return new Tensor<double>(proba, n, 2);
        }
        else
        {
            var proba = new double[n * NumClasses];
            for (int i = 0; i < n; i++)
            {
                double sum = 0;
                for (int c = 0; c < NumClasses; c++)
                {
                    double z = DotRow(xAug, i, pAug, _weights![c]);
                    double pc = ToProba(z);
                    proba[i * NumClasses + c] = pc;
                    sum += pc;
                }
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

    // -- Private --

    private double[] FitBinary(double[] xAug, Tensor<double> y, int n, int pAug, double positiveClass)
    {
        var ySpan = y.Span;
        var w = new double[pAug];
        var rng = new Random(Seed);
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        double prevLoss = double.MaxValue;
        int t = 1;
        var grad = new double[pAug];

        for (int epoch = 0; epoch < MaxEpochs; epoch++)
        {
            // Shuffle
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }

            double epochLoss = 0;
            int batches = 0;

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
                    double z = 0;
                    int off = idx * pAug;
                    for (int j = 0; j < pAug; j++)
                        z += xAug[off + j] * w[j];

                    double yi = (Math.Abs(ySpan[idx] - positiveClass) < 1e-9) ? 1.0 : -1.0;
                    double margin = yi * z;

                    ComputeGradient(xAug, off, pAug, yi, z, margin, grad, ref epochLoss);
                }

                double invB = 1.0 / bSize;
                for (int j = 0; j < pAug; j++)
                {
                    double g = grad[j] * invB;
                    if (Alpha > 0 && j > 0)
                        g += Alpha * w[j];
                    w[j] -= eta * g;
                }

                batches++;
            }

            epochLoss /= n;
            if (Math.Abs(prevLoss - epochLoss) < Tolerance) break;
            prevLoss = epochLoss;
        }

        return w;
    }

    private void ComputeGradient(double[] xAug, int off, int pAug, double yi, double z, double margin, double[] grad, ref double loss)
    {
        switch (Loss)
        {
            case ClassificationLoss.Hinge:
                if (margin < 1.0)
                {
                    loss += 1.0 - margin;
                    for (int j = 0; j < pAug; j++)
                        grad[j] -= yi * xAug[off + j];
                }
                break;

            case ClassificationLoss.Log:
            {
                double prob = Math.Clamp(Sigmoid(z), 1e-15, 1.0 - 1e-15);
                double yi01 = (yi > 0) ? 1.0 : 0.0;
                loss += -yi01 * Math.Log(prob) - (1.0 - yi01) * Math.Log(1.0 - prob);
                double err = prob - yi01;
                for (int j = 0; j < pAug; j++)
                    grad[j] += err * xAug[off + j];
                break;
            }

            case ClassificationLoss.ModifiedHuber:
                if (margin >= 1.0)
                {
                    // no gradient
                }
                else if (margin >= -1.0)
                {
                    loss += (1.0 - margin) * (1.0 - margin);
                    double factor = -2.0 * yi * (1.0 - margin);
                    for (int j = 0; j < pAug; j++)
                        grad[j] += factor * xAug[off + j];
                }
                else
                {
                    loss += -4.0 * margin;
                    for (int j = 0; j < pAug; j++)
                        grad[j] += -4.0 * yi * xAug[off + j];
                }
                break;
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

    private double ToProba(double z)
    {
        return Loss switch
        {
            ClassificationLoss.Log => Sigmoid(z),
            ClassificationLoss.ModifiedHuber => Math.Clamp(Sigmoid(z), 0.0, 1.0),
            // Hinge: use Platt-style sigmoid approximation
            _ => Sigmoid(z)
        };
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

    private static double DotRow(double[] xAug, int row, int pAug, double[] w)
    {
        double sum = 0;
        int off = row * pAug;
        for (int j = 0; j < pAug; j++)
            sum += xAug[off + j] * w[j];
        return sum;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
