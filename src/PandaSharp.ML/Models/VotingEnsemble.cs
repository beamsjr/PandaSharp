using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>Voting strategy for classifier ensemble.</summary>
public enum VotingStrategy
{
    /// <summary>Majority vote on predicted class labels.</summary>
    Hard,
    /// <summary>Average predicted probabilities, pick argmax.</summary>
    Soft
}

/// <summary>
/// Ensemble classifier that combines multiple classifiers via voting.
/// Hard voting picks the class with the most votes across models.
/// Soft voting averages predicted probabilities and picks argmax.
/// All models are predicted in parallel.
/// </summary>
public class VotingEnsemble : IClassifier
{
    private readonly IClassifier[] _models;
    private readonly VotingStrategy _strategy;
    private readonly double[]? _weights;

    public string Name => "VotingEnsemble";
    public bool IsFitted { get; private set; }
    public int NumClasses { get; private set; }

    /// <summary>Create a voting ensemble.</summary>
    /// <param name="models">Classifiers to combine.</param>
    /// <param name="strategy">Hard (majority vote) or Soft (probability averaging).</param>
    /// <param name="weights">Optional per-model weights (must match model count). Null = equal weights.</param>
    public VotingEnsemble(
        IClassifier[] models,
        VotingStrategy strategy = VotingStrategy.Hard,
        double[]? weights = null)
    {
        if (models.Length == 0)
            throw new ArgumentException("At least one model is required.", nameof(models));
        if (weights is not null && weights.Length != models.Length)
            throw new ArgumentException("Weights length must match number of models.", nameof(weights));

        _models = models;
        _strategy = strategy;
        _weights = weights;
    }

    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);

        // Fit all models in parallel
        Parallel.For(0, _models.Length, i =>
        {
            _models[i].Fit(X, y);
        });

        NumClasses = _models[0].NumClasses;
        IsFitted = true;
        return this;
    }

    public Tensor<double> PredictProba(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();

        int n = X.Shape[0];
        int nc = NumClasses;

        // Collect probabilities from all models in parallel
        var allProba = new Tensor<double>[_models.Length];
        Parallel.For(0, _models.Length, i =>
        {
            allProba[i] = _models[i].PredictProba(X);
        });

        // Weighted average of probabilities
        var result = new double[n * nc];
        double totalWeight = 0;

        for (int m = 0; m < _models.Length; m++)
        {
            double w = _weights is not null ? _weights[m] : 1.0;
            totalWeight += w;
            var span = allProba[m].Span;
            for (int i = 0; i < n * nc; i++)
                result[i] += w * span[i];
        }

        // Normalise
        if (totalWeight > 0)
        {
            for (int i = 0; i < result.Length; i++)
                result[i] /= totalWeight;
        }

        return new Tensor<double>(result, n, nc);
    }

    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();

        if (_strategy == VotingStrategy.Soft)
            return PredictFromProba(PredictProba(X));

        // Hard voting: each model votes for a class, majority wins
        int n = X.Shape[0];

        // Collect predictions from all models in parallel
        var allPreds = new Tensor<double>[_models.Length];
        Parallel.For(0, _models.Length, i =>
        {
            allPreds[i] = _models[i].Predict(X);
        });

        var result = new double[n];
        for (int i = 0; i < n; i++)
        {
            // Count weighted votes per class label
            var votes = new Dictionary<double, double>();
            for (int m = 0; m < _models.Length; m++)
            {
                double label = allPreds[m].Span[i];
                double w = _weights is not null ? _weights[m] : 1.0;
                votes.TryGetValue(label, out double current);
                votes[label] = current + w;
            }

            // Pick class with most votes
            double bestLabel = 0;
            double bestVotes = double.NegativeInfinity;
            foreach (var (label, count) in votes)
            {
                if (count > bestVotes)
                {
                    bestVotes = count;
                    bestLabel = label;
                }
            }
            result[i] = bestLabel;
        }

        return new Tensor<double>(result, n);
    }

    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        EnsureFitted();

        var pred = Predict(X);
        var pSpan = pred.Span;
        var ySpan = y.Span;
        int n = y.Length;
        int correct = 0;
        for (int i = 0; i < n; i++)
            if (Math.Abs(pSpan[i] - ySpan[i]) < 1e-9) correct++;
        return (double)correct / n;
    }

    private Tensor<double> PredictFromProba(Tensor<double> proba)
    {
        int n = proba.Shape[0];
        int nc = proba.Shape[1];
        var result = new double[n];
        var span = proba.Span;

        for (int i = 0; i < n; i++)
        {
            int bestIdx = 0;
            double bestVal = span[i * nc];
            for (int c = 1; c < nc; c++)
            {
                double v = span[i * nc + c];
                if (v > bestVal) { bestVal = v; bestIdx = c; }
            }
            result[i] = bestIdx;
        }

        return new Tensor<double>(result, n);
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
