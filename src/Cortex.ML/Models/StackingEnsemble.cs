using Cortex.ML.ModelSelection;
using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Stacking (stacked generalisation) ensemble.
/// Base models produce out-of-fold predictions via K-Fold cross-validation.
/// A meta-learner is trained on the stacked out-of-fold feature matrix.
/// At predict time, base model predictions are fed to the meta-learner.
/// Folds are computed in parallel.
/// </summary>
public class StackingEnsemble : IModel
{
    private readonly IModel[] _baseModels;
    private readonly IModel _metaLearner;
    private readonly int _nFolds;
    private readonly int? _seed;

    /// <summary>Fitted copies of base models (trained on full data after stacking).</summary>
    private IModel[]? _fittedBaseModels;

    public string Name => "StackingEnsemble";
    public bool IsFitted { get; private set; }

    /// <summary>Create a stacking ensemble.</summary>
    /// <param name="baseModels">Base learners whose predictions form the meta-feature matrix.</param>
    /// <param name="metaLearner">Second-level model trained on stacked predictions.</param>
    /// <param name="nFolds">Number of CV folds for generating out-of-fold predictions (default 5).</param>
    /// <param name="seed">Random seed for fold shuffling.</param>
    public StackingEnsemble(
        IModel[] baseModels,
        IModel metaLearner,
        int nFolds = 5,
        int? seed = null)
    {
        if (baseModels.Length == 0)
            throw new ArgumentException("At least one base model is required.", nameof(baseModels));
        if (nFolds < 2)
            throw new ArgumentOutOfRangeException(nameof(nFolds), "Must have at least 2 folds.");

        _baseModels = baseModels;
        _metaLearner = metaLearner;
        _nFolds = nFolds;
        _seed = seed;
    }

    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int nSamples = X.Shape[0];
        int nBaseModels = _baseModels.Length;

        if (nSamples < _nFolds)
            throw new ArgumentException(
                $"Cannot have {_nFolds} folds with only {nSamples} samples.", nameof(X));

        // Generate shuffled indices
        var indices = Enumerable.Range(0, nSamples).ToArray();
        var rng = _seed.HasValue ? new Random(_seed.Value) : Random.Shared;
        for (int i = nSamples - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Out-of-fold predictions matrix: nSamples x nBaseModels
        var oofPredictions = new double[nSamples * nBaseModels];

        int foldSize = nSamples / _nFolds;

        // Process folds sequentially to avoid data races on shared base model instances.
        // IModel.Fit() returns `this` (mutates in place), so concurrent folds on the
        // same model object would corrupt intermediate state.
        for (int fold = 0; fold < _nFolds; fold++)
        {
            int valStart = fold * foldSize;
            int valEnd = fold == _nFolds - 1 ? nSamples : valStart + foldSize;

            var valIndices = indices[valStart..valEnd];
            var trainIndices = indices[..valStart].Concat(indices[valEnd..]).ToArray();

            var xTrain = CrossValidation.SliceRows(X, trainIndices);
            var yTrain = CrossValidation.SliceElements(y, trainIndices);
            var xVal = CrossValidation.SliceRows(X, valIndices);

            for (int m = 0; m < nBaseModels; m++)
            {
                var foldModel = _baseModels[m].Fit(xTrain, yTrain);
                var preds = foldModel.Predict(xVal);
                var predSpan = preds.Span;

                for (int i = 0; i < valIndices.Length; i++)
                {
                    int sampleIdx = valIndices[i];
                    oofPredictions[sampleIdx * nBaseModels + m] = predSpan[i];
                }
            }
        }

        // Train meta-learner on out-of-fold predictions
        var metaX = new Tensor<double>(oofPredictions, nSamples, nBaseModels);
        _metaLearner.Fit(metaX, y);

        // Refit all base models on full training data for prediction
        _fittedBaseModels = new IModel[nBaseModels];
        Parallel.For(0, nBaseModels, m =>
        {
            _fittedBaseModels[m] = _baseModels[m].Fit(X, y);
        });

        IsFitted = true;
        return this;
    }

    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();

        int n = X.Shape[0];
        int nBaseModels = _fittedBaseModels!.Length;

        // Collect base model predictions in parallel
        var basePreds = new Tensor<double>[nBaseModels];
        Parallel.For(0, nBaseModels, m =>
        {
            basePreds[m] = _fittedBaseModels[m].Predict(X);
        });

        // Stack predictions into meta-feature matrix
        var metaData = new double[n * nBaseModels];
        for (int m = 0; m < nBaseModels; m++)
        {
            var span = basePreds[m].Span;
            for (int i = 0; i < n; i++)
                metaData[i * nBaseModels + m] = span[i];
        }

        var metaX = new Tensor<double>(metaData, n, nBaseModels);
        return _metaLearner.Predict(metaX);
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

        // Default to R² score
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

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
