using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Random forest classifier using bagged ensemble of <see cref="DecisionTreeClassifier"/> trees.
/// Supports bootstrap sampling, feature sub-sampling, and parallel tree training.
/// </summary>
public class RandomForestClassifier : IClassifier
{
    private readonly int _nEstimators;
    private readonly int _maxDepth;
    private readonly int _maxFeatures;
    private readonly bool _bootstrap;
    private readonly int _seed;
    private readonly SplitCriterion _criterion;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;

    private DecisionTreeClassifier[]? _trees;
    private int _numClasses;

    /// <inheritdoc />
    public string Name => "RandomForestClassifier";

    /// <inheritdoc />
    public bool IsFitted => _trees is not null;

    /// <inheritdoc />
    public int NumClasses => _numClasses;

    /// <summary>
    /// Creates a new random forest classifier.
    /// </summary>
    /// <param name="nEstimators">Number of trees in the forest.</param>
    /// <param name="maxDepth">Maximum depth per tree. 0 means unlimited.</param>
    /// <param name="maxFeatures">Maximum features per split. Negative means sqrt(n_features).</param>
    /// <param name="bootstrap">Whether to use bootstrap sampling.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="criterion">Split criterion for individual trees.</param>
    /// <param name="minSamplesSplit">Minimum samples to split a node.</param>
    /// <param name="minSamplesLeaf">Minimum samples in a leaf.</param>
    public RandomForestClassifier(
        int nEstimators = 100,
        int maxDepth = 0,
        int maxFeatures = -1,
        bool bootstrap = true,
        int seed = 42,
        SplitCriterion criterion = SplitCriterion.Gini,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1)
    {
        if (nEstimators < 1)
            throw new ArgumentOutOfRangeException(nameof(nEstimators), "nEstimators must be >= 1.");
        _nEstimators = nEstimators;
        _maxDepth = maxDepth;
        _maxFeatures = maxFeatures;
        _bootstrap = bootstrap;
        _seed = seed;
        _criterion = criterion;
        _minSamplesSplit = minSamplesSplit;
        _minSamplesLeaf = minSamplesLeaf;
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

        int nSamples = X.Shape[0];
        int nFeatures = X.Shape[1];
        var xData = X.ToArray();
        var yData = y.ToArray();

        // Determine number of classes
        _numClasses = 0;
        for (int i = 0; i < nSamples; i++)
        {
            int c = (int)yData[i];
            if (c + 1 > _numClasses) _numClasses = c + 1;
        }

        // Compute maxFeatures: -1 means sqrt(n_features)
        int maxFeat = _maxFeatures < 0 ? (int)Math.Ceiling(Math.Sqrt(nFeatures)) : _maxFeatures;
        if (maxFeat == 0) maxFeat = nFeatures;

        _trees = new DecisionTreeClassifier[_nEstimators];

        Parallel.For(0, _nEstimators, new ParallelOptions(), treeIdx =>
        {
            var rng = new Random(_seed + treeIdx);

            double[] bsX, bsY;
            int bsN;

            if (_bootstrap)
            {
                // Bootstrap sample
                bsN = nSamples;
                bsX = new double[bsN * nFeatures];
                bsY = new double[bsN];
                for (int i = 0; i < bsN; i++)
                {
                    int si = rng.Next(nSamples);
                    Array.Copy(xData, si * nFeatures, bsX, i * nFeatures, nFeatures);
                    bsY[i] = yData[si];
                }
            }
            else
            {
                bsN = nSamples;
                bsX = xData;
                bsY = yData;
            }

            var tree = new DecisionTreeClassifier(
                criterion: _criterion,
                maxDepth: _maxDepth,
                minSamplesSplit: _minSamplesSplit,
                minSamplesLeaf: _minSamplesLeaf,
                maxFeatures: maxFeat,
                seed: _seed + treeIdx);

            var xTensor = new Tensor<double>(bsX, bsN, nFeatures);
            var yTensor = new Tensor<double>(bsY, bsN);
            tree.Fit(xTensor, yTensor);

            _trees[treeIdx] = tree;
        });

        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (!IsFitted) throw new InvalidOperationException("Model has not been fitted.");

        var proba = PredictProba(X);
        int nSamples = X.Shape[0];
        var result = new double[nSamples];
        var probaArr = proba.ToArray();

        for (int i = 0; i < nSamples; i++)
        {
            int bestClass = 0;
            double bestProb = probaArr[i * _numClasses];
            for (int c = 1; c < _numClasses; c++)
            {
                double p = probaArr[i * _numClasses + c];
                if (p > bestProb)
                {
                    bestProb = p;
                    bestClass = c;
                }
            }

            result[i] = bestClass;
        }

        return new Tensor<double>(result, nSamples);
    }

    /// <inheritdoc />
    public Tensor<double> PredictProba(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (!IsFitted) throw new InvalidOperationException("Model has not been fitted.");

        int nSamples = X.Shape[0];
        int nFeatures = X.Shape[1];
        var xData = X.ToArray();
        var result = new double[nSamples * _numClasses];

        // Aggregate probabilities from all trees
        foreach (var tree in _trees!)
        {
            for (int i = 0; i < nSamples; i++)
            {
                var leaf = tree.PredictSingleLeaf(xData, i, nFeatures);
                int len = Math.Min(leaf.ClassDistribution.Length, _numClasses);
                for (int c = 0; c < len; c++)
                    result[i * _numClasses + c] += leaf.ClassDistribution[c];
            }
        }

        // Average
        double invN = 1.0 / _nEstimators;
        for (int i = 0; i < result.Length; i++)
            result[i] *= invN;

        return new Tensor<double>(result, nSamples, _numClasses);
    }

    /// <inheritdoc />
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Shape[0] == 0) return 0.0;
        var preds = Predict(X);
        var predsArr = preds.ToArray();
        var yArr = y.ToArray();
        int correct = 0;
        for (int i = 0; i < yArr.Length; i++)
        {
            if ((int)predsArr[i] == (int)yArr[i]) correct++;
        }

        return (double)correct / yArr.Length;
    }
}
