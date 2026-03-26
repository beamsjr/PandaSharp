using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Random forest regressor using bagged ensemble of <see cref="DecisionTreeRegressor"/> trees.
/// Supports bootstrap sampling, feature sub-sampling, and parallel tree training.
/// </summary>
public class RandomForestRegressor : IModel
{
    private readonly int _nEstimators;
    private readonly int _maxDepth;
    private readonly int _maxFeatures;
    private readonly bool _bootstrap;
    private readonly int _seed;
    private readonly RegressorCriterion _criterion;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;

    private DecisionTreeRegressor[]? _trees;

    /// <inheritdoc />
    public string Name => "RandomForestRegressor";

    /// <inheritdoc />
    public bool IsFitted => _trees is not null;

    /// <summary>
    /// Creates a new random forest regressor.
    /// </summary>
    /// <param name="nEstimators">Number of trees in the forest.</param>
    /// <param name="maxDepth">Maximum depth per tree. 0 means unlimited.</param>
    /// <param name="maxFeatures">Maximum features per split. Negative means sqrt(n_features).</param>
    /// <param name="bootstrap">Whether to use bootstrap sampling.</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="criterion">Split criterion for individual trees.</param>
    /// <param name="minSamplesSplit">Minimum samples to split a node.</param>
    /// <param name="minSamplesLeaf">Minimum samples in a leaf.</param>
    public RandomForestRegressor(
        int nEstimators = 100,
        int maxDepth = 0,
        int maxFeatures = -1,
        bool bootstrap = true,
        int seed = 42,
        RegressorCriterion criterion = RegressorCriterion.MSE,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1)
    {
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

        int maxFeat = _maxFeatures < 0 ? (int)Math.Ceiling(Math.Sqrt(nFeatures)) : _maxFeatures;
        if (maxFeat == 0) maxFeat = nFeatures;

        _trees = new DecisionTreeRegressor[_nEstimators];

        Parallel.For(0, _nEstimators, new ParallelOptions(), treeIdx =>
        {
            var rng = new Random(_seed + treeIdx);

            double[] bsX, bsY;
            int bsN;

            if (_bootstrap)
            {
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

            var tree = new DecisionTreeRegressor(
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

        int nSamples = X.Shape[0];
        int nFeatures = X.Shape[1];
        var xData = X.ToArray();
        var result = new double[nSamples];

        foreach (var tree in _trees!)
        {
            for (int i = 0; i < nSamples; i++)
                result[i] += tree.PredictSingleValue(xData, i, nFeatures);
        }

        double invN = 1.0 / _nEstimators;
        for (int i = 0; i < nSamples; i++)
            result[i] *= invN;

        return new Tensor<double>(result, nSamples);
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
        int n = yArr.Length;

        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += yArr[i];
        yMean /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = yArr[i] - predsArr[i];
            ssRes += diff * diff;
            double diffMean = yArr[i] - yMean;
            ssTot += diffMean * diffMean;
        }

        return ssTot == 0 ? 1.0 : 1.0 - ssRes / ssTot;
    }
}
