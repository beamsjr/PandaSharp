using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>Loss function for gradient boosted regression.</summary>
public enum BoostingLoss
{
    /// <summary>Least squares (L2) loss.</summary>
    SquaredError,

    /// <summary>Least absolute deviation (L1) loss.</summary>
    AbsoluteError,

    /// <summary>Huber loss (combination of L1 and L2).</summary>
    Huber
}

/// <summary>
/// Gradient boosted tree regressor using sequential additive training of shallow
/// <see cref="DecisionTreeRegressor"/> trees with configurable loss functions.
/// </summary>
public class GradientBoostedTreeRegressor : IModel
{
    private readonly int _nEstimators;
    private readonly double _learningRate;
    private readonly int _maxDepth;
    private readonly double _subsample;
    private readonly int _seed;
    private readonly BoostingLoss _loss;
    private readonly double _huberAlpha;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;

    private double _initialPrediction;
    private List<DecisionTreeRegressor>? _trees;
    private int _numFeatures;

    /// <inheritdoc />
    public string Name => "GradientBoostedTreeRegressor";

    /// <inheritdoc />
    public bool IsFitted => _trees is not null;

    /// <summary>
    /// Creates a new gradient boosted tree regressor.
    /// </summary>
    /// <param name="nEstimators">Number of boosting rounds.</param>
    /// <param name="learningRate">Shrinkage factor applied to each tree.</param>
    /// <param name="maxDepth">Maximum depth of each tree.</param>
    /// <param name="subsample">Fraction of samples used per tree (1.0 = no subsampling).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="loss">Loss function to optimize.</param>
    /// <param name="huberAlpha">Quantile for Huber loss delta estimation (default 0.9).</param>
    /// <param name="minSamplesSplit">Minimum samples to split a node.</param>
    /// <param name="minSamplesLeaf">Minimum samples in a leaf.</param>
    public GradientBoostedTreeRegressor(
        int nEstimators = 100,
        double learningRate = 0.1,
        int maxDepth = 3,
        double subsample = 1.0,
        int seed = 42,
        BoostingLoss loss = BoostingLoss.SquaredError,
        double huberAlpha = 0.9,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1)
    {
        _nEstimators = nEstimators;
        _learningRate = learningRate;
        _maxDepth = maxDepth;
        _subsample = Math.Clamp(subsample, 0.1, 1.0);
        _seed = seed;
        _loss = loss;
        _huberAlpha = huberAlpha;
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
        _numFeatures = X.Shape[1];
        var xData = X.ToArray();
        var yData = y.ToArray();

        // Initial prediction: mean for squared error, median for absolute/huber
        _initialPrediction = _loss == BoostingLoss.SquaredError
            ? Mean(yData)
            : Median(yData);

        var rawPreds = new double[nSamples];
        Array.Fill(rawPreds, _initialPrediction);

        _trees = new List<DecisionTreeRegressor>(_nEstimators);
        var rng = new Random(_seed);
        int subN = (int)(nSamples * _subsample);

        for (int iter = 0; iter < _nEstimators; iter++)
        {
            // Compute negative gradient (pseudo-residuals)
            var residuals = new double[nSamples];
            double huberDelta = 0;

            if (_loss == BoostingLoss.Huber)
            {
                // Compute delta as quantile of absolute residuals
                var absRes = new double[nSamples];
                for (int i = 0; i < nSamples; i++)
                    absRes[i] = Math.Abs(yData[i] - rawPreds[i]);
                Array.Sort(absRes);
                int qIdx = (int)(nSamples * _huberAlpha);
                qIdx = Math.Clamp(qIdx, 0, nSamples - 1);
                huberDelta = absRes[qIdx];
                if (huberDelta < 1e-10) huberDelta = 1e-10;
            }

            for (int i = 0; i < nSamples; i++)
            {
                double diff = yData[i] - rawPreds[i];
                residuals[i] = _loss switch
                {
                    BoostingLoss.SquaredError => diff,
                    BoostingLoss.AbsoluteError => Math.Sign(diff),
                    BoostingLoss.Huber => Math.Abs(diff) <= huberDelta
                        ? diff
                        : huberDelta * Math.Sign(diff),
                    _ => diff
                };
            }

            // Subsample
            double[] subX, subR;
            int fitN;
            if (_subsample < 1.0)
            {
                var indices = SampleIndices(nSamples, subN, rng);
                fitN = subN;
                subX = new double[fitN * _numFeatures];
                subR = new double[fitN];
                for (int i = 0; i < fitN; i++)
                {
                    Array.Copy(xData, indices[i] * _numFeatures, subX, i * _numFeatures, _numFeatures);
                    subR[i] = residuals[indices[i]];
                }
            }
            else
            {
                fitN = nSamples;
                subX = xData;
                subR = residuals;
            }

            var tree = new DecisionTreeRegressor(
                criterion: RegressorCriterion.MSE,
                maxDepth: _maxDepth,
                minSamplesSplit: _minSamplesSplit,
                minSamplesLeaf: _minSamplesLeaf,
                seed: _seed + iter);

            var xTensor = new Tensor<double>(subX, fitN, _numFeatures);
            var yTensor = new Tensor<double>(subR, fitN);
            tree.Fit(xTensor, yTensor);

            // Update raw predictions
            for (int i = 0; i < nSamples; i++)
                rawPreds[i] += _learningRate * tree.PredictSingleValue(xData, i, _numFeatures);

            _trees.Add(tree);
        }

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
        Array.Fill(result, _initialPrediction);

        foreach (var tree in _trees!)
        {
            for (int i = 0; i < nSamples; i++)
                result[i] += _learningRate * tree.PredictSingleValue(xData, i, nFeatures);
        }

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

        return ssTot == 0 ? (ssRes == 0 ? 1.0 : 0.0) : 1.0 - ssRes / ssTot;
    }

    private static double Mean(double[] data)
    {
        double sum = 0;
        for (int i = 0; i < data.Length; i++) sum += data[i];
        return sum / data.Length;
    }

    private static double Median(double[] data)
    {
        var sorted = (double[])data.Clone();
        Array.Sort(sorted);
        int n = sorted.Length;
        return n % 2 == 0
            ? (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
            : sorted[n / 2];
    }

    private static int[] SampleIndices(int n, int count, Random rng)
    {
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = rng.Next(n);
        return indices;
    }
}
