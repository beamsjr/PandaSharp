using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Gradient boosted tree classifier using sequential additive training of shallow
/// <see cref="DecisionTreeRegressor"/> trees. Supports binary and multi-class
/// classification via one-vs-rest with logistic/softmax loss.
/// </summary>
public class GradientBoostedTreeClassifier : IClassifier
{
    private readonly int _nEstimators;
    private readonly double _learningRate;
    private readonly int _maxDepth;
    private readonly double _subsample;
    private readonly int _seed;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;

    private int _numClasses;
    private int _numFeatures;

    // For binary: single list of trees and initial log-odds
    private List<DecisionTreeRegressor>? _binaryTrees;
    private double _binaryInitial;

    // For multi-class: one list of trees per class
    private List<DecisionTreeRegressor>[]? _multiTrees;
    private double[]? _multiInitial;

    /// <inheritdoc />
    public string Name => "GradientBoostedTreeClassifier";

    /// <inheritdoc />
    public bool IsFitted => _binaryTrees is not null || _multiTrees is not null;

    /// <inheritdoc />
    public int NumClasses => _numClasses;

    /// <summary>
    /// Creates a new gradient boosted tree classifier.
    /// </summary>
    /// <param name="nEstimators">Number of boosting rounds.</param>
    /// <param name="learningRate">Shrinkage factor applied to each tree.</param>
    /// <param name="maxDepth">Maximum depth of each tree.</param>
    /// <param name="subsample">Fraction of samples used per tree (1.0 = no subsampling).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    /// <param name="minSamplesSplit">Minimum samples to split a node.</param>
    /// <param name="minSamplesLeaf">Minimum samples in a leaf.</param>
    public GradientBoostedTreeClassifier(
        int nEstimators = 100,
        double learningRate = 0.1,
        int maxDepth = 3,
        double subsample = 1.0,
        int seed = 42,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1)
    {
        _nEstimators = nEstimators;
        _learningRate = learningRate;
        _maxDepth = maxDepth;
        _subsample = Math.Clamp(subsample, 0.1, 1.0);
        _seed = seed;
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

        // Determine number of classes
        _numClasses = 0;
        for (int i = 0; i < nSamples; i++)
        {
            int c = (int)yData[i];
            if (c + 1 > _numClasses) _numClasses = c + 1;
        }

        for (int i = 0; i < nSamples; i++)
            if ((int)yData[i] < 0) throw new ArgumentException("Labels must be non-negative.", nameof(y));

        if (_numClasses <= 2)
            FitBinary(xData, yData, nSamples);
        else
            FitMultiClass(xData, yData, nSamples);

        return this;
    }

    private void FitBinary(double[] xData, double[] yData, int nSamples)
    {
        int nFeatures = _numFeatures;
        _binaryTrees = new List<DecisionTreeRegressor>(_nEstimators);

        // Initial prediction: log-odds
        double posCount = 0;
        for (int i = 0; i < nSamples; i++) posCount += yData[i];
        double p0 = posCount / nSamples;
        p0 = Math.Clamp(p0, 1e-7, 1 - 1e-7);
        _binaryInitial = Math.Log(p0 / (1.0 - p0));

        // Current raw predictions (log-odds)
        var rawPreds = new double[nSamples];
        Array.Fill(rawPreds, _binaryInitial);

        var rng = new Random(_seed);
        int subN = (int)(nSamples * _subsample);

        for (int iter = 0; iter < _nEstimators; iter++)
        {
            // Compute negative gradient (residuals) for logistic loss
            var residuals = new double[nSamples];
            for (int i = 0; i < nSamples; i++)
            {
                double prob = Sigmoid(rawPreds[i]);
                residuals[i] = yData[i] - prob;
            }

            // Subsample
            double[] subX, subR;
            int fitN;
            if (_subsample < 1.0)
            {
                var indices = SampleIndices(nSamples, subN, rng);
                fitN = subN;
                subX = new double[fitN * nFeatures];
                subR = new double[fitN];
                for (int i = 0; i < fitN; i++)
                {
                    Array.Copy(xData, indices[i] * nFeatures, subX, i * nFeatures, nFeatures);
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

            var xTensor = new Tensor<double>(subX, fitN, nFeatures);
            var yTensor = new Tensor<double>(subR, fitN);
            tree.Fit(xTensor, yTensor);

            // Update raw predictions
            for (int i = 0; i < nSamples; i++)
                rawPreds[i] += _learningRate * tree.PredictSingleValue(xData, i, nFeatures);

            _binaryTrees.Add(tree);
        }
    }

    private void FitMultiClass(double[] xData, double[] yData, int nSamples)
    {
        int nFeatures = _numFeatures;
        int nClasses = _numClasses;

        _multiTrees = new List<DecisionTreeRegressor>[nClasses];
        _multiInitial = new double[nClasses];

        for (int c = 0; c < nClasses; c++)
            _multiTrees[c] = new List<DecisionTreeRegressor>(_nEstimators);

        // Compute initial class probabilities (log of prior)
        var classCounts = new double[nClasses];
        for (int i = 0; i < nSamples; i++)
            classCounts[(int)yData[i]]++;

        for (int c = 0; c < nClasses; c++)
        {
            double p = Math.Clamp(classCounts[c] / nSamples, 1e-7, 1 - 1e-7);
            _multiInitial[c] = Math.Log(p);
        }

        // Current raw predictions (log probabilities, un-normalized)
        var rawPreds = new double[nSamples * nClasses];
        for (int i = 0; i < nSamples; i++)
            for (int c = 0; c < nClasses; c++)
                rawPreds[i * nClasses + c] = _multiInitial[c];

        var rng = new Random(_seed);
        int subN = (int)(nSamples * _subsample);

        for (int iter = 0; iter < _nEstimators; iter++)
        {
            // Compute softmax probabilities
            var probs = new double[nSamples * nClasses];
            for (int i = 0; i < nSamples; i++)
                Softmax(rawPreds, i * nClasses, probs, i * nClasses, nClasses);

            // Fit one tree per class
            for (int c = 0; c < nClasses; c++)
            {
                // Negative gradient for class c
                var residuals = new double[nSamples];
                for (int i = 0; i < nSamples; i++)
                {
                    double target = (int)yData[i] == c ? 1.0 : 0.0;
                    residuals[i] = target - probs[i * nClasses + c];
                }

                double[] subX, subR;
                int fitN;
                if (_subsample < 1.0)
                {
                    var indices = SampleIndices(nSamples, subN, rng);
                    fitN = subN;
                    subX = new double[fitN * nFeatures];
                    subR = new double[fitN];
                    for (int j = 0; j < fitN; j++)
                    {
                        Array.Copy(xData, indices[j] * nFeatures, subX, j * nFeatures, nFeatures);
                        subR[j] = residuals[indices[j]];
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
                    seed: _seed + iter * nClasses + c);

                var xTensor = new Tensor<double>(subX, fitN, nFeatures);
                var yTensor = new Tensor<double>(subR, fitN);
                tree.Fit(xTensor, yTensor);

                // Update raw predictions for class c
                for (int i = 0; i < nSamples; i++)
                    rawPreds[i * nClasses + c] += _learningRate * tree.PredictSingleValue(xData, i, nFeatures);

                _multiTrees[c].Add(tree);
            }
        }
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (!IsFitted) throw new InvalidOperationException("Model has not been fitted.");

        var proba = PredictProba(X);
        int nSamples = X.Shape[0];
        var probaArr = proba.ToArray();
        var result = new double[nSamples];

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

        if (_numClasses <= 2)
            return PredictProbaBinary(xData, nSamples, nFeatures);
        else
            return PredictProbaMulti(xData, nSamples, nFeatures);
    }

    private Tensor<double> PredictProbaBinary(double[] xData, int nSamples, int nFeatures)
    {
        var result = new double[nSamples * _numClasses];

        for (int i = 0; i < nSamples; i++)
        {
            double raw = _binaryInitial;
            foreach (var tree in _binaryTrees!)
                raw += _learningRate * tree.PredictSingleValue(xData, i, nFeatures);

            double prob1 = Sigmoid(raw);
            if (_numClasses == 2)
            {
                result[i * 2] = 1.0 - prob1;
                result[i * 2 + 1] = prob1;
            }
            else
            {
                // Single class (degenerate)
                result[i] = prob1;
            }
        }

        return new Tensor<double>(result, nSamples, _numClasses);
    }

    private Tensor<double> PredictProbaMulti(double[] xData, int nSamples, int nFeatures)
    {
        int nClasses = _numClasses;
        var result = new double[nSamples * nClasses];

        for (int i = 0; i < nSamples; i++)
        {
            for (int c = 0; c < nClasses; c++)
            {
                double raw = _multiInitial![c];
                foreach (var tree in _multiTrees![c])
                    raw += _learningRate * tree.PredictSingleValue(xData, i, nFeatures);
                result[i * nClasses + c] = raw;
            }

            Softmax(result, i * nClasses, result, i * nClasses, nClasses);
        }

        return new Tensor<double>(result, nSamples, nClasses);
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

    private static double Sigmoid(double x)
    {
        if (x >= 0)
        {
            double ez = Math.Exp(-x);
            return 1.0 / (1.0 + ez);
        }
        else
        {
            double ez = Math.Exp(x);
            return ez / (1.0 + ez);
        }
    }

    private static void Softmax(double[] src, int srcOffset, double[] dst, int dstOffset, int length)
    {
        double max = double.NegativeInfinity;
        for (int i = 0; i < length; i++)
        {
            if (src[srcOffset + i] > max) max = src[srcOffset + i];
        }

        double sum = 0;
        for (int i = 0; i < length; i++)
        {
            dst[dstOffset + i] = Math.Exp(src[srcOffset + i] - max);
            sum += dst[dstOffset + i];
        }

        for (int i = 0; i < length; i++)
            dst[dstOffset + i] /= sum;
    }

    private static int[] SampleIndices(int n, int count, Random rng)
    {
        var indices = new int[count];
        for (int i = 0; i < count; i++)
            indices[i] = rng.Next(n);
        return indices;
    }
}
