using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Models;
using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.ModelSelection;

/// <summary>
/// Computes learning curves to diagnose bias/variance tradeoff.
/// Trains the model on increasing subsets of data and evaluates on cross-validation folds.
/// </summary>
public static class LearningCurve
{
    /// <summary>
    /// Compute learning curves for a model across multiple training set sizes.
    /// </summary>
    /// <param name="model">The model to evaluate.</param>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">Target vector of shape (n_samples).</param>
    /// <param name="trainSizes">
    /// Array of training set sizes. Values in (0, 1] are treated as fractions of total training samples;
    /// values &gt; 1 are treated as absolute sample counts.
    /// </param>
    /// <param name="nFolds">Number of cross-validation folds (default 5).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <returns>
    /// A DataFrame with columns: train_size, train_score_mean, train_score_std, val_score_mean, val_score_std.
    /// </returns>
    public static DataFrame Compute(
        IModel model,
        Tensor<double> X,
        Tensor<double> y,
        double[] trainSizes,
        int nFolds = 5,
        int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        ArgumentNullException.ThrowIfNull(trainSizes);
        int nSamples = X.Shape[0];
        if (nFolds < 2)
            throw new ArgumentOutOfRangeException(nameof(nFolds), "Must have at least 2 folds.");
        if (nSamples < nFolds)
            throw new ArgumentException($"Cannot have {nFolds} folds with only {nSamples} samples.", nameof(nFolds));

        // Generate shuffled indices
        var indices = Enumerable.Range(0, nSamples).ToArray();
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        for (int i = nSamples - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Prepare fold splits
        int foldSize = nSamples / nFolds;
        var folds = new List<(int[] trainIdx, int[] valIdx)>();
        for (int fold = 0; fold < nFolds; fold++)
        {
            int valStart = fold * foldSize;
            int valEnd = fold == nFolds - 1 ? nSamples : valStart + foldSize;

            var valIdx = indices[valStart..valEnd];
            var trainIdx = indices[..valStart].Concat(indices[valEnd..]).ToArray();
            folds.Add((trainIdx, valIdx));
        }

        var sizeList = new List<int>();
        var trainMeans = new List<double>();
        var trainStds = new List<double>();
        var valMeans = new List<double>();
        var valStds = new List<double>();

        foreach (double size in trainSizes)
        {
            var trainScores = new double[nFolds];
            var valScores = new double[nFolds];

            for (int fold = 0; fold < nFolds; fold++)
            {
                var (trainIdx, valIdx) = folds[fold];

                // Determine actual number of training samples to use
                int nTrainAvailable = trainIdx.Length;
                int nTrain = size <= 1.0
                    ? Math.Max(1, (int)(size * nTrainAvailable))
                    : Math.Min((int)size, nTrainAvailable);

                var subTrainIdx = trainIdx[..nTrain];

                var xTrain = CrossValidation.SliceRows(X, subTrainIdx);
                var yTrain = CrossValidation.SliceElements(y, subTrainIdx);
                var xVal = CrossValidation.SliceRows(X, valIdx);
                var yVal = CrossValidation.SliceElements(y, valIdx);

                var fitted = model.Fit(xTrain, yTrain);
                trainScores[fold] = fitted.Score(xTrain, yTrain);
                valScores[fold] = fitted.Score(xVal, yVal);
            }

            int effectiveSize = size <= 1.0
                ? Math.Max(1, (int)(size * folds[0].trainIdx.Length))
                : Math.Min((int)size, folds[0].trainIdx.Length);

            sizeList.Add(effectiveSize);
            trainMeans.Add(trainScores.Average());
            trainStds.Add(StdDev(trainScores));
            valMeans.Add(valScores.Average());
            valStds.Add(StdDev(valScores));
        }

        return new DataFrame(
            new Column<int>("train_size", sizeList.ToArray()),
            new Column<double>("train_score_mean", trainMeans.ToArray()),
            new Column<double>("train_score_std", trainStds.ToArray()),
            new Column<double>("val_score_mean", valMeans.ToArray()),
            new Column<double>("val_score_std", valStds.ToArray()));
    }

    private static double StdDev(double[] values)
    {
        double mean = values.Average();
        double sumSq = 0;
        foreach (var v in values)
        {
            double diff = v - mean;
            sumSq += diff * diff;
        }
        return Math.Sqrt(sumSq / values.Length);
    }
}
