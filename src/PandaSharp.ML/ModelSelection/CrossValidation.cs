using PandaSharp.ML.Models;
using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.ModelSelection;

/// <summary>
/// Cross-validation utilities for evaluating model performance with K-Fold splitting.
/// </summary>
public static class CrossValidation
{
    /// <summary>
    /// Evaluate a model using K-Fold cross-validation.
    /// Each fold: fit on training indices, score on validation indices.
    /// </summary>
    /// <param name="model">The model to evaluate (a fresh copy is used per fold).</param>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">Target vector of shape (n_samples).</param>
    /// <param name="nFolds">Number of folds (default 5).</param>
    /// <param name="seed">Optional random seed for reproducible shuffling.</param>
    /// <returns>Array of scores, one per fold.</returns>
    public static double[] CrossValScore(IModel model, Tensor<double> X, Tensor<double> y, int nFolds = 5, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(model);
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (nFolds < 2)
            throw new ArgumentOutOfRangeException(nameof(nFolds), "Must have at least 2 folds.");

        int nSamples = X.Shape[0];
        if (nSamples < nFolds)
            throw new ArgumentException($"Cannot have {nFolds} folds with only {nSamples} samples.");

        var indices = Enumerable.Range(0, nSamples).ToArray();

        // Shuffle
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        for (int i = nSamples - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        var scores = new double[nFolds];
        int foldSize = nSamples / nFolds;

        for (int fold = 0; fold < nFolds; fold++)
        {
            int valStart = fold * foldSize;
            int valEnd = fold == nFolds - 1 ? nSamples : valStart + foldSize;

            var valIndices = indices[valStart..valEnd];
            var trainIndices = indices[..valStart].Concat(indices[valEnd..]).ToArray();

            var xTrain = SliceRows(X, trainIndices);
            var yTrain = SliceElements(y, trainIndices);
            var xVal = SliceRows(X, valIndices);
            var yVal = SliceElements(y, valIndices);

            // Clone model by fitting a fresh instance — IModel.Fit returns a new/reset model
            var fittedModel = model.Fit(xTrain, yTrain);
            scores[fold] = fittedModel.Score(xVal, yVal);
        }

        return scores;
    }

    /// <summary>Select rows from a 2D tensor by index array using block copies.</summary>
    internal static Tensor<double> SliceRows(Tensor<double> X, int[] rowIndices)
    {
        int cols = X.Rank == 2 ? X.Shape[1] : 1;
        var data = new double[rowIndices.Length * cols];
        var src = X.Data.AsSpan();
        var dst = data.AsSpan();

        for (int i = 0; i < rowIndices.Length; i++)
        {
            src.Slice(rowIndices[i] * cols, cols).CopyTo(dst.Slice(i * cols, cols));
        }

        return X.Rank == 2
            ? new Tensor<double>(data, rowIndices.Length, cols)
            : new Tensor<double>(data, rowIndices.Length);
    }

    /// <summary>Select elements from a 1D tensor by index array.</summary>
    internal static Tensor<double> SliceElements(Tensor<double> y, int[] indices)
    {
        var data = new double[indices.Length];
        var span = y.Span;
        for (int i = 0; i < indices.Length; i++)
            data[i] = span[indices[i]];
        return new Tensor<double>(data, indices.Length);
    }
}
