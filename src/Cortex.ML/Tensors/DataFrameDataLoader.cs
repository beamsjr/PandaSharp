using System.Numerics;
using Cortex;
using Cortex.Column;

namespace Cortex.ML.Tensors;

/// <summary>
/// Batched data loader for ML training. Yields (features, labels) tensor pairs.
/// Supports shuffling per epoch and configurable batch size.
/// </summary>
public class DataFrameDataLoader : IEnumerable<(Tensor<double> Features, Tensor<double> Labels)>
{
    private readonly DataFrame _df;
    private readonly string[] _featureColumns;
    private readonly string _labelColumn;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly int? _seed;

    /// <summary>
    /// Tracks the current epoch number. Incremented each time GetEnumerator() is called
    /// when shuffle is enabled. Used to vary the random seed per epoch (seed + _epochCount)
    /// so each epoch produces a different shuffle order while remaining deterministic.
    /// When shuffle is disabled, _epochCount is not incremented and has no effect.
    /// If shuffle is toggled externally (not supported — shuffle is readonly), the counter
    /// continues from its current value.
    /// </summary>
    private int _epochCount;

    public int BatchCount => (_df.RowCount + _batchSize - 1) / _batchSize;
    public int TotalRows => _df.RowCount;
    public int FeatureCount => _featureColumns.Length;

    public DataFrameDataLoader(DataFrame df, string[] features, string label,
        int batchSize = 32, bool shuffle = true, int? seed = null)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(label);

        // Validate that all feature and label columns exist and are numeric
        foreach (var col in features)
        {
            if (!df.ColumnNames.Contains(col))
                throw new ArgumentException($"Feature column '{col}' not found in DataFrame.");
            if (!IsNumericOrBool(df[col].DataType))
                throw new ArgumentException(
                    $"Feature column '{col}' has non-numeric type '{df[col].DataType.Name}'. " +
                    "DataLoader requires all feature columns to be numeric (int, long, double, float, bool, etc.).");
        }
        if (!df.ColumnNames.Contains(label))
            throw new ArgumentException($"Label column '{label}' not found in DataFrame.");
        if (!IsNumericOrBool(df[label].DataType))
            throw new ArgumentException(
                $"Label column '{label}' has non-numeric type '{df[label].DataType.Name}'. " +
                "DataLoader requires the label column to be numeric.");

        _df = df;
        _featureColumns = features;
        _labelColumn = label;
        _batchSize = batchSize;
        _shuffle = shuffle;
        _seed = seed;
    }

    public IEnumerator<(Tensor<double> Features, Tensor<double> Labels)> GetEnumerator()
    {
        int n = _df.RowCount;
        int nFeatures = _featureColumns.Length;

        // Build indices (shuffle if requested)
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;

        if (_shuffle)
        {
            var rng = _seed.HasValue ? new Random(_seed.Value + _epochCount++) : Random.Shared;
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        // Pre-extract column spans for performance
        var featureSpans = new double[nFeatures][];
        for (int f = 0; f < nFeatures; f++)
        {
            var col = _df[_featureColumns[f]];
            var vals = new double[n];
            for (int i = 0; i < n; i++)
                vals[i] = col.IsNull(i) ? 0 : TypeHelpers.GetDouble(col, i);
            featureSpans[f] = vals;
        }

        var labelCol = _df[_labelColumn];
        var labelVals = new double[n];
        for (int i = 0; i < n; i++)
            labelVals[i] = labelCol.IsNull(i) ? 0 : TypeHelpers.GetDouble(labelCol, i);

        // Yield batches
        for (int batchStart = 0; batchStart < n; batchStart += _batchSize)
        {
            int batchEnd = Math.Min(batchStart + _batchSize, n);
            int batchLen = batchEnd - batchStart;

            var featureData = new double[batchLen * nFeatures];
            var labelData = new double[batchLen];

            for (int b = 0; b < batchLen; b++)
            {
                int idx = indices[batchStart + b];
                for (int f = 0; f < nFeatures; f++)
                    featureData[b * nFeatures + f] = featureSpans[f][idx];
                labelData[b] = labelVals[idx];
            }

            yield return (
                new Tensor<double>(featureData, batchLen, nFeatures),
                new Tensor<double>(labelData, batchLen)
            );
        }
    }

    System.Collections.IEnumerator System.Collections.IEnumerable.GetEnumerator() => GetEnumerator();

    private static bool IsNumericOrBool(Type type) =>
        TypeHelpers.IsNumeric(type) || type == typeof(bool);
}

public static class DataLoaderExtensions
{
    /// <summary>Create a batched data loader from a DataFrame.</summary>
    public static DataFrameDataLoader ToDataLoader(this DataFrame df,
        string[] features, string label, int batchSize = 32,
        bool shuffle = true, int? seed = null)
        => new(df, features, label, batchSize, shuffle, seed);
}
