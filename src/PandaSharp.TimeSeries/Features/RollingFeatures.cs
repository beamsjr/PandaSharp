using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;

namespace PandaSharp.TimeSeries.Features;

/// <summary>
/// Transformer that computes rolling window statistics (mean, std, min, max)
/// from a source column, producing new columns for each window size.
/// </summary>
public class RollingFeatures : ITransformer
{
    private readonly string _sourceColumn;
    private readonly int[] _windows;

    /// <inheritdoc />
    public string Name => "RollingFeatures";

    /// <summary>
    /// Create a rolling feature transformer.
    /// </summary>
    /// <param name="sourceColumn">Name of the numeric column to compute rolling stats for.</param>
    /// <param name="windows">Window sizes (e.g., 3, 7, 14).</param>
    public RollingFeatures(string sourceColumn, params int[] windows)
    {
        if (string.IsNullOrEmpty(sourceColumn))
            throw new ArgumentException("Source column name must not be empty.", nameof(sourceColumn));
        if (windows.Length == 0)
            throw new ArgumentException("At least one window size must be specified.", nameof(windows));
        if (windows.Any(w => w < 2))
            throw new ArgumentOutOfRangeException(nameof(windows), "All windows must be >= 2.");

        _sourceColumn = sourceColumn;
        _windows = windows;
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        if (!df.ColumnNames.Contains(_sourceColumn))
            throw new KeyNotFoundException($"Column '{_sourceColumn}' not found.");
        return this;
    }

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        var values = TypeHelpers.GetDoubleArray(df[_sourceColumn]);
        int n = df.RowCount;
        var result = df;

        foreach (int w in _windows)
        {
            var rollingMean = new double?[n];
            var rollingStd = new double?[n];
            var rollingMin = new double?[n];
            var rollingMax = new double?[n];

            // Use a running sum for efficient mean computation
            double sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += values[i];
                if (i >= w) sum -= values[i - w];

                if (i < w - 1)
                {
                    rollingMean[i] = null;
                    rollingStd[i] = null;
                    rollingMin[i] = null;
                    rollingMax[i] = null;
                    continue;
                }

                double mean = sum / w;
                rollingMean[i] = mean;

                // Compute std, min, max over window
                double variance = 0;
                double min = double.MaxValue, max = double.MinValue;
                for (int j = i - w + 1; j <= i; j++)
                {
                    double diff = values[j] - mean;
                    variance += diff * diff;
                    if (values[j] < min) min = values[j];
                    if (values[j] > max) max = values[j];
                }
                rollingStd[i] = w > 1 ? Math.Sqrt(variance / (w - 1)) : 0;
                rollingMin[i] = min;
                rollingMax[i] = max;
            }

            string prefix = $"{_sourceColumn}_rolling_{w}";
            result = result.Assign($"{prefix}_mean", Column<double>.FromNullable($"{prefix}_mean", rollingMean));
            result = result.Assign($"{prefix}_std", Column<double>.FromNullable($"{prefix}_std", rollingStd));
            result = result.Assign($"{prefix}_min", Column<double>.FromNullable($"{prefix}_min", rollingMin));
            result = result.Assign($"{prefix}_max", Column<double>.FromNullable($"{prefix}_max", rollingMax));
        }

        return result;
    }

    /// <inheritdoc />
    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
