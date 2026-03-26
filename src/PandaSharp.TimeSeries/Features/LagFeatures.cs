using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;

namespace PandaSharp.TimeSeries.Features;

/// <summary>
/// Transformer that generates lag columns from a source column.
/// For each specified lag, creates a new column <c>{source}_lag_{n}</c> containing
/// the value shifted by <c>n</c> positions. Leading positions are filled with NaN.
/// </summary>
public class LagFeatures : ITransformer
{
    private readonly string _sourceColumn;
    private readonly int[] _lags;

    /// <inheritdoc />
    public string Name => "LagFeatures";

    /// <summary>
    /// Create a lag feature transformer.
    /// </summary>
    /// <param name="sourceColumn">Name of the column to create lags from.</param>
    /// <param name="lags">Lag values to generate (e.g., 1, 2, 3 for lag_1, lag_2, lag_3).</param>
    public LagFeatures(string sourceColumn, params int[] lags)
    {
        if (string.IsNullOrEmpty(sourceColumn))
            throw new ArgumentException("Source column name must not be empty.", nameof(sourceColumn));
        if (lags.Length == 0)
            throw new ArgumentException("At least one lag must be specified.", nameof(lags));
        if (lags.Any(l => l < 1))
            throw new ArgumentOutOfRangeException(nameof(lags), "All lags must be >= 1.");

        _sourceColumn = sourceColumn;
        _lags = lags;
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        // Stateless transformer; Fit is a no-op but validates the column exists
        if (!df.ColumnNames.Contains(_sourceColumn))
            throw new KeyNotFoundException($"Column '{_sourceColumn}' not found.");
        return this;
    }

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        var col = df[_sourceColumn];
        int n = df.RowCount;
        var result = df;

        foreach (int lag in _lags)
        {
            var values = new double?[n];
            for (int i = 0; i < n; i++)
            {
                if (i < lag || col.IsNull(i - lag))
                    values[i] = null;
                else
                    values[i] = TypeHelpers.GetDouble(col, i - lag);
            }

            string colName = $"{_sourceColumn}_lag_{lag}";
            var lagCol = Column<double>.FromNullable(colName, values);
            result = result.Assign(colName, lagCol);
        }

        return result;
    }

    /// <inheritdoc />
    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
