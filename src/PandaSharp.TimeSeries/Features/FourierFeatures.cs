using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;

namespace PandaSharp.TimeSeries.Features;

/// <summary>
/// Transformer that generates Fourier (sin/cos) features for capturing seasonal patterns.
/// For each specified period and harmonic order, creates sin and cos columns.
/// </summary>
public class FourierFeatures : ITransformer
{
    private readonly double[] _periods;
    private readonly int _harmonics;
    private readonly string _indexColumn;

    /// <inheritdoc />
    public string Name => "FourierFeatures";

    /// <summary>
    /// Create a Fourier feature transformer.
    /// </summary>
    /// <param name="periods">Seasonal periods to model (e.g., 7.0 for weekly, 365.25 for yearly).</param>
    /// <param name="harmonics">Number of harmonic orders per period (e.g., 3 generates sin/cos for k=1,2,3).</param>
    /// <param name="indexColumn">
    /// Optional name of an integer or DateTime column to use as the time index.
    /// If null, row position (0, 1, 2, ...) is used.
    /// </param>
    public FourierFeatures(double[] periods, int harmonics = 3, string? indexColumn = null)
    {
        if (periods.Length == 0)
            throw new ArgumentException("At least one period must be specified.", nameof(periods));
        if (periods.Any(p => p <= 0))
            throw new ArgumentOutOfRangeException(nameof(periods), "All periods must be > 0.");
        if (harmonics < 1)
            throw new ArgumentOutOfRangeException(nameof(harmonics), "Harmonics must be >= 1.");

        _periods = periods;
        _harmonics = harmonics;
        _indexColumn = indexColumn ?? "";
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        // Stateless transformer
        return this;
    }

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        ArgumentNullException.ThrowIfNull(df);
        int n = df.RowCount;
        var result = df;

        // Determine time index values
        var timeIndex = new double[n];
        if (!string.IsNullOrEmpty(_indexColumn) && df.ColumnNames.Contains(_indexColumn))
        {
            var col = df[_indexColumn];
            if (col.DataType == typeof(DateTime) && n > 0)
            {
                var dtCol = df.GetColumn<DateTime>(_indexColumn);
                var span = dtCol.Values;
                // Convert to fractional days from the first date
                DateTime origin = span[0];
                for (int i = 0; i < n; i++)
                    timeIndex[i] = (span[i] - origin).TotalDays;
            }
            else
            {
                for (int i = 0; i < n; i++)
                    timeIndex[i] = TypeHelpers.GetDouble(col, i);
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
                timeIndex[i] = i;
        }

        foreach (double period in _periods)
        {
            for (int k = 1; k <= _harmonics; k++)
            {
                var sinValues = new double[n];
                var cosValues = new double[n];
                double freq = 2.0 * Math.PI * k / period;

                for (int i = 0; i < n; i++)
                {
                    double angle = freq * timeIndex[i];
                    sinValues[i] = Math.Sin(angle);
                    cosValues[i] = Math.Cos(angle);
                }

                string sinName = $"fourier_p{period:F0}_k{k}_sin";
                string cosName = $"fourier_p{period:F0}_k{k}_cos";
                result = result.Assign(sinName, new Column<double>(sinName, sinValues));
                result = result.Assign(cosName, new Column<double>(cosName, cosValues));
            }
        }

        return result;
    }

    /// <inheritdoc />
    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
