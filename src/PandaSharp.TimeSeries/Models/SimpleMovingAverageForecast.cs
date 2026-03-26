using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.TimeSeries.Models;

/// <summary>
/// Baseline forecaster that predicts future values as the rolling mean of the
/// last <c>window</c> observations. Confidence intervals are derived from the
/// standard deviation of the window.
/// </summary>
public class SimpleMovingAverageForecast : IForecaster
{
    private readonly int _window;
    private double[] _history = [];
    private DateTime[] _dates = [];
    private TimeSpan _step;

    /// <summary>
    /// Create a simple moving average forecaster.
    /// </summary>
    /// <param name="window">Number of trailing observations to average.</param>
    public SimpleMovingAverageForecast(int window = 5)
    {
        if (window < 1) throw new ArgumentOutOfRangeException(nameof(window), "Window must be >= 1.");
        _window = window;
    }

    /// <inheritdoc />
    public IForecaster Fit(DataFrame df, string dateColumn, string valueColumn)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(dateColumn);
        ArgumentNullException.ThrowIfNull(valueColumn);

        _history = TypeHelpers.GetDoubleArray(df[valueColumn]);
        var dateCol = df.GetColumn<DateTime>(dateColumn);
        _dates = new DateTime[dateCol.Length];
        for (int i = 0; i < dateCol.Length; i++)
            _dates[i] = dateCol.Values[i];

        _step = _dates.Length >= 2
            ? _dates[^1] - _dates[^2]
            : TimeSpan.FromDays(1);

        return this;
    }

    /// <inheritdoc />
    public ForecastResult Forecast(int horizon)
    {
        if (horizon < 1) throw new ArgumentOutOfRangeException(nameof(horizon), "Horizon must be >= 1.");
        EnsureFitted();
        int w = Math.Min(_window, _history.Length);
        double sum = 0;
        for (int i = _history.Length - w; i < _history.Length; i++)
            sum += _history[i];
        double mean = sum / w;

        var values = new double[horizon];
        var dates = new DateTime[horizon];
        Array.Fill(values, mean);
        for (int i = 0; i < horizon; i++)
            dates[i] = _dates[^1] + _step * (i + 1);

        return new ForecastResult(dates, values, [], []);
    }

    /// <inheritdoc />
    public ForecastResult ForecastWithInterval(int horizon, double alpha = 0.05)
    {
        if (horizon < 1) throw new ArgumentOutOfRangeException(nameof(horizon), "Horizon must be >= 1.");
        EnsureFitted();
        int w = Math.Min(_window, _history.Length);
        double sum = 0;
        for (int i = _history.Length - w; i < _history.Length; i++)
            sum += _history[i];
        double mean = sum / w;

        double variance = 0;
        for (int i = _history.Length - w; i < _history.Length; i++)
            variance += (_history[i] - mean) * (_history[i] - mean);
        double std = w > 1 ? Math.Sqrt(variance / (w - 1)) : 0;

        double z = NormalQuantile(1.0 - alpha / 2.0);

        var values = new double[horizon];
        var dates = new DateTime[horizon];
        var lower = new double[horizon];
        var upper = new double[horizon];
        for (int i = 0; i < horizon; i++)
        {
            values[i] = mean;
            dates[i] = _dates[^1] + _step * (i + 1);
            double margin = z * std * Math.Sqrt(1.0 + 1.0 / w);
            lower[i] = mean - margin;
            upper[i] = mean + margin;
        }

        return new ForecastResult(dates, values, lower, upper);
    }

    private void EnsureFitted()
    {
        if (_history.Length == 0)
            throw new InvalidOperationException("Call Fit() before forecasting.");
    }

    /// <summary>
    /// Rational approximation of the standard normal quantile (Abramowitz and Stegun 26.2.23).
    /// </summary>
    internal static double NormalQuantile(double p)
    {
        if (p <= 0) return double.NegativeInfinity;
        if (p >= 1) return double.PositiveInfinity;
        if (Math.Abs(p - 0.5) < 1e-15) return 0;

        bool negate = p < 0.5;
        if (negate) p = 1.0 - p;

        double t = Math.Sqrt(-2.0 * Math.Log(1.0 - p));
        const double c0 = 2.515517, c1 = 0.802853, c2 = 0.010328;
        const double d1 = 1.432788, d2 = 0.189269, d3 = 0.001308;
        double result = t - (c0 + t * (c1 + t * c2)) / (1.0 + t * (d1 + t * (d2 + t * d3)));

        return negate ? -result : result;
    }
}
