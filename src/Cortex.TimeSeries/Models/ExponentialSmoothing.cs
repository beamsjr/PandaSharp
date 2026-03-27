using Cortex;
using Cortex.Column;

namespace Cortex.TimeSeries.Models;

/// <summary>Type of exponential smoothing model.</summary>
public enum ESType
{
    /// <summary>Simple exponential smoothing (level only).</summary>
    Simple,
    /// <summary>Holt's linear method (level + trend).</summary>
    Double,
    /// <summary>Holt-Winters method (level + trend + seasonal).</summary>
    Triple
}

/// <summary>Seasonal pattern type for Triple exponential smoothing.</summary>
public enum Seasonal
{
    /// <summary>Additive seasonal component: Y = Level + Trend + Seasonal + Error.</summary>
    Additive,
    /// <summary>Multiplicative seasonal component: Y = (Level + Trend) * Seasonal * Error.</summary>
    Multiplicative
}

/// <summary>
/// Exponential smoothing forecaster supporting Simple, Double (Holt), and
/// Triple (Holt-Winters) variants with additive or multiplicative seasonality.
/// </summary>
public class ExponentialSmoothing : IForecaster
{
    private readonly ESType _type;
    private readonly Seasonal _seasonal;
    private readonly double _alpha;
    private readonly double _beta;
    private readonly double _gamma;
    private readonly int _seasonalPeriod;

    private double[] _history = [];
    private DateTime[] _dates = [];
    private TimeSpan _step;

    // Fitted state
    private double _level;
    private double _trend;
    private double[] _seasonalComponents = [];
    private double _residualStd;

    /// <summary>
    /// Create an exponential smoothing forecaster.
    /// </summary>
    /// <param name="type">Model type (Simple, Double, Triple).</param>
    /// <param name="alpha">Level smoothing parameter in (0,1).</param>
    /// <param name="beta">Trend smoothing parameter in (0,1). Used for Double/Triple.</param>
    /// <param name="gamma">Seasonal smoothing parameter in (0,1). Used for Triple.</param>
    /// <param name="seasonalPeriod">Length of a seasonal cycle. Used for Triple.</param>
    /// <param name="seasonal">Additive or Multiplicative seasonality.</param>
    public ExponentialSmoothing(
        ESType type = ESType.Simple,
        double alpha = 0.3,
        double beta = 0.1,
        double gamma = 0.1,
        int seasonalPeriod = 12,
        Seasonal seasonal = Seasonal.Additive)
    {
        if (alpha is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(alpha));
        if (type >= ESType.Double && beta is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(beta));
        if (type == ESType.Triple && gamma is <= 0 or >= 1) throw new ArgumentOutOfRangeException(nameof(gamma));
        if (type == ESType.Triple && seasonalPeriod < 2) throw new ArgumentOutOfRangeException(nameof(seasonalPeriod));

        _type = type;
        _alpha = alpha;
        _beta = beta;
        _gamma = gamma;
        _seasonalPeriod = seasonalPeriod;
        _seasonal = seasonal;
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
        _step = _dates.Length >= 2 ? _dates[^1] - _dates[^2] : TimeSpan.FromDays(1);

        int n = _history.Length;
        if (n == 0) throw new InvalidOperationException("Series is empty.");

        switch (_type)
        {
            case ESType.Simple:
                FitSimple();
                break;
            case ESType.Double:
                FitDouble();
                break;
            case ESType.Triple:
                if (n < _seasonalPeriod * 2)
                    throw new InvalidOperationException($"Need at least {_seasonalPeriod * 2} observations for Triple ES.");
                FitTriple();
                break;
        }

        return this;
    }

    private void FitSimple()
    {
        _level = _history[0];
        double sse = 0;
        for (int t = 1; t < _history.Length; t++)
        {
            double forecast = _level;
            _level = _alpha * _history[t] + (1 - _alpha) * _level;
            double err = _history[t] - forecast;
            sse += err * err;
        }
        _residualStd = _history.Length > 2 ? Math.Sqrt(sse / (_history.Length - 1)) : 0;
    }

    private void FitDouble()
    {
        _level = _history[0];
        _trend = _history.Length >= 2 ? _history[1] - _history[0] : 0;
        double sse = 0;
        for (int t = 1; t < _history.Length; t++)
        {
            double forecast = _level + _trend;
            double prevLevel = _level;
            _level = _alpha * _history[t] + (1 - _alpha) * (_level + _trend);
            _trend = _beta * (_level - prevLevel) + (1 - _beta) * _trend;
            double err = _history[t] - forecast;
            sse += err * err;
        }
        _residualStd = _history.Length > 2 ? Math.Sqrt(sse / (_history.Length - 2)) : 0;
    }

    private void FitTriple()
    {
        int m = _seasonalPeriod;
        int n = _history.Length;

        // Initialize level and trend from first season
        _level = 0;
        for (int i = 0; i < m; i++)
            _level += _history[i];
        _level /= m;

        _trend = 0;
        for (int i = 0; i < m; i++)
            _trend += (_history[m + i] - _history[i]);
        _trend /= (m * m);

        // Initialize seasonal components
        _seasonalComponents = new double[m];
        if (_seasonal == Seasonal.Additive)
        {
            for (int i = 0; i < m; i++)
                _seasonalComponents[i] = _history[i] - _level;
        }
        else
        {
            for (int i = 0; i < m; i++)
            {
                // If level is zero or near-zero in multiplicative mode, default seasonal factors to 1.0
                if (Math.Abs(_level) < 1e-10)
                    _seasonalComponents[i] = 1.0;
                else
                    _seasonalComponents[i] = _history[i] / _level;
            }
        }

        // Run smoothing
        double sse = 0;
        for (int t = m; t < n; t++)
        {
            int si = t % m;
            double forecast;
            double prevLevel = _level;

            if (_seasonal == Seasonal.Additive)
            {
                forecast = _level + _trend + _seasonalComponents[si];
                _level = _alpha * (_history[t] - _seasonalComponents[si]) + (1 - _alpha) * (_level + _trend);
                _trend = _beta * (_level - prevLevel) + (1 - _beta) * _trend;
                _seasonalComponents[si] = _gamma * (_history[t] - _level) + (1 - _gamma) * _seasonalComponents[si];
            }
            else
            {
                double safeS = Math.Abs(_seasonalComponents[si]) < 1e-8 ? 1e-8 : _seasonalComponents[si];
                forecast = (_level + _trend) * safeS;
                _level = _alpha * (_history[t] / safeS) + (1 - _alpha) * (_level + _trend);
                _trend = _beta * (_level - prevLevel) + (1 - _beta) * _trend;
                double safeLevel = Math.Abs(_level) < 1e-8 ? 1e-8 : _level;
                _seasonalComponents[si] = _gamma * (_history[t] / safeLevel) + (1 - _gamma) * _seasonalComponents[si];
            }

            double err = _history[t] - forecast;
            sse += err * err;
        }

        int denom = n - m - 3;
        _residualStd = denom > 0 ? Math.Sqrt(sse / denom) : 0;
    }

    /// <inheritdoc />
    public ForecastResult Forecast(int horizon)
    {
        EnsureFitted();
        var (dates, values) = ComputeForecast(horizon);
        return new ForecastResult(dates, values, [], []);
    }

    /// <inheritdoc />
    public ForecastResult ForecastWithInterval(int horizon, double alpha = 0.05)
    {
        EnsureFitted();
        var (dates, values) = ComputeForecast(horizon);
        double z = SimpleMovingAverageForecast.NormalQuantile(1.0 - alpha / 2.0);

        var lower = new double[horizon];
        var upper = new double[horizon];
        for (int h = 0; h < horizon; h++)
        {
            double margin = z * _residualStd * Math.Sqrt(h + 1);
            lower[h] = values[h] - margin;
            upper[h] = values[h] + margin;
        }

        return new ForecastResult(dates, values, lower, upper);
    }

    private (DateTime[] Dates, double[] Values) ComputeForecast(int horizon)
    {
        var dates = new DateTime[horizon];
        var values = new double[horizon];
        int m = _seasonalPeriod;

        for (int h = 0; h < horizon; h++)
        {
            dates[h] = _dates[^1] + _step * (h + 1);

            switch (_type)
            {
                case ESType.Simple:
                    values[h] = _level;
                    break;
                case ESType.Double:
                    values[h] = _level + _trend * (h + 1);
                    break;
                case ESType.Triple:
                    int si = (_history.Length + h) % m;
                    if (_seasonal == Seasonal.Additive)
                        values[h] = _level + _trend * (h + 1) + _seasonalComponents[si];
                    else
                        values[h] = (_level + _trend * (h + 1)) * _seasonalComponents[si];
                    break;
            }
        }

        return (dates, values);
    }

    private void EnsureFitted()
    {
        if (_history.Length == 0)
            throw new InvalidOperationException("Call Fit() before forecasting.");
    }
}
