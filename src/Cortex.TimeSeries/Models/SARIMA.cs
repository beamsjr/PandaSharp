using Cortex;
using Cortex.Column;

namespace Cortex.TimeSeries.Models;

/// <summary>
/// Seasonal ARIMA (SARIMA) forecaster with parameters (p,d,q)(P,D,Q,m).
/// Extends ARIMA by applying seasonal differencing and seasonal AR/MA components.
/// </summary>
public class SARIMA : IForecaster
{
    private readonly int _p, _d, _q;
    private readonly int _P, _D, _Q, _m;

    private double[] _history = [];
    private DateTime[] _dates = [];
    private TimeSpan _step;
    private double[] _diffSeries = [];
    private double[] _arCoeffs = [];
    private double[] _maCoeffs = [];
    private double[] _sarCoeffs = [];
    private double[] _smaCoeffs = [];
    private double[] _residuals = [];
    private double _intercept;
    private double _residualStd;
    private bool _fitted;

    /// <summary>
    /// Create a SARIMA(p,d,q)(P,D,Q,m) forecaster.
    /// </summary>
    /// <param name="p">Non-seasonal AR order.</param>
    /// <param name="d">Non-seasonal differencing order.</param>
    /// <param name="q">Non-seasonal MA order.</param>
    /// <param name="P">Seasonal AR order.</param>
    /// <param name="D">Seasonal differencing order.</param>
    /// <param name="Q">Seasonal MA order.</param>
    /// <param name="m">Seasonal period (e.g., 12 for monthly with yearly seasonality).</param>
    public SARIMA(int p, int d, int q, int P, int D, int Q, int m)
    {
        if (p < 0 || d < 0 || q < 0) throw new ArgumentOutOfRangeException("Non-seasonal orders must be >= 0.");
        if (P < 0 || D < 0 || Q < 0) throw new ArgumentOutOfRangeException("Seasonal orders must be >= 0.");
        if (m < 2) throw new ArgumentOutOfRangeException(nameof(m), "Seasonal period must be >= 2.");
        _p = p; _d = d; _q = q;
        _P = P; _D = D; _Q = Q; _m = m;
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

        // Apply non-seasonal differencing d times
        _diffSeries = (double[])_history.Clone();
        for (int i = 0; i < _d; i++)
            _diffSeries = ARIMA.Difference(_diffSeries);

        // Apply seasonal differencing D times
        for (int i = 0; i < _D; i++)
            _diffSeries = SeasonalDifference(_diffSeries, _m);

        int minLen = Math.Max(_p, _P * _m) + Math.Max(_q, _Q * _m) + 1;
        if (_diffSeries.Length <= minLen)
            throw new InvalidOperationException("Not enough data after differencing for the specified orders.");

        _intercept = _diffSeries.Average();
        var centered = new double[_diffSeries.Length];
        for (int i = 0; i < centered.Length; i++)
            centered[i] = _diffSeries[i] - _intercept;

        // Fit non-seasonal AR
        _arCoeffs = _p > 0 ? FitARYuleWalker(centered, _p) : [];

        // Fit seasonal AR (at lags m, 2m, ..., Pm)
        _sarCoeffs = _P > 0 ? FitSeasonalAR(centered, _P, _m) : [];

        // Compute residuals from AR components
        _residuals = ComputeFullResiduals(centered, _arCoeffs, new double[_q], _sarCoeffs, new double[_Q]);

        // Fit non-seasonal MA iteratively
        _maCoeffs = _q > 0 ? FitMAIterative(centered, _q, false) : [];

        // Fit seasonal MA iteratively
        _smaCoeffs = _Q > 0 ? FitMAIterative(centered, _Q, true) : [];

        // Final residuals
        _residuals = ComputeFullResiduals(centered, _arCoeffs, _maCoeffs, _sarCoeffs, _smaCoeffs);

        double sse = 0;
        for (int i = 0; i < _residuals.Length; i++) sse += _residuals[i] * _residuals[i];
        int denom = Math.Max(1, _residuals.Length - _p - _q - _P - _Q);
        _residualStd = Math.Sqrt(sse / denom);

        _fitted = true;
        return this;
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

        int n = _diffSeries.Length;
        var extended = new double[n + horizon];
        Array.Copy(_diffSeries, extended, n);

        var extResiduals = new double[n + horizon];
        Array.Copy(_residuals, 0, extResiduals, 0, Math.Min(_residuals.Length, n));

        for (int h = 0; h < horizon; h++)
        {
            int t = n + h;
            double val = _intercept;

            // Non-seasonal AR
            for (int i = 0; i < _p; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) val += _arCoeffs[i] * (extended[idx] - _intercept);
            }

            // Seasonal AR
            for (int i = 0; i < _P; i++)
            {
                int idx = t - (i + 1) * _m;
                if (idx >= 0) val += _sarCoeffs[i] * (extended[idx] - _intercept);
            }

            // Non-seasonal MA
            for (int i = 0; i < _q; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) val += _maCoeffs[i] * extResiduals[idx];
            }

            // Seasonal MA
            for (int i = 0; i < _Q; i++)
            {
                int idx = t - (i + 1) * _m;
                if (idx >= 0) val += _smaCoeffs[i] * extResiduals[idx];
            }

            extended[t] = val;
        }

        // Integrate back: seasonal then non-seasonal
        var forecasts = new double[horizon];
        Array.Copy(extended, n, forecasts, 0, horizon);

        // Undo seasonal differencing
        var fullDiff = new double[_diffSeries.Length + horizon];
        Array.Copy(_diffSeries, fullDiff, _diffSeries.Length);
        Array.Copy(forecasts, 0, fullDiff, _diffSeries.Length, horizon);

        for (int dd = 0; dd < _D; dd++)
            IntegrateSeasonalInPlace(fullDiff, _m, _diffSeries.Length);

        // Undo non-seasonal differencing
        // Rebuild from history: apply non-seasonal diff _d times to history to get the known prefix
        var nonSeasonalDiffed = (double[])_history.Clone();
        for (int dd = 0; dd < _d; dd++)
            nonSeasonalDiffed = ARIMA.Difference(nonSeasonalDiffed);
        // The fullDiff now has both known + forecast portions with seasonal integration done
        // We need to integrate the non-seasonal differences
        for (int dd = 0; dd < _d; dd++)
            IntegrateNonSeasonalInPlace(fullDiff, nonSeasonalDiffed, _diffSeries.Length);

        for (int h = 0; h < horizon; h++)
        {
            dates[h] = _dates[^1] + _step * (h + 1);
            values[h] = fullDiff[_diffSeries.Length + h];
        }

        return (dates, values);
    }

    private static double[] SeasonalDifference(double[] series, int m)
    {
        if (series.Length <= m) return [];
        var result = new double[series.Length - m];
        for (int i = 0; i < result.Length; i++)
            result[i] = series[i + m] - series[i];
        return result;
    }

    private static void IntegrateSeasonalInPlace(double[] series, int m, int forecastStart)
    {
        for (int i = forecastStart; i < series.Length; i++)
        {
            if (i - m >= 0)
                series[i] += series[i - m];
        }
    }

    private static void IntegrateNonSeasonalInPlace(double[] series, double[] knownPrefix, int forecastStart)
    {
        // Restore known portion from the original non-seasonal differenced series
        int copyLen = Math.Min(knownPrefix.Length, forecastStart);
        for (int i = 0; i < copyLen; i++)
            series[i] = knownPrefix[i];

        // For the forecast portion, cumulative sum continuing from last known value
        for (int i = forecastStart; i < series.Length; i++)
        {
            if (i > 0)
                series[i] += series[i - 1];
        }
    }

    private static double[] FitARYuleWalker(double[] centered, int p)
    {
        int n = centered.Length;
        var acf = new double[p + 1];
        for (int lag = 0; lag <= p; lag++)
        {
            double sum = 0;
            for (int t = lag; t < n; t++)
                sum += centered[t] * centered[t - lag];
            acf[lag] = sum / n;
        }

        if (acf[0] == 0) return new double[p];

        // Levinson-Durbin
        var phi = new double[p];
        var phiPrev = new double[p];
        phi[0] = acf[1] / acf[0];

        for (int k = 1; k < p; k++)
        {
            double num = acf[k + 1];
            for (int j = 0; j < k; j++)
                num -= phi[j] * acf[k - j];

            double den = acf[0];
            for (int j = 0; j < k; j++)
                den -= phi[j] * acf[j + 1];

            if (Math.Abs(den) < 1e-15) break;

            Array.Copy(phi, phiPrev, p);
            phi[k] = num / den;
            for (int j = 0; j < k; j++)
                phi[j] = phiPrev[j] - phi[k] * phiPrev[k - 1 - j];
        }

        return phi;
    }

    private double[] FitSeasonalAR(double[] centered, int P, int m)
    {
        int n = centered.Length;
        var coeffs = new double[P];

        // Simple OLS for seasonal AR lags
        for (int i = 0; i < P; i++)
        {
            int lag = (i + 1) * m;
            double num = 0, den = 0;
            for (int t = lag; t < n; t++)
            {
                num += centered[t] * centered[t - lag];
                den += centered[t - lag] * centered[t - lag];
            }
            coeffs[i] = Math.Abs(den) > 1e-15 ? num / den : 0;
        }

        return coeffs;
    }

    private double[] FitMAIterative(double[] centered, int order, bool seasonal)
    {
        var coeffs = new double[order];

        for (int iter = 0; iter < 5; iter++)
        {
            var residuals = ComputeFullResiduals(centered, _arCoeffs,
                seasonal ? _maCoeffs : coeffs, _sarCoeffs,
                seasonal ? coeffs : _smaCoeffs);

            for (int j = 0; j < order; j++)
            {
                int lag = seasonal ? (j + 1) * _m : j + 1;
                double num = 0, den = 0;
                for (int t = lag; t < centered.Length; t++)
                {
                    int ridx = t - lag;
                    if (ridx >= 0 && ridx < residuals.Length)
                    {
                        num += (centered[t] - PredictAR(centered, t)) * residuals[ridx];
                        den += residuals[ridx] * residuals[ridx];
                    }
                }
                if (Math.Abs(den) > 1e-15)
                    coeffs[j] = num / den;
            }
        }

        return coeffs;
    }

    private double PredictAR(double[] centered, int t)
    {
        double val = 0;
        for (int i = 0; i < _p; i++)
        {
            int idx = t - i - 1;
            if (idx >= 0) val += _arCoeffs[i] * centered[idx];
        }
        for (int i = 0; i < _P; i++)
        {
            int idx = t - (i + 1) * _m;
            if (idx >= 0) val += _sarCoeffs[i] * centered[idx];
        }
        return val;
    }

    private double[] ComputeFullResiduals(double[] centered, double[] ar, double[] ma, double[] sar, double[] sma)
    {
        int n = centered.Length;
        var residuals = new double[n];

        for (int t = 0; t < n; t++)
        {
            double pred = 0;

            for (int i = 0; i < ar.Length; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) pred += ar[i] * centered[idx];
            }

            for (int i = 0; i < sar.Length; i++)
            {
                int idx = t - (i + 1) * _m;
                if (idx >= 0) pred += sar[i] * centered[idx];
            }

            for (int i = 0; i < ma.Length; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) pred += ma[i] * residuals[idx];
            }

            for (int i = 0; i < sma.Length; i++)
            {
                int idx = t - (i + 1) * _m;
                if (idx >= 0) pred += sma[i] * residuals[idx];
            }

            residuals[t] = centered[t] - pred;
        }

        return residuals;
    }

    private void EnsureFitted()
    {
        if (!_fitted) throw new InvalidOperationException("Call Fit() before forecasting.");
    }
}
