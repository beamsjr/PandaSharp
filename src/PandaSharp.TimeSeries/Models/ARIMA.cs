using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.TimeSeries.Models;

/// <summary>
/// ARIMA(p,d,q) forecaster. Differences the series <c>d</c> times, fits AR(p) coefficients
/// via Yule-Walker equations, and estimates MA(q) coefficients iteratively.
/// </summary>
public class ARIMA : IForecaster
{
    private readonly int _p;
    private readonly int _d;
    private readonly int _q;

    private double[] _history = [];
    private DateTime[] _dates = [];
    private TimeSpan _step;
    private double[] _diffSeries = [];
    private double[] _arCoeffs = [];
    private double[] _maCoeffs = [];
    private double[] _residuals = [];
    private double _intercept;
    private double _residualStd;
    private bool _fitted;

    /// <summary>
    /// Create an ARIMA(p,d,q) forecaster.
    /// </summary>
    /// <param name="p">Autoregressive order.</param>
    /// <param name="d">Differencing order.</param>
    /// <param name="q">Moving average order.</param>
    public ARIMA(int p = 1, int d = 1, int q = 0)
    {
        if (p < 0) throw new ArgumentOutOfRangeException(nameof(p));
        if (d < 0) throw new ArgumentOutOfRangeException(nameof(d));
        if (q < 0) throw new ArgumentOutOfRangeException(nameof(q));
        _p = p;
        _d = d;
        _q = q;
    }

    /// <summary>The AR coefficients after fitting.</summary>
    public ReadOnlySpan<double> ARCoefficients => _arCoeffs;

    /// <summary>The MA coefficients after fitting.</summary>
    public ReadOnlySpan<double> MACoefficients => _maCoeffs;

    /// <summary>The model intercept (mean of differenced series).</summary>
    public double Intercept => _intercept;

    /// <summary>
    /// Compute the Akaike Information Criterion for model selection.
    /// </summary>
    public double AIC
    {
        get
        {
            if (!_fitted) throw new InvalidOperationException("Call Fit() first.");
            int k = _p + _q + 1; // number of estimated parameters
            int n = _residuals.Length;
            double sse = 0;
            for (int i = 0; i < n; i++) sse += _residuals[i] * _residuals[i];
            double sigma2 = sse / n;
            return n * Math.Log(sigma2) + 2 * k;
        }
    }

    /// <summary>
    /// Compute the Bayesian Information Criterion for model selection.
    /// </summary>
    public double BIC
    {
        get
        {
            if (!_fitted) throw new InvalidOperationException("Call Fit() first.");
            int k = _p + _q + 1;
            int n = _residuals.Length;
            double sse = 0;
            for (int i = 0; i < n; i++) sse += _residuals[i] * _residuals[i];
            double sigma2 = sse / n;
            return n * Math.Log(sigma2) + k * Math.Log(n);
        }
    }

    /// <inheritdoc />
    public IForecaster Fit(DataFrame df, string dateColumn, string valueColumn)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(dateColumn);
        ArgumentNullException.ThrowIfNull(valueColumn);

        _history = TypeHelpers.GetDoubleArray(df[valueColumn]);
        if (_history.Length == 0)
            throw new ArgumentException("Series must not be empty.", nameof(valueColumn));
        var dateCol = df.GetColumn<DateTime>(dateColumn);
        _dates = new DateTime[dateCol.Length];
        for (int i = 0; i < dateCol.Length; i++)
            _dates[i] = dateCol.Values[i];
        _step = _dates.Length >= 2 ? _dates[^1] - _dates[^2] : TimeSpan.FromDays(1);

        // Difference the series d times
        _diffSeries = (double[])_history.Clone();
        for (int i = 0; i < _d; i++)
            _diffSeries = Difference(_diffSeries);

        if (_diffSeries.Length <= Math.Max(_p, _q) + 1)
            throw new InvalidOperationException($"Not enough data after differencing for the specified order. Have {_diffSeries.Length} observations but need at least {Math.Max(_p, _q) + 2} for ARIMA({_p},{_d},{_q}).");

        // Compute mean and center
        _intercept = _diffSeries.Average();
        var centered = new double[_diffSeries.Length];
        for (int i = 0; i < centered.Length; i++)
            centered[i] = _diffSeries[i] - _intercept;

        // Fit AR coefficients via Yule-Walker
        _arCoeffs = _p > 0 ? FitAR(centered, _p) : [];

        // Compute residuals for AR model
        _residuals = ComputeResiduals(centered, _arCoeffs, new double[_q]);

        // Fit MA coefficients iteratively (conditional least squares)
        if (_q > 0)
        {
            _maCoeffs = FitMA(centered, _arCoeffs, _q);
            _residuals = ComputeResiduals(centered, _arCoeffs, _maCoeffs);
        }
        else
        {
            _maCoeffs = [];
        }

        // Residual standard deviation
        double sse = 0;
        for (int i = 0; i < _residuals.Length; i++) sse += _residuals[i] * _residuals[i];
        int denom = _residuals.Length - _p - _q;
        _residualStd = denom > 0 ? Math.Sqrt(sse / denom) : Math.Sqrt(sse / Math.Max(1, _residuals.Length));

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

        // Forecast on the differenced series, then integrate back
        int n = _diffSeries.Length;
        var extended = new double[n + horizon];
        Array.Copy(_diffSeries, extended, n);

        var extResiduals = new double[n + horizon];
        Array.Copy(_residuals, 0, extResiduals, 0, Math.Min(_residuals.Length, n));

        for (int h = 0; h < horizon; h++)
        {
            int t = n + h;
            double val = _intercept;

            for (int i = 0; i < _p; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) val += _arCoeffs[i] * (extended[idx] - _intercept);
            }

            for (int i = 0; i < _q; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) val += _maCoeffs[i] * extResiduals[idx];
            }

            extended[t] = val;
            // Future residuals are zero (best forecast)
        }

        // Integrate back d times
        var forecastDiff = new double[horizon];
        Array.Copy(extended, n, forecastDiff, 0, horizon);

        // Build the chain of differenced series from the original history
        // diffChain[0] = original history, diffChain[k] = k-times differenced
        var diffChain = new double[_d + 1][];
        diffChain[0] = (double[])_history.Clone();
        for (int dd = 0; dd < _d; dd++)
            diffChain[dd + 1] = Difference(diffChain[dd]);

        // Start with the fully-differenced series (known + forecast)
        var current = new double[_diffSeries.Length + horizon];
        Array.Copy(_diffSeries, current, _diffSeries.Length);
        Array.Copy(forecastDiff, 0, current, _diffSeries.Length, horizon);

        // Undo differencing one level at a time: from d-times to (d-1)-times, etc.
        for (int dd = _d - 1; dd >= 0; dd--)
        {
            // diffChain[dd] is the dd-times differenced original series
            // We need to extend it with the forecast portion via cumulative sum
            var prev = diffChain[dd];
            var next = new double[prev.Length + horizon];
            Array.Copy(prev, next, prev.Length);

            // The forecast portion: cumsum from last known value of this level
            double lastVal = prev[^1];
            for (int i = 0; i < horizon; i++)
            {
                // current[knownLen + i] is the (dd+1)-times differenced forecast value
                // Adding it to the running sum gives the dd-times differenced value
                lastVal += current[diffChain[dd + 1].Length + i];
                next[prev.Length + i] = lastVal;
            }

            current = next;
        }

        for (int h = 0; h < horizon; h++)
        {
            dates[h] = _dates[^1] + _step * (h + 1);
            values[h] = current[_history.Length + h];
        }

        return (dates, values);
    }

    /// <summary>Difference a series once.</summary>
    internal static double[] Difference(double[] series)
    {
        if (series.Length <= 1) return [];
        var result = new double[series.Length - 1];
        for (int i = 0; i < result.Length; i++)
            result[i] = series[i + 1] - series[i];
        return result;
    }


    /// <summary>Fit AR(p) coefficients using Yule-Walker equations solved via Levinson-Durbin.</summary>
    private static double[] FitAR(double[] centered, int p)
    {
        int n = centered.Length;
        // Compute autocorrelations
        var acf = new double[p + 1];
        for (int lag = 0; lag <= p; lag++)
        {
            double sum = 0;
            for (int t = lag; t < n; t++)
                sum += centered[t] * centered[t - lag];
            acf[lag] = sum / n;
        }

        if (acf[0] == 0) return new double[p];

        // Levinson-Durbin recursion
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

            if (Math.Abs(den) < 1e-12 * Math.Abs(acf[0])) break;

            Array.Copy(phi, phiPrev, p);
            phi[k] = num / den;

            for (int j = 0; j < k; j++)
                phi[j] = phiPrev[j] - phi[k] * phiPrev[k - 1 - j];
        }

        return phi;
    }

    /// <summary>Fit MA(q) coefficients via iterative conditional least squares (3 iterations).</summary>
    private static double[] FitMA(double[] centered, double[] arCoeffs, int q)
    {
        int n = centered.Length;
        int p = arCoeffs.Length;
        var maCoeffs = new double[q];

        // Iterative estimation: compute residuals, then regress on lagged residuals
        for (int iter = 0; iter < 5; iter++)
        {
            var residuals = ComputeResiduals(centered, arCoeffs, maCoeffs);

            // OLS: estimate each MA coeff from residual correlation
            for (int j = 0; j < q; j++)
            {
                double num = 0, den = 0;
                for (int t = Math.Max(p, j + 1); t < n; t++)
                {
                    double arPart = 0;
                    for (int i = 0; i < p; i++)
                        if (t - i - 1 >= 0) arPart += arCoeffs[i] * centered[t - i - 1];

                    double maPart = 0;
                    for (int i = 0; i < q; i++)
                    {
                        if (i == j) continue;
                        int idx = t - i - 1;
                        if (idx >= 0 && idx < residuals.Length)
                            maPart += maCoeffs[i] * residuals[idx];
                    }

                    double target = centered[t] - arPart - maPart;
                    int ridx = t - j - 1;
                    if (ridx >= 0 && ridx < residuals.Length)
                    {
                        num += target * residuals[ridx];
                        den += residuals[ridx] * residuals[ridx];
                    }
                }

                if (Math.Abs(den) > 1e-15)
                    maCoeffs[j] = num / den;
            }
        }

        return maCoeffs;
    }

    /// <summary>Compute residuals given AR and MA coefficients.</summary>
    internal static double[] ComputeResiduals(double[] centered, double[] arCoeffs, double[] maCoeffs)
    {
        int n = centered.Length;
        int p = arCoeffs.Length;
        int q = maCoeffs.Length;
        var residuals = new double[n];

        for (int t = 0; t < n; t++)
        {
            double arPart = 0;
            for (int i = 0; i < p; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) arPart += arCoeffs[i] * centered[idx];
            }

            double maPart = 0;
            for (int i = 0; i < q; i++)
            {
                int idx = t - i - 1;
                if (idx >= 0) maPart += maCoeffs[i] * residuals[idx];
            }

            residuals[t] = centered[t] - arPart - maPart;
        }

        return residuals;
    }

    private void EnsureFitted()
    {
        if (!_fitted) throw new InvalidOperationException("Call Fit() before forecasting.");
    }
}
