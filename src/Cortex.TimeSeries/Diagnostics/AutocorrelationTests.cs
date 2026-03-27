namespace Cortex.TimeSeries.Diagnostics;

/// <summary>
/// Result of the Ljung-Box test for autocorrelation.
/// </summary>
/// <param name="TestStatistic">The Ljung-Box Q statistic.</param>
/// <param name="PValue">P-value from chi-squared distribution.</param>
public record LjungBoxResult(double TestStatistic, double PValue);

/// <summary>
/// Autocorrelation function (ACF), partial autocorrelation function (PACF),
/// and the Ljung-Box portmanteau test for serial correlation.
/// </summary>
public static class AutocorrelationTests
{
    /// <summary>
    /// Compute the autocorrelation function for lags 0 through <paramref name="maxLags"/>.
    /// ACF[0] is always 1.0.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="maxLags">Maximum lag to compute.</param>
    /// <returns>Array of length <c>maxLags + 1</c> containing autocorrelation coefficients.</returns>
    public static double[] ACF(double[] series, int maxLags)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (maxLags < 0) throw new ArgumentOutOfRangeException(nameof(maxLags), "maxLags must be >= 0.");
        int n = series.Length;
        if (n < 2) throw new ArgumentException("Series must have at least 2 observations.", nameof(series));
        maxLags = Math.Min(maxLags, n - 1);

        double mean = 0;
        for (int i = 0; i < n; i++) mean += series[i];
        mean /= n;

        double variance = 0;
        for (int i = 0; i < n; i++)
            variance += (series[i] - mean) * (series[i] - mean);

        var acf = new double[maxLags + 1];
        acf[0] = 1.0;

        if (variance < 1e-15) return acf;

        for (int lag = 1; lag <= maxLags; lag++)
        {
            double cov = 0;
            for (int t = lag; t < n; t++)
                cov += (series[t] - mean) * (series[t - lag] - mean);
            acf[lag] = cov / variance;
        }

        return acf;
    }

    /// <summary>
    /// Compute the partial autocorrelation function using the Levinson-Durbin recursion.
    /// PACF[0] is always 1.0.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="maxLags">Maximum lag to compute.</param>
    /// <returns>Array of length <c>maxLags + 1</c> containing partial autocorrelation coefficients.</returns>
    public static double[] PACF(double[] series, int maxLags)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (maxLags < 0) throw new ArgumentOutOfRangeException(nameof(maxLags), "maxLags must be >= 0.");
        int n = series.Length;
        if (n < 2) throw new ArgumentException("Series must have at least 2 observations.", nameof(series));
        maxLags = Math.Min(maxLags, n - 1);

        var acf = ACF(series, maxLags);
        var pacf = new double[maxLags + 1];
        pacf[0] = 1.0;

        if (maxLags == 0) return pacf;

        // Levinson-Durbin recursion
        var phi = new double[maxLags + 1];
        var phiPrev = new double[maxLags + 1];

        phi[1] = acf[1];
        pacf[1] = acf[1];

        for (int k = 2; k <= maxLags; k++)
        {
            double num = acf[k];
            for (int j = 1; j < k; j++)
                num -= phi[j] * acf[k - j];

            double den = 1.0;
            for (int j = 1; j < k; j++)
                den -= phi[j] * acf[j];

            if (Math.Abs(den) < 1e-15)
            {
                pacf[k] = 0;
                continue;
            }

            Array.Copy(phi, phiPrev, maxLags + 1);
            phi[k] = num / den;
            pacf[k] = phi[k];

            for (int j = 1; j < k; j++)
                phi[j] = phiPrev[j] - phi[k] * phiPrev[k - j];
        }

        return pacf;
    }

    /// <summary>
    /// Ljung-Box test for autocorrelation.
    /// H0: the data are independently distributed (no serial correlation up to lag <paramref name="lags"/>).
    /// </summary>
    /// <param name="series">The time series values (or residuals).</param>
    /// <param name="lags">Number of lags to test.</param>
    /// <returns>Test statistic Q and approximate p-value from chi-squared distribution.</returns>
    public static LjungBoxResult LjungBox(double[] series, int lags)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (lags < 1) throw new ArgumentOutOfRangeException(nameof(lags), "lags must be >= 1.");
        int n = series.Length;
        if (n < 2) throw new ArgumentException("Series must have at least 2 observations.", nameof(series));
        lags = Math.Min(lags, n - 1);

        var acf = ACF(series, lags);

        double Q = 0;
        for (int k = 1; k <= lags; k++)
        {
            double rk = acf[k];
            Q += (rk * rk) / (n - k);
        }
        Q *= n * (n + 2);

        // Approximate p-value from chi-squared(lags) distribution
        double pValue = ChiSquaredSurvival(Q, lags);

        return new LjungBoxResult(Q, pValue);
    }

    /// <summary>
    /// Approximate chi-squared survival function P(X > x) using the regularized
    /// incomplete gamma function. Uses series expansion for moderate values.
    /// </summary>
    internal static double ChiSquaredSurvival(double x, int df)
    {
        if (x <= 0) return 1.0;
        if (df <= 0) return 0.0;

        double a = df / 2.0;
        double z = x / 2.0;

        // Regularized lower incomplete gamma via series expansion
        double gamma = RegularizedGammaP(a, z);
        return Math.Max(0, Math.Min(1, 1.0 - gamma));
    }

    /// <summary>Regularized lower incomplete gamma function P(a,x) via series expansion.</summary>
    private static double RegularizedGammaP(double a, double x)
    {
        if (x < 0) return 0;
        if (x == 0) return 0;

        // For x < a + 1, use series expansion
        if (x < a + 1)
        {
            double sum = 1.0 / a;
            double term = 1.0 / a;
            for (int n = 1; n < 200; n++)
            {
                term *= x / (a + n);
                sum += term;
                if (Math.Abs(term) < Math.Abs(sum) * 1e-15) break;
            }
            return sum * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
        }
        else
        {
            // For x >= a + 1, use continued fraction (Lentz's method)
            return 1.0 - RegularizedGammaQ(a, x);
        }
    }

    /// <summary>Regularized upper incomplete gamma Q(a,x) via continued fraction.</summary>
    private static double RegularizedGammaQ(double a, double x)
    {
        double c = 1e-30;
        double d = 1.0 / (x + 1 - a);
        double h = d;

        for (int n = 1; n < 200; n++)
        {
            double an = -n * (n - a);
            double bn = x + 2 * n + 1 - a;
            d = bn + an * d;
            if (Math.Abs(d) < 1e-30) d = 1e-30;
            c = bn + an / c;
            if (Math.Abs(c) < 1e-30) c = 1e-30;
            d = 1.0 / d;
            double delta = c * d;
            h *= delta;
            if (Math.Abs(delta - 1.0) < 1e-15) break;
        }

        return h * Math.Exp(-x + a * Math.Log(x) - LogGamma(a));
    }

    private static readonly double[] LanczosCoefficients =
    [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.1208650973866179e-2,
        -0.5395239384953e-5
    ];

    /// <summary>Log-gamma function (Stirling approximation + Lanczos for small values).</summary>
    internal static double LogGamma(double x)
    {
        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += LanczosCoefficients[j] / ++y;

        return -tmp + Math.Log(2.5066282746310005 * ser / x);
    }
}
