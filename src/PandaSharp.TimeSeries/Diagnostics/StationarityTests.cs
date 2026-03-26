namespace PandaSharp.TimeSeries.Diagnostics;

/// <summary>
/// Result of a stationarity test (ADF or KPSS).
/// </summary>
/// <param name="TestStatistic">The computed test statistic.</param>
/// <param name="PValue">Approximate p-value.</param>
/// <param name="UsedLags">Number of lags used in the test.</param>
/// <param name="CriticalValues">Critical values at standard significance levels (1%, 5%, 10%).</param>
public record StationarityTestResult(
    double TestStatistic,
    double PValue,
    int UsedLags,
    Dictionary<string, double> CriticalValues);

/// <summary>
/// Stationarity tests for time series analysis: Augmented Dickey-Fuller and KPSS.
/// </summary>
public static class StationarityTests
{
    /// <summary>
    /// Augmented Dickey-Fuller test for unit root (non-stationarity).
    /// H0: series has a unit root (non-stationary).
    /// Reject H0 if test statistic &lt; critical value.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="maxLags">Maximum number of augmenting lags. If 0, uses floor(cbrt(n)).</param>
    /// <returns>Test result with statistic, approximate p-value, lags used, and critical values.</returns>
    public static StationarityTestResult AugmentedDickeyFuller(double[] series, int maxLags = 0)
    {
        ArgumentNullException.ThrowIfNull(series);
        int n = series.Length;
        if (n < 4) throw new ArgumentException("Series must have at least 4 observations.", nameof(series));

        if (maxLags <= 0)
            maxLags = (int)Math.Floor(Math.Cbrt(n - 1));

        // Compute first differences
        var dy = new double[n - 1];
        for (int i = 0; i < n - 1; i++)
            dy[i] = series[i + 1] - series[i];

        // Build regression: dy[t] = alpha + beta * y[t-1] + sum(gamma_i * dy[t-i]) + error
        int usedLags = Math.Min(maxLags, dy.Length - 2);
        int nObs = dy.Length - usedLags;
        if (nObs < 3) throw new ArgumentException("Not enough observations after differencing and lagging.", nameof(series));

        int nParams = 2 + usedLags; // intercept + beta + lag coeffs
        var X = new double[nObs, nParams];
        var Y = new double[nObs];

        for (int t = 0; t < nObs; t++)
        {
            int idx = t + usedLags;
            Y[t] = dy[idx];
            X[t, 0] = 1.0; // intercept
            X[t, 1] = series[idx]; // y[t-1] (lagged level)
            for (int lag = 0; lag < usedLags; lag++)
                X[t, 2 + lag] = dy[idx - lag - 1]; // lagged differences
        }

        // OLS: beta = (X'X)^-1 X'Y
        var coeffs = OLS(X, Y, nObs, nParams);

        // Compute residuals and standard error of beta coefficient
        var residuals = new double[nObs];
        double sse = 0;
        for (int t = 0; t < nObs; t++)
        {
            double pred = 0;
            for (int j = 0; j < nParams; j++)
                pred += X[t, j] * coeffs[j];
            residuals[t] = Y[t] - pred;
            sse += residuals[t] * residuals[t];
        }

        double sigma2 = sse / (nObs - nParams);

        // Compute (X'X)^-1 for standard errors
        var xtxInv = ComputeXtXInverse(X, nObs, nParams);
        double seBeta = Math.Sqrt(sigma2 * xtxInv[1, 1]);

        double testStat = seBeta > 1e-15 ? coeffs[1] / seBeta : 0;

        // Approximate p-value using MacKinnon critical values (constant, no trend)
        var criticalValues = new Dictionary<string, double>
        {
            ["1%"] = -3.43,
            ["5%"] = -2.86,
            ["10%"] = -2.57
        };

        double pValue = ApproximateADFPValue(testStat, n);

        return new StationarityTestResult(testStat, pValue, usedLags, criticalValues);
    }

    /// <summary>
    /// Kwiatkowski-Phillips-Schmidt-Shin (KPSS) test for stationarity.
    /// H0: series is stationary. Reject H0 if test statistic &gt; critical value.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="regressionType">Type of regression: "c" for constant (level stationarity), "ct" for constant + trend.</param>
    /// <returns>Test result with statistic, approximate p-value, bandwidth used, and critical values.</returns>
    public static StationarityTestResult KPSS(double[] series, string regressionType = "c")
    {
        ArgumentNullException.ThrowIfNull(series);
        ArgumentNullException.ThrowIfNull(regressionType);
        if (regressionType is not ("c" or "ct"))
            throw new ArgumentException("regressionType must be \"c\" or \"ct\".", nameof(regressionType));
        int n = series.Length;
        if (n < 4) throw new ArgumentException("Series must have at least 4 observations.", nameof(series));

        // Detrend: regress on constant (and trend if "ct")
        var residuals = new double[n];
        if (regressionType == "ct")
        {
            // OLS: y = a + b*t + e
            double sumT = 0, sumT2 = 0, sumY = 0, sumTY = 0;
            for (int i = 0; i < n; i++)
            {
                double t = i + 1;
                sumT += t;
                sumT2 += t * t;
                sumY += series[i];
                sumTY += t * series[i];
            }
            double den = n * sumT2 - sumT * sumT;
            double b = den != 0 ? (n * sumTY - sumT * sumY) / den : 0;
            double a = (sumY - b * sumT) / n;
            for (int i = 0; i < n; i++)
                residuals[i] = series[i] - a - b * (i + 1);
        }
        else
        {
            double mean = series.Average();
            for (int i = 0; i < n; i++)
                residuals[i] = series[i] - mean;
        }

        // Cumulative sum of residuals
        var cumResiduals = new double[n];
        cumResiduals[0] = residuals[0];
        for (int i = 1; i < n; i++)
            cumResiduals[i] = cumResiduals[i - 1] + residuals[i];

        // Long-run variance estimate using Bartlett kernel
        int bandwidth = (int)Math.Floor(Math.Sqrt(n));
        double s2 = 0;
        for (int i = 0; i < n; i++)
            s2 += residuals[i] * residuals[i];
        s2 /= n;

        for (int lag = 1; lag <= bandwidth; lag++)
        {
            double weight = 1.0 - (double)lag / (bandwidth + 1);
            double cov = 0;
            for (int i = lag; i < n; i++)
                cov += residuals[i] * residuals[i - lag];
            cov /= n;
            s2 += 2 * weight * cov;
        }

        // KPSS statistic
        double eta = 0;
        for (int i = 0; i < n; i++)
            eta += cumResiduals[i] * cumResiduals[i];
        double stat = eta / (n * n * s2);

        // Critical values for constant case
        Dictionary<string, double> criticalValues;
        if (regressionType == "ct")
        {
            criticalValues = new Dictionary<string, double>
            {
                ["1%"] = 0.216,
                ["5%"] = 0.146,
                ["10%"] = 0.119
            };
        }
        else
        {
            criticalValues = new Dictionary<string, double>
            {
                ["1%"] = 0.739,
                ["5%"] = 0.463,
                ["10%"] = 0.347
            };
        }

        double pValue = ApproximateKPSSPValue(stat, regressionType);

        return new StationarityTestResult(stat, pValue, bandwidth, criticalValues);
    }

    /// <summary>OLS regression returning coefficient vector.</summary>
    private static double[] OLS(double[,] X, double[] Y, int nObs, int nParams)
    {
        // X'X
        var xtx = new double[nParams, nParams];
        for (int i = 0; i < nParams; i++)
            for (int j = 0; j < nParams; j++)
            {
                double sum = 0;
                for (int t = 0; t < nObs; t++)
                    sum += X[t, i] * X[t, j];
                xtx[i, j] = sum;
            }

        // X'Y
        var xty = new double[nParams];
        for (int i = 0; i < nParams; i++)
        {
            double sum = 0;
            for (int t = 0; t < nObs; t++)
                sum += X[t, i] * Y[t];
            xty[i] = sum;
        }

        // Solve via Gaussian elimination
        return SolveLinearSystem(xtx, xty, nParams);
    }

    private static double[,] ComputeXtXInverse(double[,] X, int nObs, int nParams)
    {
        var xtx = new double[nParams, nParams];
        for (int i = 0; i < nParams; i++)
            for (int j = 0; j < nParams; j++)
            {
                double sum = 0;
                for (int t = 0; t < nObs; t++)
                    sum += X[t, i] * X[t, j];
                xtx[i, j] = sum;
            }

        return InvertMatrix(xtx, nParams);
    }

    /// <summary>Solve Ax=b via Gaussian elimination with partial pivoting.</summary>
    internal static double[] SolveLinearSystem(double[,] A, double[] b, int n)
    {
        var aug = new double[n, n + 1];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n] = b[i];
        }

        for (int col = 0; col < n; col++)
        {
            // Partial pivoting
            int maxRow = col;
            double maxVal = Math.Abs(aug[col, col]);
            for (int row = col + 1; row < n; row++)
            {
                if (Math.Abs(aug[row, col]) > maxVal)
                {
                    maxVal = Math.Abs(aug[row, col]);
                    maxRow = row;
                }
            }
            if (maxRow != col)
            {
                for (int j = 0; j <= n; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);
            }

            double pivot = aug[col, col];
            if (Math.Abs(pivot) < 1e-15) continue;

            for (int row = col + 1; row < n; row++)
            {
                double factor = aug[row, col] / pivot;
                for (int j = col; j <= n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        // Back substitution
        var x = new double[n];
        for (int i = n - 1; i >= 0; i--)
        {
            x[i] = aug[i, n];
            for (int j = i + 1; j < n; j++)
                x[i] -= aug[i, j] * x[j];
            if (Math.Abs(aug[i, i]) > 1e-15)
                x[i] /= aug[i, i];
        }

        return x;
    }

    /// <summary>Invert a matrix via Gauss-Jordan elimination.</summary>
    private static double[,] InvertMatrix(double[,] A, int n)
    {
        var aug = new double[n, 2 * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                aug[i, j] = A[i, j];
            aug[i, n + i] = 1.0;
        }

        for (int col = 0; col < n; col++)
        {
            int maxRow = col;
            for (int row = col + 1; row < n; row++)
                if (Math.Abs(aug[row, col]) > Math.Abs(aug[maxRow, col]))
                    maxRow = row;

            if (maxRow != col)
                for (int j = 0; j < 2 * n; j++)
                    (aug[col, j], aug[maxRow, j]) = (aug[maxRow, j], aug[col, j]);

            double pivot = aug[col, col];
            if (Math.Abs(pivot) < 1e-15) continue;

            for (int j = 0; j < 2 * n; j++)
                aug[col, j] /= pivot;

            for (int row = 0; row < n; row++)
            {
                if (row == col) continue;
                double factor = aug[row, col];
                for (int j = 0; j < 2 * n; j++)
                    aug[row, j] -= factor * aug[col, j];
            }
        }

        var inv = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                inv[i, j] = aug[i, n + j];
        return inv;
    }

    /// <summary>Approximate ADF p-value via interpolation of MacKinnon tables.</summary>
    private static double ApproximateADFPValue(double stat, int n)
    {
        // Simplified interpolation based on MacKinnon (1994) response surface
        // For constant, no trend case
        if (stat < -3.96) return 0.01;
        if (stat < -3.41) return 0.01 + (stat + 3.96) / (-3.41 + 3.96) * 0.04;
        if (stat < -2.86) return 0.05 + (stat + 3.41) / (-2.86 + 3.41) * 0.05;
        if (stat < -2.57) return 0.10 + (stat + 2.86) / (-2.57 + 2.86) * 0.15;
        if (stat < -1.94) return 0.25 + (stat + 2.57) / (-1.94 + 2.57) * 0.25;
        if (stat < -1.62) return 0.50 + (stat + 1.94) / (-1.62 + 1.94) * 0.25;
        return Math.Min(1.0, 0.75 + (stat + 1.62) * 0.1);
    }

    /// <summary>Approximate KPSS p-value.</summary>
    private static double ApproximateKPSSPValue(double stat, string type)
    {
        double[] thresholds;
        double[] pValues;

        if (type == "ct")
        {
            thresholds = [0.119, 0.146, 0.216];
            pValues = [0.10, 0.05, 0.01];
        }
        else
        {
            thresholds = [0.347, 0.463, 0.739];
            pValues = [0.10, 0.05, 0.01];
        }

        if (stat < thresholds[0]) return 0.10; // p > 0.10
        if (stat >= thresholds[2]) return 0.01; // p < 0.01

        // Linear interpolation
        for (int i = 0; i < thresholds.Length - 1; i++)
        {
            if (stat >= thresholds[i] && stat < thresholds[i + 1])
            {
                double frac = (stat - thresholds[i]) / (thresholds[i + 1] - thresholds[i]);
                return pValues[i] - frac * (pValues[i] - pValues[i + 1]);
            }
        }

        return 0.01;
    }
}
