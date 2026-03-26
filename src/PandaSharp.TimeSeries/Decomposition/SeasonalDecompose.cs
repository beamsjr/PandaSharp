namespace PandaSharp.TimeSeries.Decomposition;

/// <summary>Type of seasonal decomposition.</summary>
public enum DecomposeType
{
    /// <summary>Additive: Y = Trend + Seasonal + Residual.</summary>
    Additive,
    /// <summary>Multiplicative: Y = Trend * Seasonal * Residual.</summary>
    Multiplicative
}

/// <summary>
/// Result of seasonal decomposition containing trend, seasonal, and residual components.
/// </summary>
/// <param name="Observed">Original observed values.</param>
/// <param name="Trend">Trend component (NaN where moving average cannot be computed).</param>
/// <param name="Seasonal">Seasonal component.</param>
/// <param name="Residual">Residual component (NaN where trend is NaN).</param>
public record DecomposeResult(
    double[] Observed,
    double[] Trend,
    double[] Seasonal,
    double[] Residual);

/// <summary>
/// Classical seasonal decomposition of a time series into trend, seasonal,
/// and residual components using a moving average for trend extraction.
/// </summary>
public static class SeasonalDecompose
{
    /// <summary>
    /// Decompose a time series into trend, seasonal, and residual components.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="period">The seasonal period (e.g., 12 for monthly data with yearly seasonality).</param>
    /// <param name="type">Additive or Multiplicative decomposition.</param>
    /// <returns>A <see cref="DecomposeResult"/> with the three components.</returns>
    public static DecomposeResult Decompose(double[] series, int period, DecomposeType type = DecomposeType.Additive)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (period < 2)
            throw new ArgumentOutOfRangeException(nameof(period), "Period must be >= 2.");
        int n = series.Length;
        if (n < period * 2)
            throw new ArgumentException($"Series length ({n}) must be at least 2x the period ({period}).", nameof(series));

        // Step 1: Extract trend via centered moving average
        var trend = CenteredMovingAverage(series, period);

        // Step 2: De-trend to get seasonal + residual
        var detrended = new double[n];
        for (int i = 0; i < n; i++)
        {
            if (double.IsNaN(trend[i]))
            {
                detrended[i] = double.NaN;
            }
            else if (type == DecomposeType.Additive)
            {
                detrended[i] = series[i] - trend[i];
            }
            else
            {
                detrended[i] = trend[i] != 0 ? series[i] / trend[i] : double.NaN;
            }
        }

        // Step 3: Average seasonal component for each position in the cycle
        var seasonalAvg = new double[period];
        var seasonalCount = new int[period];
        for (int i = 0; i < n; i++)
        {
            if (!double.IsNaN(detrended[i]))
            {
                seasonalAvg[i % period] += detrended[i];
                seasonalCount[i % period]++;
            }
        }
        for (int i = 0; i < period; i++)
        {
            seasonalAvg[i] = seasonalCount[i] > 0 ? seasonalAvg[i] / seasonalCount[i] : 0;
        }

        // Normalize seasonal component
        if (type == DecomposeType.Additive)
        {
            double mean = seasonalAvg.Average();
            for (int i = 0; i < period; i++)
                seasonalAvg[i] -= mean;
        }
        else
        {
            double mean = seasonalAvg.Average();
            if (Math.Abs(mean) < 1e-10)
                throw new ArgumentException("Multiplicative decomposition requires non-zero mean seasonal values. Consider using additive decomposition.");
            for (int i = 0; i < period; i++)
                seasonalAvg[i] /= mean;
        }

        // Step 4: Build full seasonal and residual arrays
        var seasonal = new double[n];
        var residual = new double[n];
        for (int i = 0; i < n; i++)
        {
            seasonal[i] = seasonalAvg[i % period];

            if (double.IsNaN(trend[i]))
            {
                residual[i] = double.NaN;
            }
            else if (type == DecomposeType.Additive)
            {
                residual[i] = series[i] - trend[i] - seasonal[i];
            }
            else
            {
                double denom = trend[i] * seasonal[i];
                residual[i] = denom != 0 ? series[i] / denom : double.NaN;
            }
        }

        return new DecomposeResult(series, trend, seasonal, residual);
    }

    /// <summary>
    /// Compute a centered moving average. For even periods, a 2xMA is used.
    /// Boundary values are set to NaN.
    /// </summary>
    private static double[] CenteredMovingAverage(double[] series, int period)
    {
        int n = series.Length;
        var result = new double[n];
        Array.Fill(result, double.NaN);

        if (period % 2 == 1)
        {
            // Odd period: simple centered MA
            int half = period / 2;
            for (int i = half; i < n - half; i++)
            {
                double sum = 0;
                for (int j = i - half; j <= i + half; j++)
                    sum += series[j];
                result[i] = sum / period;
            }
        }
        else
        {
            // Even period: 2xMA (average of two consecutive MAs)
            int half = period / 2;
            var ma = new double[n];
            Array.Fill(ma, double.NaN);

            for (int i = half - 1; i < n - half; i++)
            {
                double sum = 0;
                for (int j = i - half + 1; j <= i + half; j++)
                    sum += series[j];
                ma[i] = sum / period;
            }

            for (int i = half; i < n - half; i++)
            {
                if (!double.IsNaN(ma[i]) && !double.IsNaN(ma[i - 1]))
                    result[i] = (ma[i] + ma[i - 1]) / 2.0;
            }
        }

        return result;
    }
}
