using PandaSharp;

namespace PandaSharp.TimeSeries.Decomposition;

/// <summary>
/// Frequency-domain analysis via discrete Fourier transform (DFT).
/// Computes a periodogram to identify dominant frequencies and periods in a time series.
/// </summary>
public static class Periodogram
{
    /// <summary>
    /// Compute the periodogram (power spectral density) of a time series.
    /// </summary>
    /// <param name="values">Time series observations.</param>
    /// <returns>
    /// DataFrame with columns: Frequency, Period, Power.
    /// Only non-negative frequencies up to the Nyquist frequency are returned.
    /// </returns>
    public static DataFrame Compute(double[] values)
    {
        ArgumentNullException.ThrowIfNull(values);
        int n = values.Length;
        if (n < 2) throw new ArgumentException("Need at least 2 observations.", nameof(values));

        // De-mean
        double mean = 0;
        for (int i = 0; i < n; i++) mean += values[i];
        mean /= n;

        var centered = new double[n];
        for (int i = 0; i < n; i++)
            centered[i] = values[i] - mean;

        // Compute DFT (no external dependency — simple O(n^2) for correctness)
        int nFreqs = n / 2 + 1;
        var realPart = new double[nFreqs];
        var imagPart = new double[nFreqs];

        for (int k = 0; k < nFreqs; k++)
        {
            double re = 0, im = 0;
            double twoPiKOverN = 2.0 * Math.PI * k / n;
            for (int t = 0; t < n; t++)
            {
                double angle = twoPiKOverN * t;
                re += centered[t] * Math.Cos(angle);
                im -= centered[t] * Math.Sin(angle);
            }
            realPart[k] = re;
            imagPart[k] = im;
        }

        // Power = |F(k)|^2 / n
        var frequencies = new double[nFreqs];
        var periods = new double[nFreqs];
        var power = new double[nFreqs];

        for (int k = 0; k < nFreqs; k++)
        {
            frequencies[k] = (double)k / n;
            periods[k] = k > 0 ? (double)n / k : double.PositiveInfinity;
            power[k] = (realPart[k] * realPart[k] + imagPart[k] * imagPart[k]) / n;
        }

        return DataFrame.FromDictionary(new Dictionary<string, Array>
        {
            ["Frequency"] = frequencies,
            ["Period"] = periods,
            ["Power"] = power
        });
    }

    /// <summary>
    /// Detect the dominant frequencies (peaks) in the periodogram.
    /// </summary>
    /// <param name="values">Time series observations.</param>
    /// <param name="topK">Number of top frequencies to return.</param>
    /// <returns>
    /// DataFrame with columns: Frequency, Period, Power — sorted by power descending,
    /// limited to <paramref name="topK"/> entries. The DC component (frequency 0) is excluded.
    /// </returns>
    public static DataFrame DominantFrequencies(double[] values, int topK = 5)
    {
        ArgumentNullException.ThrowIfNull(values);
        int n = values.Length;
        if (n < 2) throw new ArgumentException("Need at least 2 observations.", nameof(values));
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK), "topK must be >= 1.");

        double mean = 0;
        for (int i = 0; i < n; i++) mean += values[i];
        mean /= n;

        var centered = new double[n];
        for (int i = 0; i < n; i++)
            centered[i] = values[i] - mean;

        int nFreqs = n / 2 + 1;

        // Compute power at each frequency (skip DC at k=0)
        var freqPower = new (double Frequency, double Period, double Power)[Math.Max(0, nFreqs - 1)];

        for (int k = 1; k < nFreqs; k++)
        {
            double re = 0, im = 0;
            double twoPiKOverN = 2.0 * Math.PI * k / n;
            for (int t = 0; t < n; t++)
            {
                double angle = twoPiKOverN * t;
                re += centered[t] * Math.Cos(angle);
                im -= centered[t] * Math.Sin(angle);
            }

            double pw = (re * re + im * im) / n;
            freqPower[k - 1] = ((double)k / n, (double)n / k, pw);
        }

        // Sort by power descending and take top K
        Array.Sort(freqPower, (a, b) => b.Power.CompareTo(a.Power));

        int count = Math.Min(topK, freqPower.Length);
        var outFreq = new double[count];
        var outPeriod = new double[count];
        var outPower = new double[count];

        for (int i = 0; i < count; i++)
        {
            outFreq[i] = freqPower[i].Frequency;
            outPeriod[i] = freqPower[i].Period;
            outPower[i] = freqPower[i].Power;
        }

        return DataFrame.FromDictionary(new Dictionary<string, Array>
        {
            ["Frequency"] = outFreq,
            ["Period"] = outPeriod,
            ["Power"] = outPower
        });
    }
}
