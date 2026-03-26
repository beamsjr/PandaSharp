namespace PandaSharp.TimeSeries.Decomposition;

/// <summary>
/// Result of a periodogram computation.
/// </summary>
/// <param name="Frequencies">Frequency values (cycles per sample).</param>
/// <param name="Power">Power spectral density at each frequency.</param>
public record PeriodogramResult(double[] Frequencies, double[] Power);

/// <summary>
/// Compute a periodogram from a time series using a radix-2 Cooley-Tukey FFT.
/// Identifies dominant frequencies and their spectral power.
/// </summary>
public static class FFTPeriodogram
{
    /// <summary>
    /// Compute the periodogram of a time series.
    /// The input is zero-padded to the next power of two if necessary.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <returns>A <see cref="PeriodogramResult"/> with frequencies and power.</returns>
    public static PeriodogramResult Compute(double[] series)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (series.Length == 0) throw new ArgumentException("Series must not be empty.", nameof(series));

        int n = series.Length;
        int fftSize = NextPowerOfTwo(n);

        // Prepare complex input (zero-padded)
        var real = new double[fftSize];
        var imag = new double[fftSize];
        Array.Copy(series, real, n);

        // Remove mean to focus on oscillatory components
        double mean = 0;
        for (int i = 0; i < n; i++) mean += series[i];
        mean /= n;
        for (int i = 0; i < n; i++) real[i] -= mean;

        // Compute FFT
        FFT(real, imag, false);

        // Compute power spectrum (one-sided)
        int halfN = fftSize / 2;
        var frequencies = new double[halfN];
        var power = new double[halfN];

        for (int k = 0; k < halfN; k++)
        {
            frequencies[k] = (double)k / fftSize;
            power[k] = (real[k] * real[k] + imag[k] * imag[k]) / (n * n);
            // Double the power for non-DC and non-Nyquist
            if (k > 0 && k < halfN - 1)
                power[k] *= 2;
        }

        return new PeriodogramResult(frequencies, power);
    }

    /// <summary>
    /// Find the top K dominant frequencies by power.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="topK">Number of dominant frequencies to return.</param>
    /// <returns>
    /// Array of tuples (Frequency, Power, Period) sorted by power descending.
    /// Period is 1/Frequency (in sample units). DC component (frequency=0) is excluded.
    /// </returns>
    public static (double Frequency, double Power, double Period)[] DominantFrequencies(double[] series, int topK = 5)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (topK < 1) throw new ArgumentOutOfRangeException(nameof(topK), "topK must be >= 1.");
        var result = Compute(series);
        topK = Math.Min(topK, result.Frequencies.Length - 1); // Exclude DC

        // Build (frequency, power) pairs excluding DC
        var pairs = new (double Freq, double Pow, int Idx)[result.Frequencies.Length - 1];
        for (int i = 1; i < result.Frequencies.Length; i++)
            pairs[i - 1] = (result.Frequencies[i], result.Power[i], i);

        // Partial sort: find top K by power
        Array.Sort(pairs, (a, b) => b.Pow.CompareTo(a.Pow));

        var top = new (double Frequency, double Power, double Period)[topK];
        for (int i = 0; i < topK; i++)
        {
            double freq = pairs[i].Freq;
            top[i] = (freq, pairs[i].Pow, freq > 0 ? 1.0 / freq : double.PositiveInfinity);
        }

        return top;
    }

    /// <summary>
    /// In-place radix-2 Cooley-Tukey FFT (or inverse FFT).
    /// </summary>
    /// <param name="real">Real parts (modified in place).</param>
    /// <param name="imag">Imaginary parts (modified in place).</param>
    /// <param name="inverse">If true, compute the inverse FFT.</param>
    internal static void FFT(double[] real, double[] imag, bool inverse)
    {
        int n = real.Length;
        if (n <= 1) return;

        // Bit-reversal permutation
        int bits = (int)Math.Log2(n);
        for (int i = 0; i < n; i++)
        {
            int j = BitReverse(i, bits);
            if (j > i)
            {
                (real[i], real[j]) = (real[j], real[i]);
                (imag[i], imag[j]) = (imag[j], imag[i]);
            }
        }

        // Butterfly operations
        double sign = inverse ? 1.0 : -1.0;
        for (int size = 2; size <= n; size *= 2)
        {
            int halfSize = size / 2;
            double angle = sign * 2.0 * Math.PI / size;
            double wReal = Math.Cos(angle);
            double wImag = Math.Sin(angle);

            for (int start = 0; start < n; start += size)
            {
                double curReal = 1.0, curImag = 0.0;
                for (int k = 0; k < halfSize; k++)
                {
                    int even = start + k;
                    int odd = start + k + halfSize;

                    double tReal = curReal * real[odd] - curImag * imag[odd];
                    double tImag = curReal * imag[odd] + curImag * real[odd];

                    real[odd] = real[even] - tReal;
                    imag[odd] = imag[even] - tImag;
                    real[even] += tReal;
                    imag[even] += tImag;

                    double newCurReal = curReal * wReal - curImag * wImag;
                    curImag = curReal * wImag + curImag * wReal;
                    curReal = newCurReal;
                }
            }
        }

        // Scale for inverse FFT
        if (inverse)
        {
            for (int i = 0; i < n; i++)
            {
                real[i] /= n;
                imag[i] /= n;
            }
        }
    }

    private static int BitReverse(int value, int bits)
    {
        int result = 0;
        for (int i = 0; i < bits; i++)
        {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        return result;
    }

    private static int NextPowerOfTwo(int n)
    {
        int p = 1;
        while (p < n) p <<= 1;
        return p;
    }
}
