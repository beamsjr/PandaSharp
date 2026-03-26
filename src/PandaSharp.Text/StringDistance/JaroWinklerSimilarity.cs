namespace PandaSharp.Text.StringDistance;

/// <summary>
/// Computes Jaro-Winkler similarity between strings.
/// Returns a value between 0.0 (no similarity) and 1.0 (identical).
/// </summary>
public static class JaroWinklerSimilarity
{
    /// <summary>Compute Jaro-Winkler similarity between two strings.</summary>
    /// <param name="a">First string.</param>
    /// <param name="b">Second string.</param>
    /// <param name="prefixScale">Winkler prefix scaling factor (default 0.1, must be &lt;= 0.25).</param>
    public static double Compute(string a, string b, double prefixScale = 0.1)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (prefixScale > 0.25)
            throw new ArgumentOutOfRangeException(nameof(prefixScale), "Prefix scale must be <= 0.25");

        if (a.Length == 0 && b.Length == 0) return 1.0;
        if (a.Length == 0 || b.Length == 0) return 0.0;
        if (ReferenceEquals(a, b) || a == b) return 1.0;

        double jaro = ComputeJaro(a, b);

        // Winkler modification: bonus for common prefix up to 4 chars
        int prefixLen = 0;
        int maxPrefix = Math.Min(4, Math.Min(a.Length, b.Length));
        for (int i = 0; i < maxPrefix; i++)
        {
            if (a[i] == b[i])
                prefixLen++;
            else
                break;
        }

        return jaro + prefixLen * prefixScale * (1.0 - jaro);
    }

    private static double ComputeJaro(string a, string b)
    {
        int aLen = a.Length;
        int bLen = b.Length;

        int matchWindow = Math.Max(aLen, bLen) / 2 - 1;
        if (matchWindow < 0) matchWindow = 0;

        // Use stackalloc for short strings
        if (aLen <= 256 && bLen <= 256)
        {
            Span<bool> aMatched = stackalloc bool[aLen];
            Span<bool> bMatched = stackalloc bool[bLen];
            aMatched.Clear();
            bMatched.Clear();
            return ComputeJaroCore(a, b, aMatched, bMatched, matchWindow);
        }
        else
        {
            var aMatched = new bool[aLen];
            var bMatched = new bool[bLen];
            return ComputeJaroCore(a, b, aMatched, bMatched, matchWindow);
        }
    }

    private static double ComputeJaroCore(
        string a, string b, Span<bool> aMatched, Span<bool> bMatched, int matchWindow)
    {
        int aLen = a.Length;
        int bLen = b.Length;
        int matches = 0;

        // Find matching characters
        for (int i = 0; i < aLen; i++)
        {
            int start = Math.Max(0, i - matchWindow);
            int end = Math.Min(bLen - 1, i + matchWindow);

            for (int j = start; j <= end; j++)
            {
                if (bMatched[j] || a[i] != b[j]) continue;
                aMatched[i] = true;
                bMatched[j] = true;
                matches++;
                break;
            }
        }

        if (matches == 0) return 0.0;

        // Count transpositions
        int transpositions = 0;
        int k = 0;
        for (int i = 0; i < aLen; i++)
        {
            if (!aMatched[i]) continue;
            while (!bMatched[k]) k++;
            if (a[i] != b[k]) transpositions++;
            k++;
        }

        return ((double)matches / aLen +
                (double)matches / bLen +
                (matches - transpositions / 2.0) / matches) / 3.0;
    }

    /// <summary>
    /// Compute Jaro-Winkler similarity from each string in the array to the target.
    /// Null strings are treated as empty strings. Parallelized for large batches.
    /// </summary>
    public static double[] BatchCompute(string?[] strings, string target, double prefixScale = 0.1)
    {
        ArgumentNullException.ThrowIfNull(strings);
        ArgumentNullException.ThrowIfNull(target);

        var results = new double[strings.Length];

        Parallel.For(0, strings.Length, i =>
        {
            results[i] = Compute(strings[i] ?? string.Empty, target, prefixScale);
        });

        return results;
    }
}
