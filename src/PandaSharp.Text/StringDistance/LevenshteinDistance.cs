namespace PandaSharp.Text.StringDistance;

/// <summary>
/// Computes the Levenshtein edit distance between strings using classic DP
/// with a single-row buffer (O(min(m,n)) space).
/// </summary>
public static class LevenshteinDistance
{
    /// <summary>Compute the Levenshtein distance between two strings.</summary>
    public static int Compute(string a, string b)
    {
        ArgumentNullException.ThrowIfNull(a);
        ArgumentNullException.ThrowIfNull(b);

        if (a.Length == 0) return b.Length;
        if (b.Length == 0) return a.Length;
        if (ReferenceEquals(a, b) || a == b) return 0;

        // Ensure b is the shorter string so we allocate less
        if (a.Length < b.Length)
            (a, b) = (b, a);

        int bLen = b.Length;

        // Use stackalloc for short strings to avoid heap allocation
        if (bLen < 1024)
        {
            Span<int> buffer = stackalloc int[bLen + 1];
            return ComputeCore(a, b, buffer);
        }
        else
        {
            var buffer = new int[bLen + 1];
            return ComputeCore(a, b, buffer);
        }
    }

    private static int ComputeCore(string a, string b, Span<int> prev)
    {
        int bLen = b.Length;

        for (int j = 0; j <= bLen; j++)
            prev[j] = j;

        for (int i = 1; i <= a.Length; i++)
        {
            int prevDiag = prev[0];
            prev[0] = i;
            char ai = a[i - 1];

            for (int j = 1; j <= bLen; j++)
            {
                int temp = prev[j];
                if (ai == b[j - 1])
                {
                    prev[j] = prevDiag;
                }
                else
                {
                    prev[j] = 1 + Math.Min(prevDiag, Math.Min(prev[j], prev[j - 1]));
                }
                prevDiag = temp;
            }
        }

        return prev[bLen];
    }

    /// <summary>
    /// Compute the Levenshtein distance from each string in the array to the target.
    /// Null strings are treated as empty strings. Parallelized for large batches.
    /// </summary>
    public static int[] BatchCompute(string?[] strings, string target)
    {
        ArgumentNullException.ThrowIfNull(strings);
        ArgumentNullException.ThrowIfNull(target);

        var results = new int[strings.Length];

        Parallel.For(0, strings.Length, i =>
        {
            results[i] = Compute(strings[i] ?? string.Empty, target);
        });

        return results;
    }
}
