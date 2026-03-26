namespace PandaSharp.TimeSeries.Models;

/// <summary>
/// Changepoint detection using the Pruned Exact Linear Time (PELT) algorithm.
/// Detects abrupt shifts in the statistical properties of a time series.
/// </summary>
public static class ChangepointDetection
{
    /// <summary>
    /// Detect changepoints in a time series using the PELT algorithm with a
    /// normal mean-shift cost function.
    /// </summary>
    /// <param name="series">The time series values.</param>
    /// <param name="penalty">
    /// Penalty for adding a changepoint. Higher values yield fewer changepoints.
    /// A common default is <c>2 * log(n)</c> (BIC-like).
    /// </param>
    /// <returns>Array of changepoint indices (positions where a change begins).</returns>
    public static int[] PELT(double[] series, double penalty = double.NaN)
    {
        ArgumentNullException.ThrowIfNull(series);
        if (series.Length == 0) return [];

        int n = series.Length;
        if (double.IsNaN(penalty))
            penalty = 2.0 * Math.Log(n);

        // Precompute cumulative sums for O(1) cost computation
        var cumSum = new double[n + 1];
        var cumSumSq = new double[n + 1];
        for (int i = 0; i < n; i++)
        {
            cumSum[i + 1] = cumSum[i] + series[i];
            cumSumSq[i + 1] = cumSumSq[i] + series[i] * series[i];
        }

        // Cost of segment [s, t): negative log-likelihood for normal mean-shift
        double SegmentCost(int s, int t)
        {
            int len = t - s;
            if (len <= 0) return 0;
            double sum = cumSum[t] - cumSum[s];
            double sumSq = cumSumSq[t] - cumSumSq[s];
            return sumSq - (sum * sum) / len;
        }

        // PELT dynamic programming
        var f = new double[n + 1];
        var cp = new int[n + 1]; // last changepoint for each position
        f[0] = -penalty;
        var candidates = new List<int> { 0 };

        for (int tStar = 1; tStar <= n; tStar++)
        {
            double bestCost = double.MaxValue;
            int bestTau = 0;

            foreach (int tau in candidates)
            {
                double cost = f[tau] + SegmentCost(tau, tStar) + penalty;
                if (cost < bestCost)
                {
                    bestCost = cost;
                    bestTau = tau;
                }
            }

            f[tStar] = bestCost;
            cp[tStar] = bestTau;

            // Prune: remove candidates that can never be optimal
            candidates.RemoveAll(tau => f[tau] + SegmentCost(tau, tStar) > bestCost);
            candidates.Add(tStar);
        }

        // Backtrack to find changepoints
        var changepoints = new List<int>();
        int idx = n;
        while (idx > 0)
        {
            int prev = cp[idx];
            if (prev > 0)
                changepoints.Add(prev);
            idx = prev;
        }

        changepoints.Reverse();
        return changepoints.ToArray();
    }
}
