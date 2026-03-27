namespace Cortex.ML.Metrics;

public record RankingResult(double NDCG, double MRR, double MAP)
{
    public override string ToString() => $"NDCG: {NDCG:F4}  MRR: {MRR:F4}  MAP: {MAP:F4}";
}

public static class RankingMetricsCalculator
{
    /// <summary>
    /// Compute ranking metrics from relevance scores.
    /// yTrue: ground truth relevance (0 or 1), yScores: predicted scores (higher = more relevant).
    /// Items are ranked by descending score.
    /// </summary>
    public static RankingResult Ranking(double[] yTrue, double[] yScores, int? k = null)
    {
        if (yTrue.Length != yScores.Length)
            throw new ArgumentException("yTrue and yScores must have the same length.");

        int n = yTrue.Length;
        int cutoff = k ?? n;

        // Sort by descending score
        var indices = Enumerable.Range(0, n).OrderByDescending(i => yScores[i]).ToArray();

        double ndcg = ComputeNDCG(yTrue, indices, cutoff);
        double mrr = ComputeMRR(yTrue, indices);
        double map = ComputeMAP(yTrue, indices, cutoff);

        return new RankingResult(ndcg, mrr, map);
    }

    /// <summary>
    /// Normalized Discounted Cumulative Gain.
    /// </summary>
    private static double ComputeNDCG(double[] yTrue, int[] sortedIndices, int k)
    {
        double dcg = 0;
        for (int i = 0; i < Math.Min(k, sortedIndices.Length); i++)
            dcg += yTrue[sortedIndices[i]] / Math.Log2(i + 2); // i+2 because log2(1)=0

        // Ideal DCG: sort by true relevance
        var idealIndices = Enumerable.Range(0, yTrue.Length)
            .OrderByDescending(i => yTrue[i]).ToArray();
        double idcg = 0;
        for (int i = 0; i < Math.Min(k, idealIndices.Length); i++)
            idcg += yTrue[idealIndices[i]] / Math.Log2(i + 2);

        return idcg > 0 ? dcg / idcg : 0;
    }

    /// <summary>
    /// Mean Reciprocal Rank: 1/rank of first relevant item.
    /// </summary>
    private static double ComputeMRR(double[] yTrue, int[] sortedIndices)
    {
        for (int i = 0; i < sortedIndices.Length; i++)
        {
            if (yTrue[sortedIndices[i]] > 0)
                return 1.0 / (i + 1);
        }
        return 0;
    }

    /// <summary>
    /// Mean Average Precision.
    /// </summary>
    private static double ComputeMAP(double[] yTrue, int[] sortedIndices, int k)
    {
        double sumPrecision = 0;
        int relevantSoFar = 0;
        int totalRelevant = yTrue.Count(y => y > 0);
        if (totalRelevant == 0) return 0;

        for (int i = 0; i < Math.Min(k, sortedIndices.Length); i++)
        {
            if (yTrue[sortedIndices[i]] > 0)
            {
                relevantSoFar++;
                sumPrecision += (double)relevantSoFar / (i + 1);
            }
        }

        return sumPrecision / totalRelevant;
    }
}
