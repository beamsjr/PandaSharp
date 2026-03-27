using FluentAssertions;
using Cortex.ML.Metrics;

namespace Cortex.ML.Tests;

public class RankingMetricsTests
{
    [Fact]
    public void NDCG_PerfectRanking()
    {
        double[] yTrue = [1, 1, 0, 0, 0];
        double[] yScores = [5, 4, 3, 2, 1]; // perfect ranking

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.NDCG.Should().Be(1.0); // perfect → NDCG = 1
    }

    [Fact]
    public void NDCG_WorstRanking()
    {
        double[] yTrue = [1, 1, 0, 0, 0];
        double[] yScores = [1, 2, 3, 4, 5]; // reversed — irrelevant items ranked first

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.NDCG.Should().BeLessThan(1.0);
    }

    [Fact]
    public void MRR_FirstItemRelevant()
    {
        double[] yTrue = [1, 0, 0];
        double[] yScores = [3, 2, 1];

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.MRR.Should().Be(1.0); // first item is relevant
    }

    [Fact]
    public void MRR_SecondItemRelevant()
    {
        double[] yTrue = [0, 1, 0];
        double[] yScores = [3, 2, 1];

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.MRR.Should().Be(0.5); // second item is first relevant → 1/2
    }

    [Fact]
    public void MAP_PerfectRanking()
    {
        double[] yTrue = [1, 1, 0, 0];
        double[] yScores = [4, 3, 2, 1];

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.MAP.Should().Be(1.0);
    }

    [Fact]
    public void NoRelevantItems_ZeroMetrics()
    {
        double[] yTrue = [0, 0, 0];
        double[] yScores = [3, 2, 1];

        var m = RankingMetricsCalculator.Ranking(yTrue, yScores);
        m.NDCG.Should().Be(0);
        m.MRR.Should().Be(0);
        m.MAP.Should().Be(0);
    }

    [Fact]
    public void RankingAtK()
    {
        double[] yTrue = [0, 0, 1, 0, 1];
        double[] yScores = [5, 4, 3, 2, 1];

        var atK3 = RankingMetricsCalculator.Ranking(yTrue, yScores, k: 3);
        var atAll = RankingMetricsCalculator.Ranking(yTrue, yScores);

        // At k=3, only 1 relevant item is in top 3
        atK3.NDCG.Should().BeLessThan(atAll.NDCG);
    }

    [Fact]
    public void ToString_FormatsNicely()
    {
        var m = RankingMetricsCalculator.Ranking([1, 0], [2, 1]);
        m.ToString().Should().Contain("NDCG");
        m.ToString().Should().Contain("MRR");
    }
}
