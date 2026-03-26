using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.Statistics;

public class CorrelationAdvancedTests
{
    private const double Tolerance = 1e-10;

    // --- Spearman Tests ---

    [Fact]
    public void CorrSpearman_PerfectMonotonic_ReturnsOne()
    {
        // X = [1, 2, 3, 4, 5], Y = [2, 4, 6, 8, 10] -> perfectly monotonic -> rho = 1.0
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Y", [2, 4, 6, 8, 10])
        );

        var corr = df.CorrSpearman();
        corr.GetColumn<double>("X")[0].Should().BeApproximately(1.0, Tolerance); // X-X
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(1.0, Tolerance); // X-Y
        corr.GetColumn<double>("X")[1].Should().BeApproximately(1.0, Tolerance); // Y-X
        corr.GetColumn<double>("Y")[1].Should().BeApproximately(1.0, Tolerance); // Y-Y
    }

    [Fact]
    public void CorrSpearman_PerfectReversed_ReturnsNegativeOne()
    {
        // X ascending, Y descending -> rho = -1.0
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Y", [50, 40, 30, 20, 10])
        );

        var corr = df.CorrSpearman();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(-1.0, Tolerance); // X-Y
    }

    [Fact]
    public void CorrSpearman_ThreeColumns_KnownValues()
    {
        // A = [10, 20, 30, 40, 50]
        // B = [5, 6, 7, 8, 7]    -> mostly monotonic with A but has a tie
        // C = [50, 40, 30, 20, 10] -> perfectly reversed with A
        var df = new DataFrame(
            new Column<double>("A", [10, 20, 30, 40, 50]),
            new Column<double>("B", [5, 6, 7, 8, 7]),
            new Column<double>("C", [50, 40, 30, 20, 10])
        );

        var corr = df.CorrSpearman();

        // A-C should be -1.0 (perfect reverse monotonic)
        corr.GetColumn<double>("C")[0].Should().BeApproximately(-1.0, Tolerance);

        // A-A diagonal = 1.0
        corr.GetColumn<double>("A")[0].Should().BeApproximately(1.0, Tolerance);

        // A-B: ranks of A = [1,2,3,4,5], ranks of B = [1,2,3.5,5,3.5]
        // Spearman rho = Pearson on ranks
        // Verified with scipy.stats.spearmanr: 0.8207826...
        corr.GetColumn<double>("B")[0].Should().BeApproximately(0.8207826816681233, 1e-6);
    }

    [Fact]
    public void CorrSpearman_WithTies_UsesAverageRank()
    {
        // Values with ties: [10, 20, 20, 30]
        // Ranks should be [1, 2.5, 2.5, 4]
        var df = new DataFrame(
            new Column<double>("X", [10, 20, 20, 30]),
            new Column<double>("Y", [1, 2, 3, 4])
        );

        var corr = df.CorrSpearman();
        // Ranks of X: [1, 2.5, 2.5, 4], Ranks of Y: [1, 2, 3, 4]
        // These are strongly correlated but not perfectly
        corr.GetColumn<double>("Y")[0].Should().BeGreaterThan(0.9);
        corr.GetColumn<double>("Y")[0].Should().BeLessThan(1.0);
    }

    [Fact]
    public void CorrSpearman_WithIntColumns_Works()
    {
        var df = new DataFrame(
            new Column<int>("X", [1, 2, 3, 4, 5]),
            new Column<int>("Y", [5, 4, 3, 2, 1])
        );

        var corr = df.CorrSpearman();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(-1.0, Tolerance);
    }

    [Fact]
    public void CorrSpearman_ConstantColumn_ReturnsNaN()
    {
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Const", [7, 7, 7, 7, 7])
        );

        var corr = df.CorrSpearman();
        double.IsNaN(corr.GetColumn<double>("Const")[0]!.Value).Should().BeTrue(); // X vs Const = NaN
    }

    [Fact]
    public void CorrSpearman_SingleRow_ReturnsNaN()
    {
        var df = new DataFrame(
            new Column<double>("X", [42.0]),
            new Column<double>("Y", [99.0])
        );

        var corr = df.CorrSpearman();
        double.IsNaN(corr.GetColumn<double>("Y")[0]!.Value).Should().BeTrue();
    }

    // --- Kendall Tests ---

    [Fact]
    public void CorrKendall_PerfectConcordance_ReturnsOne()
    {
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Y", [10, 20, 30, 40, 50])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(1.0, Tolerance);
        corr.GetColumn<double>("X")[0].Should().BeApproximately(1.0, Tolerance); // diagonal
    }

    [Fact]
    public void CorrKendall_PerfectDiscordance_ReturnsNegativeOne()
    {
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Y", [50, 40, 30, 20, 10])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(-1.0, Tolerance);
    }

    [Fact]
    public void CorrKendall_KnownTauB_WithTies()
    {
        // X = [1, 2, 3, 3, 5], Y = [1, 3, 2, 4, 5]
        // 10 pairs total. C=8, D=1, Ties_X=1 (pair at indices 2,3), Ties_Y=0
        // tau_b = (8-1) / sqrt((10-1)*(10-0)) = 7/sqrt(90) = 0.7378647873726218
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 3, 5]),
            new Column<double>("Y", [1, 3, 2, 4, 5])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(7.0 / Math.Sqrt(90.0), 1e-10);
    }

    [Fact]
    public void CorrKendall_ThreeColumns()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 2, 3, 4, 5]),
            new Column<double>("B", [5, 4, 3, 2, 1]),
            new Column<double>("C", [1, 3, 2, 5, 4])
        );

        var corr = df.CorrKendall();

        // A-B: perfect discordance
        corr.GetColumn<double>("B")[0].Should().BeApproximately(-1.0, Tolerance);

        // A-C: [1,2,3,4,5] vs [1,3,2,5,4]
        // Concordant: (1,3),(1,2),(1,5),(1,4),(2,5),(2,4),(3,5),(3,4) = 8... let me count:
        // Pairs: (1,3)C, (1,2)C, (1,5)C, (1,4)C = 4
        //        (3,2)D, (3,5)C, (3,4)C = 2C 1D
        //        (2,5)C, (2,4)C = 2C
        //        (5,4)D = 1D
        // C=8, D=2, tau = (8-2)/10 = 0.6
        corr.GetColumn<double>("C")[0].Should().BeApproximately(0.6, Tolerance);

        // Symmetric
        corr.GetColumn<double>("A")[2].Should().BeApproximately(corr.GetColumn<double>("C")[0], Tolerance);
    }

    [Fact]
    public void CorrKendall_ConstantColumn_ReturnsNaN()
    {
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5]),
            new Column<double>("Const", [7, 7, 7, 7, 7])
        );

        var corr = df.CorrKendall();
        // All pairs in Const are ties -> denominator = 0 -> NaN
        double.IsNaN(corr.GetColumn<double>("Const")[0]!.Value).Should().BeTrue();
    }

    [Fact]
    public void CorrKendall_SingleRow_ReturnsNaN()
    {
        var df = new DataFrame(
            new Column<double>("X", [42.0]),
            new Column<double>("Y", [99.0])
        );

        var corr = df.CorrKendall();
        double.IsNaN(corr.GetColumn<double>("Y")[0]!.Value).Should().BeTrue();
    }

    [Fact]
    public void CorrKendall_WithIntColumns_Works()
    {
        var df = new DataFrame(
            new Column<int>("X", [1, 2, 3, 4]),
            new Column<int>("Y", [4, 3, 2, 1])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("Y")[0].Should().BeApproximately(-1.0, Tolerance);
    }

    // --- Edge Cases Shared ---

    [Fact]
    public void CorrSpearman_DiagonalIsOne()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 5, 3, 2, 4]),
            new Column<double>("B", [9, 3, 7, 1, 5]),
            new Column<double>("C", [2, 4, 6, 8, 10])
        );

        var corr = df.CorrSpearman();
        corr.GetColumn<double>("A")[0].Should().BeApproximately(1.0, Tolerance);
        corr.GetColumn<double>("B")[1].Should().BeApproximately(1.0, Tolerance);
        corr.GetColumn<double>("C")[2].Should().BeApproximately(1.0, Tolerance);
    }

    [Fact]
    public void CorrKendall_DiagonalIsOne()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 5, 3, 2, 4]),
            new Column<double>("B", [9, 3, 7, 1, 5])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("A")[0].Should().BeApproximately(1.0, Tolerance);
        corr.GetColumn<double>("B")[1].Should().BeApproximately(1.0, Tolerance);
    }

    [Fact]
    public void CorrSpearman_SymmetricMatrix()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 5, 3, 2, 4]),
            new Column<double>("B", [9, 3, 7, 1, 5]),
            new Column<double>("C", [2, 4, 6, 8, 10])
        );

        var corr = df.CorrSpearman();
        // corr[A,B] == corr[B,A]
        corr.GetColumn<double>("B")[0].Should().BeApproximately(corr.GetColumn<double>("A")[1], Tolerance);
        corr.GetColumn<double>("C")[0].Should().BeApproximately(corr.GetColumn<double>("A")[2], Tolerance);
        corr.GetColumn<double>("C")[1].Should().BeApproximately(corr.GetColumn<double>("B")[2], Tolerance);
    }

    [Fact]
    public void CorrKendall_SymmetricMatrix()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 5, 3, 2, 4]),
            new Column<double>("B", [9, 3, 7, 1, 5])
        );

        var corr = df.CorrKendall();
        corr.GetColumn<double>("B")[0].Should().BeApproximately(corr.GetColumn<double>("A")[1], Tolerance);
    }
}
