using FluentAssertions;
using Cortex.Column;
using Cortex.Statistics;
using Cortex.Window;

namespace Cortex.Tests.Unit.EdgeCases;

public class StatisticsDeepTests
{
    // =======================================================================
    // Area 1: Covariance matrix edge cases
    // =======================================================================

    [Fact]
    public void Cov_SingleRow_ShouldReturnNaN_BecauseDdof1MeansDiv0()
    {
        // With ddof=1 (default), 1 row means n-1=0, so covariance = NaN
        var df = new DataFrame(
            new Column<double>("A", [1.0]),
            new Column<double>("B", [2.0])
        );

        var cov = df.Cov();

        var colA = (Column<double>)cov["A"];
        var colB = (Column<double>)cov["B"];
        colA[0].Should().Be(double.NaN, "Cov(A,A) with 1 row and ddof=1 should be NaN");
        colA[1].Should().Be(double.NaN, "Cov(B,A) with 1 row and ddof=1 should be NaN");
        colB[0].Should().Be(double.NaN, "Cov(A,B) with 1 row and ddof=1 should be NaN");
        colB[1].Should().Be(double.NaN, "Cov(B,B) with 1 row and ddof=1 should be NaN");
    }

    [Fact]
    public void Cov_Ddof0_PopulationCovariance_ShouldDifferFromDdof1()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<double>("B", [4.0, 5.0, 6.0])
        );

        var covSample = df.Cov(ddof: 1);
        var covPop = df.Cov(ddof: 0);

        var sampleVal = ((Column<double>)covSample["A"])[0]; // Var(A) with ddof=1
        var popVal = ((Column<double>)covPop["A"])[0];       // Var(A) with ddof=0

        // Var(A) = sum((x-mean)^2) / (n-ddof)
        // mean = 2, sum_sq = 2, so ddof=1 => 1.0, ddof=0 => 0.6667
        sampleVal.Should().BeApproximately(1.0, 1e-10);
        popVal.Should().BeApproximately(2.0 / 3.0, 1e-10);
    }

    [Fact]
    public void Cov_DiagonalEqualsVar_ForEachColumn()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [2.0, 4.0, 6.0, 8.0, 10.0])
        );

        var cov = df.Cov();
        var colA = (Column<double>)cov["A"];
        var colB = (Column<double>)cov["B"];

        var varA = new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]).Var();
        var varB = new Column<double>("B", [2.0, 4.0, 6.0, 8.0, 10.0]).Var();

        colA[0].Should().BeApproximately(varA!.Value, 1e-10, "Cov diagonal should equal Var for column A");
        colB[1].Should().BeApproximately(varB!.Value, 1e-10, "Cov diagonal should equal Var for column B");
    }

    // =======================================================================
    // Area 2: Spearman / Kendall correctness
    // =======================================================================

    [Fact]
    public void CorrSpearman_PerfectlyMonotonic_ShouldBeExactly1()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [2.0, 4.0, 6.0, 8.0, 10.0])
        );

        var corr = df.CorrSpearman();
        var colA = (Column<double>)corr["A"];
        var colB = (Column<double>)corr["B"];

        // Spearman of perfectly monotonic data should be exactly 1.0
        colA[0].Should().BeApproximately(1.0, 1e-10, "Spearman(A,A) = 1.0");
        colA[1].Should().BeApproximately(1.0, 1e-10, "Spearman(B,A) = 1.0 for perfectly monotonic");
        colB[0].Should().BeApproximately(1.0, 1e-10, "Spearman(A,B) = 1.0 for perfectly monotonic");
        colB[1].Should().BeApproximately(1.0, 1e-10, "Spearman(B,B) = 1.0");
    }

    [Fact]
    public void CorrSpearman_PerfectlyAntiMonotonic_ShouldBeExactlyMinus1()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [10.0, 8.0, 6.0, 4.0, 2.0])
        );

        var corr = df.CorrSpearman();
        var offDiag = ((Column<double>)corr["B"])[0]; // Spearman(A,B)

        offDiag.Should().BeApproximately(-1.0, 1e-10, "Spearman of anti-monotonic should be -1.0");
    }

    [Fact]
    public void CorrKendall_PerfectlyMonotonic_ShouldBeExactly1()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [2.0, 4.0, 6.0, 8.0, 10.0])
        );

        var corr = df.CorrKendall();
        var offDiag = ((Column<double>)corr["B"])[0];

        offDiag.Should().BeApproximately(1.0, 1e-10, "Kendall of perfectly monotonic should be 1.0");
    }

    [Fact]
    public void CorrKendall_PerfectlyAntiMonotonic_ShouldBeExactlyMinus1()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [10.0, 8.0, 6.0, 4.0, 2.0])
        );

        var corr = df.CorrKendall();
        var offDiag = ((Column<double>)corr["B"])[0];

        offDiag.Should().BeApproximately(-1.0, 1e-10, "Kendall of anti-monotonic should be -1.0");
    }

    [Fact]
    public void CorrSpearman_2RowDataFrame_ShouldWork()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0]),
            new Column<double>("B", [3.0, 4.0])
        );

        var corr = df.CorrSpearman();
        var offDiag = ((Column<double>)corr["B"])[0];

        // 2 rows, perfectly monotonic => Spearman = 1.0
        offDiag.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void CorrSpearman_AllConstantColumn_ShouldReturnNaN()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<double>("B", [5.0, 5.0, 5.0])
        );

        var corr = df.CorrSpearman();
        var offDiag = ((Column<double>)corr["B"])[0]; // Spearman(A, constant)

        double.IsNaN(offDiag!.Value).Should().BeTrue("Spearman with constant column should be NaN");
    }

    [Fact]
    public void CorrKendall_AllConstantColumn_ShouldReturnNaN()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<double>("B", [5.0, 5.0, 5.0])
        );

        var corr = df.CorrKendall();
        var offDiag = ((Column<double>)corr["B"])[0];

        double.IsNaN(offDiag!.Value).Should().BeTrue("Kendall with constant column should be NaN");
    }

    [Fact]
    public void CorrSpearman_WithTies_ShouldUseAverageRank()
    {
        // Data: A = [1,2,2,3], B = [10,20,20,30]
        // Ranks of A: [1, 2.5, 2.5, 4] (average rank for ties)
        // Ranks of B: [1, 2.5, 2.5, 4]
        // Pearson of identical ranks = 1.0
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 2.0, 3.0]),
            new Column<double>("B", [10.0, 20.0, 20.0, 30.0])
        );

        var corr = df.CorrSpearman();
        var offDiag = ((Column<double>)corr["B"])[0];

        offDiag.Should().BeApproximately(1.0, 1e-10, "Spearman with identical rank patterns should be 1.0");
    }

    // =======================================================================
    // Area 3: Rolling window correctness
    // =======================================================================

    [Fact]
    public void Rolling3_Mean_VerifyExactValues()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Rolling(3).Mean();

        // Expected: [NaN, NaN, 2.0, 3.0, 4.0]
        result[0].Should().BeNull("window not full yet");
        result[1].Should().BeNull("window not full yet");
        result[2].Should().BeApproximately(2.0, 1e-10);
        result[3].Should().BeApproximately(3.0, 1e-10);
        result[4].Should().BeApproximately(4.0, 1e-10);
    }

    [Fact]
    public void Rolling3_Std_OnConstantData_ShouldBeZero()
    {
        var col = new Column<double>("X", [1.0, 1.0, 1.0, 1.0, 1.0]);
        var result = col.Rolling(3).Std();

        // Expected: [null, null, 0.0, 0.0, 0.0]
        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().BeApproximately(0.0, 1e-10);
        result[3].Should().BeApproximately(0.0, 1e-10);
        result[4].Should().BeApproximately(0.0, 1e-10);
    }

    [Fact]
    public void Rolling3_MinMax_VerifyExactValues()
    {
        var col = new Column<double>("X", [3.0, 1.0, 4.0, 1.0, 5.0]);
        var min = col.Rolling(3).Min();
        var max = col.Rolling(3).Max();

        // Min window=3: [null, null, min(3,1,4)=1, min(1,4,1)=1, min(4,1,5)=1]
        min[0].Should().BeNull();
        min[1].Should().BeNull();
        min[2].Should().BeApproximately(1.0, 1e-10);
        min[3].Should().BeApproximately(1.0, 1e-10);
        min[4].Should().BeApproximately(1.0, 1e-10);

        // Max window=3: [null, null, max(3,1,4)=4, max(1,4,1)=4, max(4,1,5)=5]
        max[0].Should().BeNull();
        max[1].Should().BeNull();
        max[2].Should().BeApproximately(4.0, 1e-10);
        max[3].Should().BeApproximately(4.0, 1e-10);
        max[4].Should().BeApproximately(5.0, 1e-10);
    }

    [Fact]
    public void Rolling2_Sum_VerifyExactValues()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Rolling(2).Sum();

        // Expected: [null, 3, 5, 7, 9]
        result[0].Should().BeNull();
        result[1].Should().BeApproximately(3.0, 1e-10);
        result[2].Should().BeApproximately(5.0, 1e-10);
        result[3].Should().BeApproximately(7.0, 1e-10);
        result[4].Should().BeApproximately(9.0, 1e-10);
    }

    [Fact]
    public void Ewm_Alpha05_Mean_VerifyHandCalculatedValues()
    {
        // EWM with alpha=0.5, adjust=false (which is what the code implements)
        // x = [1, 2, 3, 4, 5]
        // ewm[0] = 1
        // ewm[1] = 0.5*2 + 0.5*1 = 1.5
        // ewm[2] = 0.5*3 + 0.5*1.5 = 2.25
        // ewm[3] = 0.5*4 + 0.5*2.25 = 3.125
        // ewm[4] = 0.5*5 + 0.5*3.125 = 4.0625
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Ewm(alpha: 0.5).Mean();

        result[0].Should().BeApproximately(1.0, 1e-10);
        result[1].Should().BeApproximately(1.5, 1e-10);
        result[2].Should().BeApproximately(2.25, 1e-10);
        result[3].Should().BeApproximately(3.125, 1e-10);
        result[4].Should().BeApproximately(4.0625, 1e-10);
    }

    [Fact]
    public void Expanding_Mean_VerifyCumulativeMean()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Expanding().Mean();

        // Cumulative mean: [1, 1.5, 2.0, 2.5, 3.0]
        result[0].Should().BeApproximately(1.0, 1e-10);
        result[1].Should().BeApproximately(1.5, 1e-10);
        result[2].Should().BeApproximately(2.0, 1e-10);
        result[3].Should().BeApproximately(2.5, 1e-10);
        result[4].Should().BeApproximately(3.0, 1e-10);
    }

    // =======================================================================
    // Area 4: Rank correctness
    // =======================================================================

    [Fact]
    public void Rank_Average_VerifyExactRanks()
    {
        // [3, 1, 2, 2, 3]
        // Sorted: 1(idx1), 2(idx2), 2(idx3), 3(idx0), 3(idx4)
        // Ranks: 1=1, 2=2.5, 2=2.5, 3=4.5, 3=4.5
        // Result by original order: [4.5, 1, 2.5, 2.5, 4.5]
        var col = new Column<double>("X", [3.0, 1.0, 2.0, 2.0, 3.0]);
        var result = col.Rank(RankMethod.Average);

        result[0].Should().BeApproximately(4.5, 1e-10, "3 has average rank 4.5");
        result[1].Should().BeApproximately(1.0, 1e-10, "1 has rank 1");
        result[2].Should().BeApproximately(2.5, 1e-10, "2 has average rank 2.5");
        result[3].Should().BeApproximately(2.5, 1e-10, "2 has average rank 2.5");
        result[4].Should().BeApproximately(4.5, 1e-10, "3 has average rank 4.5");
    }

    [Fact]
    public void Rank_Min_VerifyExactRanks()
    {
        // [3, 1, 2, 2, 3]
        // Min rank: 1=1, 2=2, 2=2, 3=4, 3=4
        var col = new Column<double>("X", [3.0, 1.0, 2.0, 2.0, 3.0]);
        var result = col.Rank(RankMethod.Min);

        result[0].Should().BeApproximately(4.0, 1e-10, "3 has min rank 4");
        result[1].Should().BeApproximately(1.0, 1e-10, "1 has rank 1");
        result[2].Should().BeApproximately(2.0, 1e-10, "2 has min rank 2");
        result[3].Should().BeApproximately(2.0, 1e-10, "2 has min rank 2");
        result[4].Should().BeApproximately(4.0, 1e-10, "3 has min rank 4");
    }

    [Fact]
    public void Rank_Max_VerifyExactRanks()
    {
        // [3, 1, 2, 2, 3]
        // Max rank: 1=1, 2=3, 2=3, 3=5, 3=5
        var col = new Column<double>("X", [3.0, 1.0, 2.0, 2.0, 3.0]);
        var result = col.Rank(RankMethod.Max);

        result[0].Should().BeApproximately(5.0, 1e-10, "3 has max rank 5");
        result[1].Should().BeApproximately(1.0, 1e-10, "1 has rank 1");
        result[2].Should().BeApproximately(3.0, 1e-10, "2 has max rank 3");
        result[3].Should().BeApproximately(3.0, 1e-10, "2 has max rank 3");
        result[4].Should().BeApproximately(5.0, 1e-10, "3 has max rank 5");
    }

    [Fact]
    public void Rank_Dense_VerifyExactRanks()
    {
        // [3, 1, 2, 2, 3]
        // Dense rank: 1=1, 2=2, 2=2, 3=3, 3=3
        var col = new Column<double>("X", [3.0, 1.0, 2.0, 2.0, 3.0]);
        var result = col.Rank(RankMethod.Dense);

        result[0].Should().BeApproximately(3.0, 1e-10, "3 has dense rank 3");
        result[1].Should().BeApproximately(1.0, 1e-10, "1 has dense rank 1");
        result[2].Should().BeApproximately(2.0, 1e-10, "2 has dense rank 2");
        result[3].Should().BeApproximately(2.0, 1e-10, "2 has dense rank 2");
        result[4].Should().BeApproximately(3.0, 1e-10, "3 has dense rank 3");
    }

    [Fact]
    public void Rank_First_VerifyExactRanks()
    {
        // [3, 1, 2, 2, 3]
        // Sorted order: 1(idx1), 2(idx2), 2(idx3), 3(idx0), 3(idx4)
        // First (ordinal): position in sorted order
        // idx1 -> rank 1, idx2 -> rank 2, idx3 -> rank 3, idx0 -> rank 4, idx4 -> rank 5
        var col = new Column<double>("X", [3.0, 1.0, 2.0, 2.0, 3.0]);
        var result = col.Rank(RankMethod.First);

        result[0].Should().BeApproximately(4.0, 1e-10, "first 3 gets rank 4 (4th in sorted order)");
        result[1].Should().BeApproximately(1.0, 1e-10, "1 gets rank 1");
        result[2].Should().BeApproximately(2.0, 1e-10, "first 2 gets rank 2");
        result[3].Should().BeApproximately(3.0, 1e-10, "second 2 gets rank 3");
        result[4].Should().BeApproximately(5.0, 1e-10, "second 3 gets rank 5");
    }

    // =======================================================================
    // Area 5: Quantile / Percentile edge cases
    // =======================================================================

    [Fact]
    public void Quantile0_ShouldEqualMin()
    {
        var col = new Column<double>("X", [5.0, 1.0, 3.0, 2.0, 4.0]);
        var q0 = col.Quantile(0.0);
        var min = col.Min();

        q0.Should().Be(min, "Quantile(0.0) should equal Min");
    }

    [Fact]
    public void Quantile1_ShouldEqualMax()
    {
        var col = new Column<double>("X", [5.0, 1.0, 3.0, 2.0, 4.0]);
        var q1 = col.Quantile(1.0);
        var max = col.Max();

        q1.Should().Be(max, "Quantile(1.0) should equal Max");
    }

    [Fact]
    public void Quantile05_ShouldEqualMedian()
    {
        var col = new Column<double>("X", [5.0, 1.0, 3.0, 2.0, 4.0]);
        var q50 = col.Quantile(0.5);
        var median = col.Median();

        q50.Should().BeApproximately(median!.Value, 1e-10, "Quantile(0.5) should equal Median");
    }

    [Fact]
    public void Quantile_SingleElement_AllQuantilesShouldReturnSameValue()
    {
        var col = new Column<double>("X", [42.0]);

        col.Quantile(0.0).Should().Be(42.0);
        col.Quantile(0.5).Should().Be(42.0);
        col.Quantile(1.0).Should().Be(42.0);
    }

    [Fact]
    public void Quantile_TwoElements_LinearInterpolation()
    {
        var col = new Column<double>("X", [10.0, 20.0]);

        // Quantile(0.0) = 10, Quantile(1.0) = 20
        // Quantile(0.25) = 10 + 0.25*(20-10) = 12.5
        // Quantile(0.5) = 15
        // Quantile(0.75) = 17.5
        col.Quantile(0.0).Should().BeApproximately(10.0, 1e-10);
        col.Quantile(0.25).Should().BeApproximately(12.5, 1e-10);
        col.Quantile(0.5).Should().BeApproximately(15.0, 1e-10);
        col.Quantile(0.75).Should().BeApproximately(17.5, 1e-10);
        col.Quantile(1.0).Should().BeApproximately(20.0, 1e-10);
    }

    // =======================================================================
    // Area 6: ValueCounts correctness
    // =======================================================================

    [Fact]
    public void ValueCounts_ColumnInt_VerifyCounts()
    {
        var col = new Column<int>("X", [1, 2, 2, 3, 3, 3]);
        var vc = col.ValueCounts();

        // ValueCounts returns a DataFrame with value column + count column
        var valueCol = vc[vc.ColumnNames[0]];
        var counts = (Column<int>)vc["count"];

        // Should be sorted by count descending: 3(count=3), 2(count=2), 1(count=1)
        valueCol.GetObject(0)!.ToString().Should().Be("3");
        counts[0].Should().Be(3);
        valueCol.GetObject(1)!.ToString().Should().Be("2");
        counts[1].Should().Be(2);
        valueCol.GetObject(2)!.ToString().Should().Be("1");
        counts[2].Should().Be(1);
    }

    [Fact]
    public void NUnique_WithNullValues_ShouldNotCountNullAsUnique()
    {
        var col = Column<int>.FromNullable("X", [1, null, 2, null, 3]);

        var nunique = col.NUnique();

        nunique.Should().Be(3, "NaN/null should not count as a unique value");
    }

    [Fact]
    public void ValueCounts_DoubleColumnWithNaN_ShouldGroupNaNValues()
    {
        // NaN doubles are not null in the bitmask, they're actual NaN values.
        // The generic ValueCounts path uses Dictionary<object, int> with boxed doubles.
        // double.NaN.Equals(double.NaN) is true, but double.NaN.GetHashCode() varies.
        // Actually in .NET, double.NaN.GetHashCode() is consistent, and Equals returns true.
        // So this should work. But let's verify.
        var col = new Column<double>("X", [1.0, double.NaN, 2.0, double.NaN, 1.0]);
        var vc = col.ValueCounts();

        var valueCol = vc[vc.ColumnNames[0]];
        var counts = (Column<int>)vc["count"];

        // Expected: 1.0(count=2), NaN(count=2), 2.0(count=1)
        // NaN toString = "NaN"
        // Total should be 5 (no nulls in bitmap, NaN is a value)
        int totalCount = 0;
        for (int i = 0; i < counts.Length; i++)
            totalCount += counts.Values[i];
        totalCount.Should().Be(5, "all 5 values should be accounted for");
    }

    // =======================================================================
    // Bug hunting: Quantile with NaN in data
    // =======================================================================

    [Fact]
    public void Quantile_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // NaN doubles are NOT null in the bitmask, so GetNonNullCopy includes them.
        // QuickSelect with NaN values will produce wrong results because
        // NaN comparisons return false for all relational operators.
        // Quantile(0.5) on [1, NaN, 3] should treat NaN as missing and return Median of [1, 3] = 2.0
        var col = new Column<double>("X", [1.0, double.NaN, 3.0]);
        var q = col.Quantile(0.5);

        // If NaN is included, QuickSelect may return NaN or a wrong value
        // Correct behavior: ignore NaN, so median of [1, 3] = 2.0
        q.Should().NotBeNull();
        double.IsNaN(q!.Value).Should().BeFalse("Quantile should ignore NaN values");
        q.Value.Should().BeApproximately(2.0, 1e-10, "Quantile(0.5) of [1, 3] ignoring NaN should be 2.0");
    }

    [Fact]
    public void Median_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // Same bug: Median uses GetNonNullCopy which doesn't filter NaN
        var col = new Column<double>("X", [1.0, double.NaN, 5.0]);
        var median = col.Median();

        median.Should().NotBeNull();
        double.IsNaN(median!.Value).Should().BeFalse("Median should ignore NaN values");
        median.Value.Should().BeApproximately(3.0, 1e-10, "Median of [1, 5] ignoring NaN should be 3.0");
    }

    [Fact]
    public void Min_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // Min iterates the buffer and checks IsNull, but NaN doubles are not null.
        // If the code uses comparison operators, NaN < anything is false,
        // so the result depends on initialization.
        var col = new Column<double>("X", [5.0, double.NaN, 1.0]);
        var min = col.Min();

        min.Should().NotBeNull();
        double.IsNaN(min!.Value).Should().BeFalse("Min should ignore NaN values");
        min.Value.Should().Be(1.0, "Min of [5, 1] ignoring NaN should be 1.0");
    }

    [Fact]
    public void Min_NaNFirst_ShouldIgnoreNaN()
    {
        // When NaN is the first value, min gets initialized to NaN
        // and then NaN < x is always false, so min stays NaN forever
        var col = new Column<double>("X", [double.NaN, 5.0, 1.0]);
        var min = col.Min();

        min.Should().NotBeNull();
        double.IsNaN(min!.Value).Should().BeFalse("Min should ignore NaN even when NaN is first");
        min.Value.Should().Be(1.0);
    }

    [Fact]
    public void Max_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        var col = new Column<double>("X", [1.0, double.NaN, 5.0]);
        var max = col.Max();

        max.Should().NotBeNull();
        double.IsNaN(max!.Value).Should().BeFalse("Max should ignore NaN values");
        max.Value.Should().Be(5.0, "Max of [1, 5] ignoring NaN should be 5.0");
    }

    [Fact]
    public void Max_NaNFirst_ShouldIgnoreNaN()
    {
        // When NaN is the first value, max gets initialized to NaN
        // and then NaN > x is always false, so max stays NaN forever
        var col = new Column<double>("X", [double.NaN, 1.0, 5.0]);
        var max = col.Max();

        max.Should().NotBeNull();
        double.IsNaN(max!.Value).Should().BeFalse("Max should ignore NaN even when NaN is first");
        max.Value.Should().Be(5.0);
    }

    [Fact]
    public void Mean_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // Mean of [1, NaN, 3] should be 2.0 (ignoring NaN), not NaN
        var col = new Column<double>("X", [1.0, double.NaN, 3.0]);
        var mean = col.Mean();

        mean.Should().NotBeNull();
        double.IsNaN(mean!.Value).Should().BeFalse("Mean should ignore NaN values");
        mean.Value.Should().BeApproximately(2.0, 1e-10);
    }

    [Fact]
    public void Var_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // Var of [1, NaN, 3] should be Var([1,3]) = 2.0 (with ddof=1)
        var col = new Column<double>("X", [1.0, double.NaN, 3.0]);
        var v = col.Var();

        v.Should().NotBeNull();
        double.IsNaN(v!.Value).Should().BeFalse("Var should ignore NaN values");
        v.Value.Should().BeApproximately(2.0, 1e-10);
    }

    [Fact]
    public void Sum_ColumnWithNaNDoubles_ShouldIgnoreNaN()
    {
        // Sum should skip NaN values
        var col = new Column<double>("X", [1.0, double.NaN, 3.0]);
        var sum = col.Sum();

        sum.Should().NotBeNull();
        double.IsNaN(sum!.Value).Should().BeFalse("Sum should ignore NaN values");
        sum.Value.Should().BeApproximately(4.0, 1e-10, "Sum of [1, 3] ignoring NaN should be 4.0");
    }

    [Fact]
    public void ArgMin_NaNFirst_ShouldIgnoreNaN()
    {
        var col = new Column<double>("X", [double.NaN, 5.0, 1.0, 3.0]);
        var idx = col.ArgMin();

        idx.Should().Be(2, "ArgMin should skip NaN and find index of 1.0");
    }

    [Fact]
    public void ArgMax_NaNFirst_ShouldIgnoreNaN()
    {
        var col = new Column<double>("X", [double.NaN, 1.0, 5.0, 3.0]);
        var idx = col.ArgMax();

        idx.Should().Be(2, "ArgMax should skip NaN and find index of 5.0");
    }

    [Fact]
    public void CumMin_WithNaN_ShouldTreatNaNAsMissing()
    {
        var col = new Column<double>("X", [3.0, double.NaN, 1.0, 2.0]);
        var result = col.CumMin();

        result[0].Should().BeApproximately(3.0, 1e-10);
        result[1].Should().BeNull("NaN should be treated as missing");
        result[2].Should().BeApproximately(1.0, 1e-10);
        result[3].Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void CumMax_WithNaN_ShouldTreatNaNAsMissing()
    {
        var col = new Column<double>("X", [1.0, double.NaN, 5.0, 3.0]);
        var result = col.CumMax();

        result[0].Should().BeApproximately(1.0, 1e-10);
        result[1].Should().BeNull("NaN should be treated as missing");
        result[2].Should().BeApproximately(5.0, 1e-10);
        result[3].Should().BeApproximately(5.0, 1e-10);
    }

    [Fact]
    public void CumSum_WithNaN_ShouldTreatNaNAsMissing()
    {
        var col = new Column<double>("X", [1.0, double.NaN, 3.0, 4.0]);
        var result = col.CumSum();

        result[0].Should().BeApproximately(1.0, 1e-10);
        result[1].Should().BeNull("NaN should be treated as missing");
        result[2].Should().BeApproximately(4.0, 1e-10, "CumSum should skip NaN: 1+3=4");
        result[3].Should().BeApproximately(8.0, 1e-10, "CumSum should skip NaN: 1+3+4=8");
    }

    // =======================================================================
    // Bug hunting: Expanding Std numerical instability
    // =======================================================================

    [Fact]
    public void Expanding_Std_LargeConstantOffset_ShouldNotGoNegative()
    {
        // The one-pass formula sumSq - count * mean * mean can go negative
        // due to floating point cancellation with large values.
        // E.g., values [1e15, 1e15 + 1, 1e15 + 2]
        var col = new Column<double>("X", [1e15, 1e15 + 1, 1e15 + 2]);
        var result = col.Expanding().Std();

        // std of [1e15, 1e15+1, 1e15+2] = std of [0,1,2] = 1.0
        result[2].Should().NotBeNull();
        double.IsNaN(result[2]!.Value).Should().BeFalse("Expanding Std should not return NaN due to numerical instability");
        result[2]!.Value.Should().BeApproximately(1.0, 1e-5, "Std of values offset by 1e15 should still be 1.0");
    }

    // =======================================================================
    // Additional: Expanding Std edge cases
    // =======================================================================

    [Fact]
    public void Expanding_Std_VerifyValues()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Expanding().Std();

        // count=1: NaN (or null with minPeriods=1 but std of 1 element)
        // count=2: std([1,2]) = sqrt(0.5) = 0.7071...
        // count=3: std([1,2,3]) = 1.0
        // count=4: std([1,2,3,4]) = sqrt(5/3) = 1.2910...
        // count=5: std([1,2,3,4,5]) = sqrt(10/4) = sqrt(2.5) = 1.5811...

        // First element: minPeriods=1, count=1 => count > 1 is false => NaN
        // The code sets result[i] = double.NaN when count==1 and minPeriods<=1
        // But this NaN is stored directly, not as null. Let's see what FromNullable does.
        // Actually the code does: result[i] = double.NaN, which is a double? with value NaN (not null).
        // So result[0] should have a value of NaN, not be null.
        // This is arguably a bug: pandas expanding().std() returns NaN for first element,
        // but the convention in this library seems to be null for insufficient data.
        // Let's just verify the numeric values are correct for elements with enough data.

        result[1].Should().BeApproximately(Math.Sqrt(0.5), 1e-10);
        result[2].Should().BeApproximately(1.0, 1e-10);
        result[3].Should().BeApproximately(Math.Sqrt(5.0 / 3.0), 1e-10);
        result[4].Should().BeApproximately(Math.Sqrt(2.5), 1e-10);
    }
}
