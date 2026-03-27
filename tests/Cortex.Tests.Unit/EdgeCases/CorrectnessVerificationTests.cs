using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Statistics;
using Cortex.Window;
using Xunit;

namespace Cortex.Tests.Unit.EdgeCases;

/// <summary>
/// Round 8: Correctness verification tests.
/// Verifies that Cortex produces the RIGHT NUMBERS for known inputs.
/// </summary>
public class CorrectnessVerificationTests
{
    // ═══════════════════════════════════════════════════════════════
    // Area 1: GroupBy Aggregation Correctness
    // Data: key=["a","a","a","b","b"], val=[10, 20, 30, 40, 50]
    // ═══════════════════════════════════════════════════════════════

    private DataFrame CreateGroupByTestData()
    {
        return DataFrame.FromDictionary(new()
        {
            ["key"] = new[] { "a", "a", "a", "b", "b" },
            ["val"] = new[] { 10, 20, 30, 40, 50 }
        });
    }

    [Fact]
    public void GroupBy_Sum_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Sum();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        Convert.ToDouble(result["val"].GetObject(aRow)).Should().Be(60.0, "a: 10+20+30=60");
        Convert.ToDouble(result["val"].GetObject(bRow)).Should().Be(90.0, "b: 40+50=90");
    }

    [Fact]
    public void GroupBy_Mean_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Mean();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        Convert.ToDouble(result["val"].GetObject(aRow)).Should().Be(20.0, "a: (10+20+30)/3=20");
        Convert.ToDouble(result["val"].GetObject(bRow)).Should().Be(45.0, "b: (40+50)/2=45");
    }

    [Fact]
    public void GroupBy_Std_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Std();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        // a: sample std of [10,20,30] = sqrt(((10-20)^2+(20-20)^2+(30-20)^2)/2) = sqrt(200/2) = 10.0
        Convert.ToDouble(result["val"].GetObject(aRow)).Should().BeApproximately(10.0, 1e-10, "a: std([10,20,30])=10");
        // b: sample std of [40,50] = sqrt(((40-45)^2+(50-45)^2)/1) = sqrt(50/1) = 7.0710678...
        Convert.ToDouble(result["val"].GetObject(bRow)).Should().BeApproximately(Math.Sqrt(50.0), 1e-10, "b: std([40,50])=sqrt(50)");
    }

    [Fact]
    public void GroupBy_Min_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Min();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        Convert.ToDouble(result["val"].GetObject(aRow)).Should().Be(10.0, "a: min=10");
        Convert.ToDouble(result["val"].GetObject(bRow)).Should().Be(40.0, "b: min=40");
    }

    [Fact]
    public void GroupBy_Max_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Max();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        Convert.ToDouble(result["val"].GetObject(aRow)).Should().Be(30.0, "a: max=30");
        Convert.ToDouble(result["val"].GetObject(bRow)).Should().Be(50.0, "b: max=50");
    }

    [Fact]
    public void GroupBy_Count_ReturnsCorrectValues()
    {
        var df = CreateGroupByTestData();
        var result = df.GroupBy("key").Count();

        var aRow = GetGroupRow(result, "key", "a");
        var bRow = GetGroupRow(result, "key", "b");

        Convert.ToInt32(result["val"].GetObject(aRow)).Should().Be(3, "a: count=3");
        Convert.ToInt32(result["val"].GetObject(bRow)).Should().Be(2, "b: count=2");
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 2: Correlation Correctness
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Corr_PerfectPositiveLinear_ReturnsExactlyOne()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["x"] = new double[] { 1, 2, 3, 4, 5 },
            ["y"] = new double[] { 2, 4, 6, 8, 10 }
        });

        var corr = df.Corr();
        var xyCorr = (double)corr["y"].GetObject(0)!; // row 0 is "x", col "y"

        xyCorr.Should().BeApproximately(1.0, 1e-10, "perfect positive linear => r=1.0");
    }

    [Fact]
    public void Corr_PerfectNegativeLinear_ReturnsExactlyMinusOne()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["x"] = new double[] { 1, 2, 3, 4, 5 },
            ["y"] = new double[] { 10, 8, 6, 4, 2 }
        });

        var corr = df.Corr();
        var xyCorr = (double)corr["y"].GetObject(0)!;

        xyCorr.Should().BeApproximately(-1.0, 1e-10, "perfect negative linear => r=-1.0");
    }

    [Fact]
    public void Corr_KnownValues_ReturnsCorrectPearsonR()
    {
        // x=[1,2,3,4,5], y=[2,4,1,5,3]
        // mx=3, my=3
        // dx=[-2,-1,0,1,2], dy=[-1,1,-2,2,0]
        // sum(dx*dy)=2+(-1)+0+2+0=3
        // sum(dx^2)=10, sum(dy^2)=10
        // r = 3/sqrt(10*10) = 0.3
        var df = DataFrame.FromDictionary(new()
        {
            ["x"] = new double[] { 1, 2, 3, 4, 5 },
            ["y"] = new double[] { 2, 4, 1, 5, 3 }
        });

        var corr = df.Corr();
        var xyCorr = (double)corr["y"].GetObject(0)!;

        xyCorr.Should().BeApproximately(0.3, 1e-10, "computed Pearson r should be 0.3");
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 3: Rolling Window Correctness
    // Data: [1, 2, 3, 4, 5, 6], window=3
    // ═══════════════════════════════════════════════════════════════

    private Column<double> CreateRollingTestColumn()
    {
        return new Column<double>("val", new double[] { 1, 2, 3, 4, 5, 6 });
    }

    [Fact]
    public void Rolling_Sum_ReturnsCorrectValues()
    {
        var col = CreateRollingTestColumn();
        var result = col.Rolling(3).Sum();
        // Expected: [NaN, NaN, 6, 9, 12, 15]

        result.GetObject(0).Should().BeNull("position 0: not enough data");
        result.GetObject(1).Should().BeNull("position 1: not enough data");
        ((double)result.GetObject(2)!).Should().Be(6.0, "1+2+3=6");
        ((double)result.GetObject(3)!).Should().Be(9.0, "2+3+4=9");
        ((double)result.GetObject(4)!).Should().Be(12.0, "3+4+5=12");
        ((double)result.GetObject(5)!).Should().Be(15.0, "4+5+6=15");
    }

    [Fact]
    public void Rolling_Mean_ReturnsCorrectValues()
    {
        var col = CreateRollingTestColumn();
        var result = col.Rolling(3).Mean();
        // Expected: [NaN, NaN, 2.0, 3.0, 4.0, 5.0]

        result.GetObject(0).Should().BeNull("position 0: not enough data");
        result.GetObject(1).Should().BeNull("position 1: not enough data");
        ((double)result.GetObject(2)!).Should().Be(2.0, "(1+2+3)/3=2");
        ((double)result.GetObject(3)!).Should().Be(3.0, "(2+3+4)/3=3");
        ((double)result.GetObject(4)!).Should().Be(4.0, "(3+4+5)/3=4");
        ((double)result.GetObject(5)!).Should().Be(5.0, "(4+5+6)/3=5");
    }

    [Fact]
    public void Rolling_Min_ReturnsCorrectValues()
    {
        var col = CreateRollingTestColumn();
        var result = col.Rolling(3).Min();
        // Expected: [NaN, NaN, 1, 2, 3, 4]

        result.GetObject(0).Should().BeNull();
        result.GetObject(1).Should().BeNull();
        ((double)result.GetObject(2)!).Should().Be(1.0);
        ((double)result.GetObject(3)!).Should().Be(2.0);
        ((double)result.GetObject(4)!).Should().Be(3.0);
        ((double)result.GetObject(5)!).Should().Be(4.0);
    }

    [Fact]
    public void Rolling_Max_ReturnsCorrectValues()
    {
        var col = CreateRollingTestColumn();
        var result = col.Rolling(3).Max();
        // Expected: [NaN, NaN, 3, 4, 5, 6]

        result.GetObject(0).Should().BeNull();
        result.GetObject(1).Should().BeNull();
        ((double)result.GetObject(2)!).Should().Be(3.0);
        ((double)result.GetObject(3)!).Should().Be(4.0);
        ((double)result.GetObject(4)!).Should().Be(5.0);
        ((double)result.GetObject(5)!).Should().Be(6.0);
    }

    [Fact]
    public void Rolling_Std_ReturnsCorrectValues()
    {
        var col = CreateRollingTestColumn();
        var result = col.Rolling(3).Std();
        // Window [1,2,3]: mean=2, sum_sq=2, std=sqrt(2/2)=1.0
        // Window [2,3,4]: mean=3, sum_sq=2, std=sqrt(2/2)=1.0
        // etc. all windows have std=1.0

        result.GetObject(0).Should().BeNull();
        result.GetObject(1).Should().BeNull();
        ((double)result.GetObject(2)!).Should().BeApproximately(1.0, 1e-10);
        ((double)result.GetObject(3)!).Should().BeApproximately(1.0, 1e-10);
        ((double)result.GetObject(4)!).Should().BeApproximately(1.0, 1e-10);
        ((double)result.GetObject(5)!).Should().BeApproximately(1.0, 1e-10);
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 4: Rank Correctness
    // Data: [30, 10, 20, 20, 40]
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Rank_Average_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 30, 10, 20, 20, 40 });
        var result = col.Rank(RankMethod.Average);
        // Sorted: 10(1), 20(2), 20(3), 30(4), 40(5)
        // Expected: [4.0, 1.0, 2.5, 2.5, 5.0]

        ((double)result.GetObject(0)!).Should().Be(4.0, "30 is rank 4");
        ((double)result.GetObject(1)!).Should().Be(1.0, "10 is rank 1");
        ((double)result.GetObject(2)!).Should().Be(2.5, "20 ties at ranks 2,3 => avg 2.5");
        ((double)result.GetObject(3)!).Should().Be(2.5, "20 ties at ranks 2,3 => avg 2.5");
        ((double)result.GetObject(4)!).Should().Be(5.0, "40 is rank 5");
    }

    [Fact]
    public void Rank_Min_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 30, 10, 20, 20, 40 });
        var result = col.Rank(RankMethod.Min);
        // Expected: [4, 1, 2, 2, 5]

        ((double)result.GetObject(0)!).Should().Be(4.0);
        ((double)result.GetObject(1)!).Should().Be(1.0);
        ((double)result.GetObject(2)!).Should().Be(2.0);
        ((double)result.GetObject(3)!).Should().Be(2.0);
        ((double)result.GetObject(4)!).Should().Be(5.0);
    }

    [Fact]
    public void Rank_Max_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 30, 10, 20, 20, 40 });
        var result = col.Rank(RankMethod.Max);
        // Expected: [4, 1, 3, 3, 5]

        ((double)result.GetObject(0)!).Should().Be(4.0);
        ((double)result.GetObject(1)!).Should().Be(1.0);
        ((double)result.GetObject(2)!).Should().Be(3.0);
        ((double)result.GetObject(3)!).Should().Be(3.0);
        ((double)result.GetObject(4)!).Should().Be(5.0);
    }

    [Fact]
    public void Rank_Dense_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 30, 10, 20, 20, 40 });
        var result = col.Rank(RankMethod.Dense);
        // Expected: [3, 1, 2, 2, 4]

        ((double)result.GetObject(0)!).Should().Be(3.0);
        ((double)result.GetObject(1)!).Should().Be(1.0);
        ((double)result.GetObject(2)!).Should().Be(2.0);
        ((double)result.GetObject(3)!).Should().Be(2.0);
        ((double)result.GetObject(4)!).Should().Be(4.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 5: Describe Correctness
    // Data: [1, 2, 3, 4, 5]
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Describe_ReturnsCorrectStatistics()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["val"] = new double[] { 1, 2, 3, 4, 5 }
        });

        var desc = df.Describe();
        // stat column: ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]
        // indices:        0       1       2       3      4      5      6       7

        var vals = desc.GetColumn<double>("val");

        vals.Values[0].Should().Be(5.0, "count=5");
        vals.Values[1].Should().Be(3.0, "mean=3.0");
        vals.Values[2].Should().BeApproximately(Math.Sqrt(2.5), 1e-10, "std=sqrt(2.5)=1.5811...");
        vals.Values[3].Should().Be(1.0, "min=1");
        vals.Values[4].Should().Be(2.0, "25%=2.0");
        vals.Values[5].Should().Be(3.0, "50%=3.0 (median)");
        vals.Values[6].Should().Be(4.0, "75%=4.0");
        vals.Values[7].Should().Be(5.0, "max=5");
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 6: CumSum / CumProd / CumMin / CumMax Correctness
    // Data: [1, 2, 3, 4, 5]
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CumSum_ReturnsCorrectValues()
    {
        var col = new Column<int>("val", new int[] { 1, 2, 3, 4, 5 });
        var result = col.CumSum();
        // Expected: [1, 3, 6, 10, 15]

        result.Values[0].Should().Be(1);
        result.Values[1].Should().Be(3);
        result.Values[2].Should().Be(6);
        result.Values[3].Should().Be(10);
        result.Values[4].Should().Be(15);
    }

    [Fact]
    public void CumProd_ReturnsCorrectValues()
    {
        var col = new Column<int>("val", new int[] { 1, 2, 3, 4, 5 });
        var result = col.CumProd();
        // Expected: [1, 2, 6, 24, 120]

        result.Values[0].Should().Be(1);
        result.Values[1].Should().Be(2);
        result.Values[2].Should().Be(6);
        result.Values[3].Should().Be(24);
        result.Values[4].Should().Be(120);
    }

    [Fact]
    public void CumMin_ReturnsCorrectValues()
    {
        var col = new Column<int>("val", new int[] { 1, 2, 3, 4, 5 });
        var result = col.CumMin();
        // Expected: [1, 1, 1, 1, 1]

        result.Values[0].Should().Be(1);
        result.Values[1].Should().Be(1);
        result.Values[2].Should().Be(1);
        result.Values[3].Should().Be(1);
        result.Values[4].Should().Be(1);
    }

    [Fact]
    public void CumMax_ReturnsCorrectValues()
    {
        var col = new Column<int>("val", new int[] { 1, 2, 3, 4, 5 });
        var result = col.CumMax();
        // Expected: [1, 2, 3, 4, 5]

        result.Values[0].Should().Be(1);
        result.Values[1].Should().Be(2);
        result.Values[2].Should().Be(3);
        result.Values[3].Should().Be(4);
        result.Values[4].Should().Be(5);
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 7: PctChange Correctness
    // Data: [100, 110, 99, 121]
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void PctChange_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 100, 110, 99, 121 });
        var result = col.PctChange();
        // Expected: [NaN, 0.1, -0.1, 0.222222...]

        result.GetObject(0).Should().BeNull("first element has no previous");
        ((double)result.GetObject(1)!).Should().BeApproximately(0.1, 1e-10, "(110-100)/100=0.1");
        ((double)result.GetObject(2)!).Should().BeApproximately(-0.1, 1e-10, "(99-110)/110=-0.1");
        ((double)result.GetObject(3)!).Should().BeApproximately(22.0 / 99.0, 1e-10, "(121-99)/99=22/99");
    }

    // ═══════════════════════════════════════════════════════════════
    // Area 8: Quantile Correctness
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Quantile_FiveElements_ReturnsCorrectValues()
    {
        var col = new Column<double>("val", new double[] { 1, 2, 3, 4, 5 });

        col.Quantile(0.0).Should().Be(1.0, "Q(0.0) = min = 1");
        col.Quantile(0.25).Should().Be(2.0, "Q(0.25) = 2.0");
        col.Quantile(0.5).Should().Be(3.0, "Q(0.5) = median = 3.0");
        col.Quantile(0.75).Should().Be(4.0, "Q(0.75) = 4.0");
        col.Quantile(1.0).Should().Be(5.0, "Q(1.0) = max = 5");
    }

    [Fact]
    public void Quantile_TwoElements_InterpolatesCorrectly()
    {
        var col = new Column<double>("val", new double[] { 10, 20 });

        col.Quantile(0.5).Should().Be(15.0, "Q(0.5) of [10,20] = 15.0 (interpolated)");
        col.Quantile(0.25).Should().Be(12.5, "Q(0.25) of [10,20] = 12.5 (interpolated)");
    }

    // ═══════════════════════════════════════════════════════════════
    // Helper
    // ═══════════════════════════════════════════════════════════════

    private static int GetGroupRow(DataFrame df, string keyCol, string keyValue)
    {
        for (int i = 0; i < df.RowCount; i++)
        {
            var val = df[keyCol].GetObject(i)?.ToString();
            if (val == keyValue) return i;
        }
        throw new InvalidOperationException($"Group '{keyValue}' not found in column '{keyCol}'");
    }
}
