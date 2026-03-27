using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Window;

namespace Cortex.Tests.Unit.GroupBy;

public class GroupByShiftTests
{
    /// <summary>
    /// Creates a DataFrame with 3 groups: A (rows 0,3,6), B (rows 1,4,7), C (rows 2,5,8)
    /// Values: 10,20,30,40,50,60,70,80,90
    /// </summary>
    private static (DataFrame Df, GroupedDataFrame Grouped) CreateThreeGroupData()
    {
        var df = new DataFrame(
            new StringColumn("Group", ["A", "B", "C", "A", "B", "C", "A", "B", "C"]),
            new Column<double>("Value", [10, 20, 30, 40, 50, 60, 70, 80, 90])
        );
        return (df, df.GroupBy("Group"));
    }

    [Fact]
    public void Shift_Forward1_NaNAtGroupStarts()
    {
        var (_, grouped) = CreateThreeGroupData();
        var result = grouped.Shift("Value", 1);

        // Group A: indices 0,3,6 -> values 10,40,70
        // Shift(1): [NaN, 10, 40] scattered to positions 0,3,6
        result.Length.Should().Be(9);

        // Position 0 (A, first in group) -> NaN
        double.IsNaN(result[0]!.Value).Should().BeTrue();
        // Position 3 (A, second in group) -> 10
        result[3].Should().Be(10);
        // Position 6 (A, third in group) -> 40
        result[6].Should().Be(40);

        // Group B: indices 1,4,7 -> values 20,50,80
        // Shift(1): [NaN, 20, 50]
        double.IsNaN(result[1]!.Value).Should().BeTrue();
        result[4].Should().Be(20);
        result[7].Should().Be(50);

        // Group C: indices 2,5,8 -> values 30,60,90
        // Shift(1): [NaN, 30, 60]
        double.IsNaN(result[2]!.Value).Should().BeTrue();
        result[5].Should().Be(30);
        result[8].Should().Be(60);
    }

    [Fact]
    public void Shift_Backward1_NaNAtGroupEnds()
    {
        var (_, grouped) = CreateThreeGroupData();
        var result = grouped.Shift("Value", -1);

        // Group A: indices 0,3,6 -> values 10,40,70
        // Shift(-1): [40, 70, NaN] scattered to positions 0,3,6
        result[0].Should().Be(40);
        result[3].Should().Be(70);
        double.IsNaN(result[6]!.Value).Should().BeTrue();

        // Group B: indices 1,4,7 -> values 20,50,80
        result[1].Should().Be(50);
        result[4].Should().Be(80);
        double.IsNaN(result[7]!.Value).Should().BeTrue();
    }

    [Fact]
    public void PctChange_Forward1_NaNAtGroupStartsAndCorrectPercentages()
    {
        var (_, grouped) = CreateThreeGroupData();
        var result = grouped.PctChange("Value", 1);

        result.Length.Should().Be(9);

        // Group A: values 10,40,70
        // PctChange(1): [NaN, (40-10)/10=3.0, (70-40)/40=0.75]
        double.IsNaN(result[0]!.Value).Should().BeTrue();
        result[3].Should().BeApproximately(3.0, 1e-10);
        result[6].Should().BeApproximately(0.75, 1e-10);

        // Group B: values 20,50,80
        // PctChange(1): [NaN, (50-20)/20=1.5, (80-50)/50=0.6]
        double.IsNaN(result[1]!.Value).Should().BeTrue();
        result[4].Should().BeApproximately(1.5, 1e-10);
        result[7].Should().BeApproximately(0.6, 1e-10);
    }

    [Fact]
    public void Rolling_Center_Mean_CorrectAlignmentAndNaN()
    {
        // 10 values, window=5, center=true
        var col = new Column<double>("x", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        var result = col.Rolling(5, center: true).Mean();

        result.Length.Should().Be(10);

        // center=true with window=5: offset = 5/2 = 2
        // Non-centered rolling(5) mean: [null,null,null,null,3,4,5,6,7,8]
        // Shifted left by 2:           [null,null,3,4,5,6,7,8,null,null]
        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().Be(3.0);  // mean(1,2,3,4,5)
        result[3].Should().Be(4.0);  // mean(2,3,4,5,6)
        result[4].Should().Be(5.0);  // mean(3,4,5,6,7)
        result[5].Should().Be(6.0);  // mean(4,5,6,7,8)
        result[6].Should().Be(7.0);  // mean(5,6,7,8,9)
        result[7].Should().Be(8.0);  // mean(6,7,8,9,10)
        result[8].Should().BeNull();
        result[9].Should().BeNull();
    }

    [Fact]
    public void Shift_EmptyGroup_Succeeds()
    {
        // A DataFrame where groups have varying sizes including a group tested with large shift
        var df = new DataFrame(
            new StringColumn("Group", ["A"]),
            new Column<double>("Value", [42.0])
        );
        var grouped = df.GroupBy("Group");

        // Single element group, shift by 1 -> all NaN
        var result = grouped.Shift("Value", 1);
        result.Length.Should().Be(1);
        double.IsNaN(result[0]!.Value).Should().BeTrue();
    }

    [Fact]
    public void Shift_SingleElementGroup_AllNaN()
    {
        var df = new DataFrame(
            new StringColumn("Group", ["A", "B", "C"]),
            new Column<double>("Value", [1.0, 2.0, 3.0])
        );
        var grouped = df.GroupBy("Group");

        // Each group has 1 element, shift(1) -> all NaN
        var result = grouped.Shift("Value", 1);
        for (int i = 0; i < 3; i++)
            double.IsNaN(result[i]!.Value).Should().BeTrue();
    }

    [Fact]
    public void Shift_LargerThanGroupSize_AllNaN()
    {
        var (_, grouped) = CreateThreeGroupData();
        // Each group has 3 elements, shift by 5 -> all NaN
        var result = grouped.Shift("Value", 5);
        for (int i = 0; i < 9; i++)
            double.IsNaN(result[i]!.Value).Should().BeTrue();
    }

    [Fact]
    public void PctChange_SingleElementGroup_AllNaN()
    {
        var df = new DataFrame(
            new StringColumn("Group", ["A", "B"]),
            new Column<double>("Value", [10.0, 20.0])
        );
        var grouped = df.GroupBy("Group");

        var result = grouped.PctChange("Value", 1);
        for (int i = 0; i < 2; i++)
            double.IsNaN(result[i]!.Value).Should().BeTrue();
    }

    [Fact]
    public void Rolling_Center_Sum_CorrectAlignment()
    {
        var col = new Column<double>("x", [1, 2, 3, 4, 5]);
        var result = col.Rolling(3, center: true).Sum();

        // center=true with window=3: Apply uses centered window
        // Position 0: window [-1,0,1] clamped to [0,1] -> 1+2=3, count=2 < minPeriods=3 -> null
        // Position 1: window [0,1,2] -> 1+2+3=6
        // Position 2: window [1,2,3] -> 2+3+4=9
        // Position 3: window [2,3,4] -> 3+4+5=12
        // Position 4: window [3,4,5] clamped to [3,4] -> 4+5=9, count=2 < minPeriods=3 -> null
        result[0].Should().BeNull();
        result[1].Should().Be(6.0);
        result[2].Should().Be(9.0);
        result[3].Should().Be(12.0);
        result[4].Should().BeNull();
    }

    [Fact]
    public void Shift_Zero_ReturnsOriginalValues()
    {
        var (_, grouped) = CreateThreeGroupData();
        var result = grouped.Shift("Value", 0);

        result[0].Should().Be(10);
        result[1].Should().Be(20);
        result[2].Should().Be(30);
        result[3].Should().Be(40);
        result[4].Should().Be(50);
        result[5].Should().Be(60);
        result[6].Should().Be(70);
        result[7].Should().Be(80);
        result[8].Should().Be(90);
    }
}
