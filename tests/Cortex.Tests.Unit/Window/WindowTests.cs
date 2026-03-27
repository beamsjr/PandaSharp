using FluentAssertions;
using Cortex.Column;
using Cortex.Window;

namespace Cortex.Tests.Unit.Window;

public class WindowTests
{
    [Fact]
    public void Rolling_Mean_ReturnsMovingAverage()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Rolling(3).Mean();

        result[0].Should().BeNull(); // not enough values
        result[1].Should().BeNull();
        result[2].Should().Be(2.0); // (1+2+3)/3
        result[3].Should().Be(3.0); // (2+3+4)/3
        result[4].Should().Be(4.0); // (3+4+5)/3
    }

    [Fact]
    public void Rolling_Sum_ReturnsMovingSum()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0]);
        var result = col.Rolling(2).Sum();

        result[0].Should().BeNull();
        result[1].Should().Be(3.0);
        result[2].Should().Be(5.0);
        result[3].Should().Be(7.0);
    }

    [Fact]
    public void Rolling_Min_ReturnsMovingMin()
    {
        var col = new Column<double>("x", [3.0, 1.0, 4.0, 1.0, 5.0]);
        var result = col.Rolling(3).Min();

        result[2].Should().Be(1.0);
        result[3].Should().Be(1.0);
        result[4].Should().Be(1.0);
    }

    [Fact]
    public void Rolling_Max_ReturnsMovingMax()
    {
        var col = new Column<double>("x", [3.0, 1.0, 4.0, 1.0, 5.0]);
        var result = col.Rolling(3).Max();

        result[2].Should().Be(4.0);
        result[3].Should().Be(4.0);
        result[4].Should().Be(5.0);
    }

    [Fact]
    public void Rolling_MinPeriods_AllowsPartialWindows()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        var result = col.Rolling(3, minPeriods: 1).Mean();

        result[0].Should().Be(1.0); // just 1 value
        result[1].Should().Be(1.5); // 1,2
        result[2].Should().Be(2.0); // 1,2,3
    }

    [Fact]
    public void Rolling_Center_CentersWindow()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Rolling(3, center: true).Mean();

        result[0].Should().BeNull();
        result[1].Should().Be(2.0); // (1+2+3)/3
        result[2].Should().Be(3.0); // (2+3+4)/3
        result[3].Should().Be(4.0); // (3+4+5)/3
        result[4].Should().BeNull();
    }

    [Fact]
    public void Expanding_Mean_ReturnsExpandingAverage()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0]);
        var result = col.Expanding().Mean();

        result[0].Should().Be(1.0);
        result[1].Should().Be(1.5);
        result[2].Should().Be(2.0);
        result[3].Should().Be(2.5);
    }

    [Fact]
    public void Expanding_Sum_ReturnsExpandingSum()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        var result = col.Expanding().Sum();

        result[0].Should().Be(1.0);
        result[1].Should().Be(3.0);
        result[2].Should().Be(6.0);
    }

    [Fact]
    public void Ewm_Mean_ReturnsExponentiallyWeightedMean()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        var result = col.Ewm(span: 2).Mean(); // alpha = 2/3

        result[0].Should().Be(1.0);
        result[1].Should().BeApproximately(1.6667, 0.001); // 2/3*2 + 1/3*1
        result[2].Should().BeApproximately(2.5556, 0.001); // 2/3*3 + 1/3*1.667
    }

    [Fact]
    public void Ewm_WithAlpha()
    {
        var col = new Column<double>("x", [10.0, 20.0, 30.0]);
        var result = col.Ewm(alpha: 0.5).Mean();

        result[0].Should().Be(10.0);
        result[1].Should().Be(15.0); // 0.5*20 + 0.5*10
        result[2].Should().Be(22.5); // 0.5*30 + 0.5*15
    }

    [Fact]
    public void Rolling_Std_ReturnsMovingStd()
    {
        var col = new Column<double>("x", [2.0, 4.0, 6.0, 8.0]);
        var result = col.Rolling(3).Std();

        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().Be(2.0); // std of [2,4,6]
    }
}
