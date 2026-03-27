using FluentAssertions;
using Cortex.Column;
using Cortex.Window;

namespace Cortex.Tests.Unit.Window;

public class ExpandingEwmTests
{
    [Fact]
    public void Expanding_Min()
    {
        var col = new Column<double>("X", [5.0, 3.0, 7.0, 1.0]);
        var result = col.Expanding().Min();
        result[0].Should().Be(5.0);
        result[1].Should().Be(3.0);
        result[2].Should().Be(3.0);
        result[3].Should().Be(1.0);
    }

    [Fact]
    public void Expanding_Max()
    {
        var col = new Column<double>("X", [1.0, 5.0, 3.0, 7.0]);
        var result = col.Expanding().Max();
        result[0].Should().Be(1.0);
        result[1].Should().Be(5.0);
        result[2].Should().Be(5.0);
        result[3].Should().Be(7.0);
    }

    [Fact]
    public void Expanding_Std()
    {
        var col = new Column<double>("X", [2.0, 4.0, 6.0]);
        var result = col.Expanding().Std();
        // First value: std of [2] = NaN (not enough values for sample std)
        result[0].Should().NotBeNull(); // NaN is a value, not null
        result[1].Should().NotBeNull();
        result[2]!.Value.Should().BeApproximately(2.0, 0.01); // std of [2,4,6] = 2
    }

    [Fact]
    public void Expanding_MinPeriods()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0]);
        var result = col.Expanding(minPeriods: 2).Mean();
        result[0].Should().BeNull(); // only 1 value, need 2
        result[1].Should().Be(1.5);
    }

    [Fact]
    public void Ewm_Span3()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0]);
        var result = col.Ewm(span: 3).Mean();
        // span=3 → alpha=0.5
        result[0].Should().Be(1.0);
        result[1].Should().Be(1.5); // 0.5*2 + 0.5*1
    }

    [Fact]
    public void Rolling_WithNulls()
    {
        var col = Column<double>.FromNullable("X", [1.0, null, 3.0, 4.0, 5.0]);
        var result = col.Rolling(3, minPeriods: 1).Mean();
        result[0].Should().Be(1.0); // just [1]
        result[1].Should().Be(1.0); // just [1] (null skipped)
        result[2].Should().Be(2.0); // [1, 3] (null skipped)
    }

    [Fact]
    public void Rolling_Custom_Apply()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0, 4.0, 5.0]);
        var result = col.Rolling(3).Apply(vals => vals.Max() - vals.Min());
        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().Be(2.0); // max(1,2,3) - min(1,2,3) = 2
    }
}
