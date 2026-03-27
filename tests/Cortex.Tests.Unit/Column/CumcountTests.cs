using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class CumcountTests
{
    [Fact]
    public void Cumcount_ReturnsRunningCount()
    {
        var col = new Column<int>("X", [10, 20, 30, 40]);
        var result = col.Cumcount();
        result[0].Should().Be(0);
        result[1].Should().Be(1);
        result[2].Should().Be(2);
        result[3].Should().Be(3);
    }

    [Fact]
    public void Cumcount_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("X", [10, null, 30, null, 50]);
        var result = col.Cumcount();
        result[0].Should().Be(0);
        result[1].Should().BeNull();
        result[2].Should().Be(1);
        result[3].Should().BeNull();
        result[4].Should().Be(2);
    }

    [Fact]
    public void Cumcount_Empty()
    {
        var col = new Column<int>("X", Array.Empty<int>());
        col.Cumcount().Length.Should().Be(0);
    }
}
