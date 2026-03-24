using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class ColumnExtensionsTests
{
    [Fact]
    public void Gt_ReturnsTrueForGreaterValues()
    {
        var col = new Column<int>("Val", [10, 20, 30, 40, 50]);
        var mask = col.Gt(25);

        mask.Should().Equal([false, false, true, true, true]);
    }

    [Fact]
    public void Lt_ReturnsTrueForLesserValues()
    {
        var col = new Column<int>("Val", [10, 20, 30, 40, 50]);
        var mask = col.Lt(30);

        mask.Should().Equal([true, true, false, false, false]);
    }

    [Fact]
    public void Eq_ReturnsTrueForEqualValues()
    {
        var col = new Column<int>("Val", [10, 20, 30]);
        var mask = col.Eq(20);

        mask.Should().Equal([false, true, false]);
    }

    [Fact]
    public void Gt_WithNulls_ReturnsFalseForNulls()
    {
        var col = Column<int>.FromNullable("Val", [10, null, 30]);
        var mask = col.Gt(5);

        mask.Should().Equal([true, false, true]);
    }

    [Fact]
    public void Sum_ReturnsTotal()
    {
        var col = new Column<int>("Val", [10, 20, 30]);
        col.Sum().Should().Be(60);
    }

    [Fact]
    public void Sum_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("Val", [10, null, 30]);
        col.Sum().Should().Be(40);
    }

    [Fact]
    public void Mean_ReturnsAverage()
    {
        var col = new Column<double>("Val", [10.0, 20.0, 30.0]);
        col.Mean().Should().Be(20.0);
    }

    [Fact]
    public void Mean_WithNulls_ExcludesNulls()
    {
        var col = Column<double>.FromNullable("Val", [10.0, null, 30.0]);
        col.Mean().Should().Be(20.0);
    }

    [Fact]
    public void Min_ReturnsSmallest()
    {
        var col = new Column<int>("Val", [30, 10, 20]);
        col.Min().Should().Be(10);
    }

    [Fact]
    public void Max_ReturnsLargest()
    {
        var col = new Column<int>("Val", [30, 10, 20]);
        col.Max().Should().Be(30);
    }

    [Fact]
    public void Sum_EmptyColumn_ReturnsNull()
    {
        var col = new Column<int>("Val", Array.Empty<int>());
        col.Sum().Should().BeNull();
    }

    [Fact]
    public void Gte_ReturnsTrueForGreaterOrEqual()
    {
        var col = new Column<int>("Val", [10, 20, 30]);
        col.Gte(20).Should().Equal([false, true, true]);
    }

    [Fact]
    public void Lte_ReturnsTrueForLessOrEqual()
    {
        var col = new Column<int>("Val", [10, 20, 30]);
        col.Lte(20).Should().Equal([true, true, false]);
    }
}
