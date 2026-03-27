using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ArgMinMaxTests
{
    [Fact]
    public void ArgMin_ReturnsIndexOfSmallest()
    {
        var col = new Column<int>("X", [30, 10, 50, 20, 40]);
        col.ArgMin().Should().Be(1);
    }

    [Fact]
    public void ArgMax_ReturnsIndexOfLargest()
    {
        var col = new Column<int>("X", [30, 10, 50, 20, 40]);
        col.ArgMax().Should().Be(2);
    }

    [Fact]
    public void ArgMin_Double()
    {
        var col = new Column<double>("X", [3.5, 1.2, 4.8]);
        col.ArgMin().Should().Be(1);
    }

    [Fact]
    public void ArgMax_Double()
    {
        var col = new Column<double>("X", [3.5, 1.2, 4.8]);
        col.ArgMax().Should().Be(2);
    }

    [Fact]
    public void ArgMin_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("X", [30, null, 10, null]);
        col.ArgMin().Should().Be(2);
    }

    [Fact]
    public void ArgMax_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("X", [null, 50, null, 20]);
        col.ArgMax().Should().Be(1);
    }

    [Fact]
    public void ArgMin_Empty_ReturnsNull()
    {
        var col = new Column<int>("X", Array.Empty<int>());
        col.ArgMin().Should().BeNull();
    }

    [Fact]
    public void ArgMax_AllNull_ReturnsNull()
    {
        var col = Column<int>.FromNullable("X", [null, null]);
        col.ArgMax().Should().BeNull();
    }

    [Fact]
    public void ArgMin_UsefulWithDataFrame()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<double>("Score", [85.0, 92.0, 78.0])
        );

        int? minIdx = df.GetColumn<double>("Score").ArgMin();
        minIdx.Should().Be(2);
        df.GetStringColumn("Name")[minIdx!.Value].Should().Be("Charlie");
    }
}
