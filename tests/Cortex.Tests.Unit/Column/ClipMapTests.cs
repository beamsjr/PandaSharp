using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ClipMapTests
{
    [Fact]
    public void Clip_ClampsValues()
    {
        var col = new Column<int>("X", [1, 5, 10, 15, 20]);
        var clipped = col.Clip(5, 15);

        clipped[0].Should().Be(5);   // clamped up
        clipped[1].Should().Be(5);
        clipped[2].Should().Be(10);  // unchanged
        clipped[3].Should().Be(15);
        clipped[4].Should().Be(15);  // clamped down
    }

    [Fact]
    public void Clip_WithNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 20]);
        var clipped = col.Clip(5, 15);

        clipped[0].Should().Be(5);
        clipped[1].Should().BeNull();
        clipped[2].Should().Be(15);
    }

    [Fact]
    public void Clip_Double()
    {
        var col = new Column<double>("X", [0.5, 1.5, 2.5]);
        var clipped = col.Clip(1.0, 2.0);

        clipped[0].Should().Be(1.0);
        clipped[1].Should().Be(1.5);
        clipped[2].Should().Be(2.0);
    }

    [Fact]
    public void Where_ReplacesMatchingValues()
    {
        var col = new Column<int>("X", [1, 2, 3, 4, 5]);
        var result = col.Where(v => v > 3, 0);

        result[0].Should().Be(1);
        result[3].Should().Be(0);
        result[4].Should().Be(0);
    }

    [Fact]
    public void Map_TransformsValues()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.Map(v => v * v);

        result[0].Should().Be(1);
        result[1].Should().Be(4);
        result[2].Should().Be(9);
    }

    [Fact]
    public void Map_CrossType()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.Map(v => (double)v / 2);

        result[0].Should().Be(0.5);
        result[1].Should().Be(1.0);
        result[2].Should().Be(1.5);
    }

    [Fact]
    public void Map_WithNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 3]);
        var result = col.Map(v => v * 10);

        result[0].Should().Be(10);
        result[1].Should().BeNull();
        result[2].Should().Be(30);
    }
}
