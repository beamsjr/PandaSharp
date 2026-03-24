using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class NormalizeTests
{
    [Fact]
    public void NormalizeMinMax_ScalesTo01()
    {
        var col = new Column<double>("X", [0.0, 50.0, 100.0]);
        var result = col.NormalizeMinMax();

        result[0].Should().Be(0.0);
        result[1].Should().Be(0.5);
        result[2].Should().Be(1.0);
    }

    [Fact]
    public void NormalizeMinMax_IntColumn()
    {
        var col = new Column<int>("X", [10, 20, 30, 40, 50]);
        var result = col.NormalizeMinMax();

        result[0].Should().Be(0.0);
        result[2].Should().Be(0.5);
        result[4].Should().Be(1.0);
    }

    [Fact]
    public void NormalizeMinMax_WithNulls()
    {
        var col = Column<double>.FromNullable("X", [0.0, null, 100.0]);
        var result = col.NormalizeMinMax();

        result[0].Should().Be(0.0);
        result[1].Should().BeNull();
        result[2].Should().Be(1.0);
    }

    [Fact]
    public void NormalizeMinMax_AllSame_Returns05()
    {
        var col = new Column<double>("X", [5.0, 5.0, 5.0]);
        var result = col.NormalizeMinMax();
        result[0].Should().Be(0.5);
    }

    [Fact]
    public void NormalizeZScore_MeanZeroStdOne()
    {
        var col = new Column<double>("X", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        var result = col.NormalizeZScore();

        // Mean of result should be ~0
        double sum = 0;
        int count = 0;
        for (int i = 0; i < result.Length; i++)
        {
            if (result[i].HasValue) { sum += result[i]!.Value; count++; }
        }
        (sum / count).Should().BeApproximately(0, 0.001);
    }

    [Fact]
    public void NormalizeZScore_IntColumn()
    {
        var col = new Column<int>("X", [10, 20, 30]);
        var result = col.NormalizeZScore();

        result[0].Should().NotBeNull();
        result[1].Should().BeApproximately(0.0, 0.001); // mean = 20, this is the middle
    }

    [Fact]
    public void NormalizeZScore_WithNulls()
    {
        var col = Column<double>.FromNullable("X", [10.0, null, 30.0]);
        var result = col.NormalizeZScore();

        result[1].Should().BeNull();
    }
}
