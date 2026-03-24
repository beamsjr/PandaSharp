using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Missing;

namespace PandaSharp.Tests.Unit;

public class MissingDataEdgeCaseTests
{
    [Fact]
    public void FillNa_Forward_AllNulls_StaysNull()
    {
        var col = Column<int>.FromNullable("X", [null, null, null]);
        var result = col.FillNa(FillStrategy.Forward);
        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().BeNull();
    }

    [Fact]
    public void FillNa_Backward_AllNulls_StaysNull()
    {
        var col = Column<int>.FromNullable("X", [null, null, null]);
        var result = col.FillNa(FillStrategy.Backward);
        result[0].Should().BeNull();
    }

    [Fact]
    public void FillNa_Scalar_NoNulls_Unchanged()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.FillNa(0);
        result[0].Should().Be(1);
        result[1].Should().Be(2);
    }

    [Fact]
    public void DropNa_NoNulls_ReturnsSame()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );
        df.DropNa().RowCount.Should().Be(3);
    }

    [Fact]
    public void Interpolate_NoNulls_Unchanged()
    {
        var col = new Column<double>("X", [1.0, 2.0, 3.0]);
        var result = col.Interpolate();
        result[0].Should().Be(1.0);
        result[1].Should().Be(2.0);
        result[2].Should().Be(3.0);
    }

    [Fact]
    public void Interpolate_SingleNull_InMiddle()
    {
        var col = Column<double>.FromNullable("X", [0.0, null, 10.0]);
        var result = col.Interpolate();
        result[1].Should().Be(5.0);
    }

    [Fact]
    public void Interpolate_MultipleConsecutiveNulls()
    {
        var col = Column<double>.FromNullable("X", [0.0, null, null, null, 8.0]);
        var result = col.Interpolate();
        result[1].Should().BeApproximately(2.0, 0.01);
        result[2].Should().BeApproximately(4.0, 0.01);
        result[3].Should().BeApproximately(6.0, 0.01);
    }
}
