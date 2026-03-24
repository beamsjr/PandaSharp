using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class MathOpsTests
{
    [Fact]
    public void Abs_ReturnsAbsoluteValues()
    {
        var col = new Column<int>("X", [-3, -1, 0, 1, 3]);
        var result = col.Abs();
        result[0].Should().Be(3);
        result[1].Should().Be(1);
        result[2].Should().Be(0);
    }

    [Fact]
    public void Abs_Double()
    {
        var col = new Column<double>("X", [-1.5, 2.5]);
        var result = col.Abs();
        result[0].Should().Be(1.5);
    }

    [Fact]
    public void Abs_WithNulls()
    {
        var col = Column<int>.FromNullable("X", [-3, null, 3]);
        var result = col.Abs();
        result[0].Should().Be(3);
        result[1].Should().BeNull();
        result[2].Should().Be(3);
    }

    [Fact]
    public void Round_RoundsToDecimals()
    {
        var col = new Column<double>("X", [1.234, 5.678, 9.999]);
        col.Round(2)[0].Should().Be(1.23);
        col.Round(2)[1].Should().Be(5.68);
        col.Round(0)[2].Should().Be(10.0);
    }

    [Fact]
    public void Sqrt_ReturnsSquareRoot()
    {
        var col = new Column<int>("X", [4, 9, 16]);
        var result = col.Sqrt();
        result[0].Should().Be(2.0);
        result[1].Should().Be(3.0);
        result[2].Should().Be(4.0);
    }

    [Fact]
    public void Log_ReturnsNaturalLog()
    {
        var col = new Column<double>("X", [1.0, Math.E, Math.E * Math.E]);
        var result = col.Log();
        result[0].Should().BeApproximately(0.0, 0.001);
        result[1].Should().BeApproximately(1.0, 0.001);
        result[2].Should().BeApproximately(2.0, 0.001);
    }

    [Fact]
    public void Log10_ReturnsBase10Log()
    {
        var col = new Column<double>("X", [1.0, 10.0, 100.0, 1000.0]);
        var result = col.Log10();
        result[0].Should().Be(0.0);
        result[1].Should().Be(1.0);
        result[2].Should().Be(2.0);
        result[3].Should().Be(3.0);
    }

    [Fact]
    public void Pow_RaisesToPower()
    {
        var col = new Column<int>("X", [2, 3, 4]);
        var result = col.Pow(2);
        result[0].Should().Be(4.0);
        result[1].Should().Be(9.0);
        result[2].Should().Be(16.0);
    }

    [Fact]
    public void Pow_FractionalExponent()
    {
        var col = new Column<double>("X", [4.0, 9.0]);
        var result = col.Pow(0.5);
        result[0].Should().Be(2.0);
        result[1].Should().Be(3.0);
    }

    [Fact]
    public void MathOps_ChainWithDataFrame()
    {
        var df = new DataFrame(
            new Column<double>("Value", [-4.0, 9.0, -16.0])
        );

        // Abs then Sqrt
        var absCol = df.GetColumn<double>("Value").Abs();
        var sqrtCol = absCol.Sqrt();

        sqrtCol[0].Should().Be(2.0);
        sqrtCol[1].Should().Be(3.0);
        sqrtCol[2].Should().Be(4.0);
    }
}
