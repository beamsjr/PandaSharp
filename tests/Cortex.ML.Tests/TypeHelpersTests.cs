using FluentAssertions;
using Cortex.Column;

namespace Cortex.ML.Tests;

public class TypeHelpersTests
{
    [Fact]
    public void GetDouble_FromDoubleColumn()
    {
        var col = new Column<double>("x", [1.5, 2.5, 3.5]);
        TypeHelpers.GetDouble(col, 0).Should().Be(1.5);
        TypeHelpers.GetDouble(col, 2).Should().Be(3.5);
    }

    [Fact]
    public void GetDouble_FromIntColumn()
    {
        var col = new Column<int>("x", [10, 20, 30]);
        TypeHelpers.GetDouble(col, 0).Should().Be(10.0);
        TypeHelpers.GetDouble(col, 1).Should().Be(20.0);
    }

    [Fact]
    public void GetDouble_FromFloatColumn()
    {
        var col = new Column<float>("x", [1.1f, 2.2f]);
        TypeHelpers.GetDouble(col, 0).Should().BeApproximately(1.1, 0.001);
    }

    [Fact]
    public void GetDouble_FromLongColumn()
    {
        var col = new Column<long>("x", [100L, 200L]);
        TypeHelpers.GetDouble(col, 0).Should().Be(100.0);
    }

    [Fact]
    public void GetDouble_NullReturnsNaN()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, 3.0]);
        TypeHelpers.GetDouble(col, 1).Should().Be(double.NaN);
    }

    [Fact]
    public void GetDoubleArray_FromIntColumn()
    {
        var col = new Column<int>("x", [1, 2, 3]);
        var result = TypeHelpers.GetDoubleArray(col);
        result.Should().Equal([1.0, 2.0, 3.0]);
    }

    [Fact]
    public void GetDoubleArray_FromDoubleColumn_WithNulls()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, 3.0]);
        var result = TypeHelpers.GetDoubleArray(col);
        result[0].Should().Be(1.0);
        double.IsNaN(result[1]).Should().BeTrue();
        result[2].Should().Be(3.0);
    }

    [Fact]
    public void GetDoubleArray_FromFloatColumn()
    {
        var col = new Column<float>("x", [1.5f, 2.5f]);
        var result = TypeHelpers.GetDoubleArray(col);
        result[0].Should().BeApproximately(1.5, 0.001);
        result[1].Should().BeApproximately(2.5, 0.001);
    }

    [Fact]
    public void GetDoubleArray_FromLongColumn()
    {
        var col = new Column<long>("x", [100L, 200L]);
        var result = TypeHelpers.GetDoubleArray(col);
        result.Should().Equal([100.0, 200.0]);
    }
}
