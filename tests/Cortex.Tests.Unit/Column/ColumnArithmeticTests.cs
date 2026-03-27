using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ColumnArithmeticTests
{
    [Fact]
    public void Add_TwoColumns()
    {
        var a = new Column<int>("A", [1, 2, 3]);
        var b = new Column<int>("B", [10, 20, 30]);
        var result = a.Add(b);
        result[0].Should().Be(11);
        result[1].Should().Be(22);
        result[2].Should().Be(33);
    }

    [Fact]
    public void Subtract_TwoColumns()
    {
        var a = new Column<double>("A", [10.0, 20.0]);
        var b = new Column<double>("B", [3.0, 5.0]);
        var result = a.Subtract(b);
        result[0].Should().Be(7.0);
        result[1].Should().Be(15.0);
    }

    [Fact]
    public void Multiply_TwoColumns()
    {
        var a = new Column<int>("A", [2, 3]);
        var b = new Column<int>("B", [4, 5]);
        var result = a.Multiply(b);
        result[0].Should().Be(8);
        result[1].Should().Be(15);
    }

    [Fact]
    public void Divide_TwoColumns()
    {
        var a = new Column<double>("A", [10.0, 20.0]);
        var b = new Column<double>("B", [2.0, 5.0]);
        var result = a.Divide(b);
        result[0].Should().Be(5.0);
        result[1].Should().Be(4.0);
    }

    [Fact]
    public void Divide_ByZero_ReturnsNull()
    {
        var a = new Column<int>("A", [10, 20]);
        var b = new Column<int>("B", [0, 5]);
        var result = a.Divide(b);
        result[0].Should().BeNull();
        result[1].Should().Be(4);
    }

    [Fact]
    public void Add_Scalar()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.Add(10);
        result[0].Should().Be(11);
        result[2].Should().Be(13);
    }

    [Fact]
    public void Multiply_Scalar()
    {
        var col = new Column<double>("X", [1.5, 2.5]);
        var result = col.Multiply(2.0);
        result[0].Should().Be(3.0);
        result[1].Should().Be(5.0);
    }

    [Fact]
    public void Divide_Scalar()
    {
        var col = new Column<int>("X", [10, 20, 30]);
        var result = col.Divide(10);
        result[0].Should().Be(1);
        result[2].Should().Be(3);
    }

    [Fact]
    public void Divide_ScalarZero_Throws()
    {
        var col = new Column<int>("X", [1]);
        var act = () => col.Divide(0);
        act.Should().Throw<DivideByZeroException>();
    }

    [Fact]
    public void Negate_Column()
    {
        var col = new Column<int>("X", [1, -2, 3]);
        var result = col.Negate();
        result[0].Should().Be(-1);
        result[1].Should().Be(2);
        result[2].Should().Be(-3);
    }

    [Fact]
    public void NullPropagation_InArithmetic()
    {
        var a = Column<int>.FromNullable("A", [1, null, 3]);
        var b = new Column<int>("B", [10, 20, 30]);
        var result = a.Add(b);
        result[0].Should().Be(11);
        result[1].Should().BeNull();
        result[2].Should().Be(33);
    }

    [Fact]
    public void Rename_Column()
    {
        var col = new Column<int>("Old", [1, 2]);
        var renamed = col.Rename("New");
        renamed.Name.Should().Be("New");
        renamed[0].Should().Be(1);
    }

    [Fact]
    public void Rename_StringColumn()
    {
        var col = new StringColumn("Old", ["a", "b"]);
        var renamed = col.Rename("New");
        renamed.Name.Should().Be("New");
        renamed[0].Should().Be("a");
    }

    [Fact]
    public void LengthMismatch_Throws()
    {
        var a = new Column<int>("A", [1, 2]);
        var b = new Column<int>("B", [1, 2, 3]);
        var act = () => a.Add(b);
        act.Should().Throw<ArgumentException>();
    }
}
