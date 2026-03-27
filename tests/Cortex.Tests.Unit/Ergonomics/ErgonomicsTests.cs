using Cortex;
using Cortex.Column;
using FluentAssertions;

namespace Cortex.Tests.Unit.Ergonomics;

public class ErgonomicsTests
{
    // ─── Feature 1: Boolean indexing ───

    [Fact]
    public void BooleanIndexing_WithBoolArray_FiltersRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4, 5 },
            ["B"] = new[] { 10, 20, 30, 40, 50 }
        });

        // df["A"].Gt(2) returns bool[]
        var mask = df.GetColumn<int>("A").Gt(2);
        var result = df[mask];

        result.RowCount.Should().Be(3);
        result.GetColumn<int>("A").Values.ToArray().Should().Equal(3, 4, 5);
        result.GetColumn<int>("B").Values.ToArray().Should().Equal(30, 40, 50);
    }

    [Fact]
    public void BooleanIndexing_WithColumnBool_FiltersRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4 },
            ["keep"] = new[] { true, false, true, false }
        });

        var maskCol = df.GetColumn<bool>("keep");
        var result = df[maskCol];

        result.RowCount.Should().Be(2);
        result.GetColumn<int>("A").Values.ToArray().Should().Equal(1, 3);
    }

    [Fact]
    public void BooleanIndexing_CompoundFilter()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["age"] = new[] { 25, 35, 45, 20, 30 },
            ["salary"] = new[] { 50000, 80000, 90000, 30000, 60000 }
        });

        // df[df["age"].Gt(25) & df["salary"].Lt(85000)]
        var mask = df.GetColumn<int>("age").Gt(25)
            .And(df.GetColumn<int>("salary").Lt(85000));
        var result = df[mask];

        result.RowCount.Should().Be(2);
        result.GetColumn<int>("age").Values.ToArray().Should().Equal(35, 30);
    }

    // ─── Feature 2: Arithmetic operators ───

    [Fact]
    public void ArithmeticOperators_ColumnPlusColumn()
    {
        var a = new Column<int>("a", [1, 2, 3]);
        var b = new Column<int>("b", [10, 20, 30]);

        var result = a + b;
        result.Values.ToArray().Should().Equal(11, 22, 33);
    }

    [Fact]
    public void ArithmeticOperators_ColumnMinusColumn()
    {
        var a = new Column<int>("a", [10, 20, 30]);
        var b = new Column<int>("b", [1, 2, 3]);

        var result = a - b;
        result.Values.ToArray().Should().Equal(9, 18, 27);
    }

    [Fact]
    public void ArithmeticOperators_ColumnTimesColumn()
    {
        var a = new Column<int>("a", [2, 3, 4]);
        var b = new Column<int>("b", [5, 6, 7]);

        var result = a * b;
        result.Values.ToArray().Should().Equal(10, 18, 28);
    }

    [Fact]
    public void ArithmeticOperators_ColumnDivideColumn()
    {
        var a = new Column<double>("a", [10.0, 20.0, 30.0]);
        var b = new Column<double>("b", [2.0, 4.0, 5.0]);

        var result = a / b;
        result.Values.ToArray().Should().Equal(5.0, 5.0, 6.0);
    }

    [Fact]
    public void ArithmeticOperators_ColumnTimesScalar()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);

        var result = col * 2.0;
        result.Values.ToArray().Should().Equal(2.0, 4.0, 6.0);
    }

    [Fact]
    public void ArithmeticOperators_ScalarTimesColumn()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);

        var result = 3.0 * col;
        result.Values.ToArray().Should().Equal(3.0, 6.0, 9.0);
    }

    [Fact]
    public void ArithmeticOperators_ColumnPlusScalar()
    {
        var col = new Column<int>("x", [1, 2, 3]);

        var result = col + 10;
        result.Values.ToArray().Should().Equal(11, 12, 13);
    }

    [Fact]
    public void ArithmeticOperators_UnaryNegate()
    {
        var col = new Column<int>("x", [1, -2, 3]);

        var result = -col;
        result.Values.ToArray().Should().Equal(-1, 2, -3);
    }

    // ─── Feature 3: AsDouble() and mixed-type arithmetic ───

    [Fact]
    public void AsDouble_FromInt()
    {
        var col = new Column<int>("x", [1, 2, 3]);
        var result = col.AsDouble();

        result.DataType.Should().Be(typeof(double));
        result.Values.ToArray().Should().Equal(1.0, 2.0, 3.0);
        result.Name.Should().Be("x");
    }

    [Fact]
    public void AsDouble_FromFloat()
    {
        var col = new Column<float>("x", [1.5f, 2.5f, 3.5f]);
        var result = col.AsDouble();

        result.DataType.Should().Be(typeof(double));
        result.Values.ToArray().Should().Equal(1.5, 2.5, 3.5);
    }

    [Fact]
    public void AsDouble_FromLong()
    {
        var col = new Column<long>("x", [100L, 200L, 300L]);
        var result = col.AsDouble();

        result.Values.ToArray().Should().Equal(100.0, 200.0, 300.0);
    }

    [Fact]
    public void MixedType_IntPlusDouble()
    {
        var intCol = new Column<int>("a", [1, 2, 3]);
        var dblCol = new Column<double>("b", [0.5, 1.5, 2.5]);

        var result = intCol.Add(dblCol);

        result.DataType.Should().Be(typeof(double));
        result.Values.ToArray().Should().Equal(1.5, 3.5, 5.5);
    }

    [Fact]
    public void MixedType_DoublePlusInt()
    {
        var dblCol = new Column<double>("a", [10.0, 20.0, 30.0]);
        var intCol = new Column<int>("b", [1, 2, 3]);

        var result = dblCol.Add(intCol);

        result.DataType.Should().Be(typeof(double));
        result.Values.ToArray().Should().Equal(11.0, 22.0, 33.0);
    }

    [Fact]
    public void MixedType_IntTimesDouble()
    {
        var intCol = new Column<int>("a", [2, 3, 4]);
        var dblCol = new Column<double>("b", [1.5, 2.5, 3.5]);

        var result = intCol.Multiply(dblCol);

        result.Values.ToArray().Should().Equal(3.0, 7.5, 14.0);
    }

    [Fact]
    public void MixedType_FloatPlusDouble()
    {
        var fltCol = new Column<float>("a", [1.0f, 2.0f, 3.0f]);
        var dblCol = new Column<double>("b", [0.1, 0.2, 0.3]);

        var result = fltCol.Add(dblCol);

        result.DataType.Should().Be(typeof(double));
        result.Length.Should().Be(3);
    }

}
