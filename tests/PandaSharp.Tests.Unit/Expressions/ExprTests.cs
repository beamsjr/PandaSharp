using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit.Expressions;

public class ExprTests
{
    private static DataFrame SampleDf() => new(
        new Column<double>("Price", [10.0, 20.0, 30.0]),
        new Column<int>("Qty", [5, 3, 8]),
        new StringColumn("Name", ["A", "B", "C"])
    );

    [Fact]
    public void Col_ReturnsColumn()
    {
        var df = SampleDf();
        var result = Col("Price").Evaluate(df);
        result.GetObject(0).Should().Be(10.0);
    }

    [Fact]
    public void Lit_BroadcastsScalar()
    {
        var df = SampleDf();
        var result = Lit(42).Evaluate(df);
        result.Length.Should().Be(3);
        result.GetObject(0).Should().Be(42);
    }

    [Fact]
    public void Add_TwoColumns()
    {
        var df = SampleDf();
        var expr = Col("Price") + Col("Qty");
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(15.0);
        result.GetObject(1).Should().Be(23.0);
    }

    [Fact]
    public void Multiply_ColumnByColumn()
    {
        var df = SampleDf();
        var expr = Col("Price") * Col("Qty");
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(50.0);
        result.GetObject(1).Should().Be(60.0);
        result.GetObject(2).Should().Be(240.0);
    }

    [Fact]
    public void Multiply_ColumnByScalar()
    {
        var df = SampleDf();
        var expr = Col("Price") * Lit(2.0);
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(20.0);
    }

    [Fact]
    public void Divide_Columns()
    {
        var df = SampleDf();
        var expr = Col("Price") / Col("Qty");
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(2.0);
    }

    [Fact]
    public void Subtract_Columns()
    {
        var df = SampleDf();
        var expr = Col("Price") - Lit(5.0);
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(5.0);
    }

    [Fact]
    public void GreaterThan_ReturnsBoolean()
    {
        var df = SampleDf();
        var expr = Col("Price") > Lit(15.0);
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(false); // 10 > 15
        result[1].Should().Be(true);  // 20 > 15
        result[2].Should().Be(true);  // 30 > 15
    }

    [Fact]
    public void LessThan_ReturnsBoolean()
    {
        var df = SampleDf();
        var expr = Col("Price") < Lit(25.0);
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(true);
        result[1].Should().Be(true);
        result[2].Should().Be(false);
    }

    [Fact]
    public void Alias_RenamesOutput()
    {
        var df = SampleDf();
        var expr = (Col("Price") * Col("Qty")).Alias("Total");
        var result = expr.Evaluate(df);

        result.Name.Should().Be("Total");
    }

    [Fact]
    public void WithColumn_AddsComputedColumn()
    {
        var df = SampleDf();
        var result = df.WithColumn(Col("Price") * Col("Qty"), "Total");

        result.ColumnNames.Should().Contain("Total");
        result.GetColumn<double>("Total")[0].Should().Be(50.0);
    }

    [Fact]
    public void Filter_WithExpression()
    {
        var df = SampleDf();
        var result = df.Filter(Col("Price") > Lit(15.0));

        result.RowCount.Should().Be(2);
        result.GetColumn<double>("Price")[0].Should().Be(20.0);
    }

    [Fact]
    public void ComplexExpression_Composes()
    {
        var df = SampleDf();
        // (Price * Qty) - 10
        var expr = Col("Price") * Col("Qty") - Lit(10.0);
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(40.0);  // 10*5 - 10
        result.GetObject(1).Should().Be(50.0);  // 20*3 - 10
        result.GetObject(2).Should().Be(230.0); // 30*8 - 10
    }

    [Fact]
    public void NullPropagation_InArithmetic()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [1.0, null, 3.0]),
            new Column<double>("B", [10.0, 20.0, 30.0])
        );

        var result = (Col("A") + Col("B")).Evaluate(df);

        result.GetObject(0).Should().Be(11.0);
        result.IsNull(1).Should().BeTrue();
        result.GetObject(2).Should().Be(33.0);
    }
}
