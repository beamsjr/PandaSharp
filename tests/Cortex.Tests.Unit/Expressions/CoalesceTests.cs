using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Expressions;

public class CoalesceTests
{
    [Fact]
    public void Coalesce_ReturnsFirstNonNull()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [null, 2.0, null]),
            Column<double>.FromNullable("B", [10.0, null, null]),
            new Column<double>("C", [99.0, 99.0, 99.0])
        );

        var result = (Column<double>)Coalesce(Col("A"), Col("B"), Col("C")).Evaluate(df);

        result[0].Should().Be(10.0); // A null → B = 10
        result[1].Should().Be(2.0);  // A = 2
        result[2].Should().Be(99.0); // A, B null → C = 99
    }

    [Fact]
    public void Coalesce_WithLiteral()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [null, 5.0, null])
        );

        var result = (Column<double>)Coalesce(Col("A"), Lit(0.0)).Evaluate(df);

        result[0].Should().Be(0.0);
        result[1].Should().Be(5.0);
        result[2].Should().Be(0.0);
    }

    [Fact]
    public void Coalesce_String()
    {
        var df = new DataFrame(
            new StringColumn("A", [null, "hello", null]),
            new StringColumn("B", ["world", null, null])
        );

        var result = (StringColumn)Coalesce(Col("A"), Col("B"), Lit("default")).Evaluate(df);

        result[0].Should().Be("world");
        result[1].Should().Be("hello");
        result[2].Should().Be("default");
    }

    [Fact]
    public void Coalesce_AllNull_ReturnsNull()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [null]),
            Column<double>.FromNullable("B", [null])
        );

        var result = Coalesce(Col("A"), Col("B")).Evaluate(df);
        result.IsNull(0).Should().BeTrue();
    }

    [Fact]
    public void Coalesce_WithColumn_InDataFrame()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("Score", [null, 85.0, null]),
            new Column<double>("Default", [50.0, 50.0, 50.0])
        );

        var result = df.WithColumn(Coalesce(Col("Score"), Col("Default")), "Final");
        result.GetColumn<double>("Final")[0].Should().Be(50.0);
        result.GetColumn<double>("Final")[1].Should().Be(85.0);
    }
}
