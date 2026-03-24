using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit.Expressions;

public class WhenThenTests
{
    [Fact]
    public void WhenThenOtherwise_Numeric()
    {
        var df = new DataFrame(
            new Column<int>("Age", [25, 35, 45, 20])
        );

        var expr = When(Col("Age") > Lit(30))
            .Then(Lit(1.0))
            .Otherwise(Lit(0.0));

        var result = (Column<double>)expr.Evaluate(df);
        result[0].Should().Be(0.0); // 25 <= 30
        result[1].Should().Be(1.0); // 35 > 30
        result[2].Should().Be(1.0); // 45 > 30
        result[3].Should().Be(0.0); // 20 <= 30
    }

    [Fact]
    public void WhenThenOtherwise_String()
    {
        var df = new DataFrame(
            new Column<int>("Score", [90, 70, 50])
        );

        var expr = When(Col("Score") >= Lit(80))
            .Then(Lit("Pass"))
            .Otherwise(Lit("Fail"));

        var result = (StringColumn)expr.Evaluate(df);
        result[0].Should().Be("Pass");
        result[1].Should().Be("Fail");
        result[2].Should().Be("Fail");
    }

    [Fact]
    public void WhenThenOtherwise_WithColumn()
    {
        var df = new DataFrame(
            new Column<double>("Value", [10.0, -5.0, 3.0, -8.0])
        );

        var result = df.WithColumn(
            When(Col("Value") > Lit(0.0))
                .Then(Lit("Positive"))
                .Otherwise(Lit("Negative")),
            "Sign"
        );

        result.ColumnNames.Should().Contain("Sign");
        result.GetStringColumn("Sign")[0].Should().Be("Positive");
        result.GetStringColumn("Sign")[1].Should().Be("Negative");
    }

    [Fact]
    public void WhenThenOtherwise_ColumnValues()
    {
        var df = new DataFrame(
            new Column<double>("A", [10.0, 20.0, 30.0]),
            new Column<double>("B", [15.0, 15.0, 15.0])
        );

        // If A > B then A, else B
        var expr = When(Col("A") > Col("B"))
            .Then(Col("A"))
            .Otherwise(Col("B"));

        var result = (Column<double>)expr.Evaluate(df);
        result[0].Should().Be(15.0); // A=10 < B=15 → B
        result[1].Should().Be(20.0); // A=20 > B=15 → A
        result[2].Should().Be(30.0); // A=30 > B=15 → A
    }

    [Fact]
    public void WhenThenOtherwise_NullCondition()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("X", [10, null, 30])
        );

        var expr = When(Col("X") > Lit(15))
            .Then(Lit(1.0))
            .Otherwise(Lit(0.0));

        var result = (Column<double>)expr.Evaluate(df);
        result[0].Should().Be(0.0);  // 10 <= 15
        result[1].Should().Be(0.0);  // null condition → otherwise
        result[2].Should().Be(1.0);  // 30 > 15
    }
}
