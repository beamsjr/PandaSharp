using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Expressions;

public class ExprLogicTests
{
    private static DataFrame Df() => new(
        new Column<int>("Age", [25, 30, 35, 40]),
        new Column<double>("Salary", [50_000, 60_000, 70_000, 80_000])
    );

    [Fact]
    public void And_CombinesTwoPredicates()
    {
        var df = Df();
        var expr = (Col("Age") > Lit(28)) & (Col("Salary") < Lit(75_000));
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(false); // 25 > 28 = false
        result[1].Should().Be(true);  // 30 > 28 AND 60k < 75k
        result[2].Should().Be(true);  // 35 > 28 AND 70k < 75k
        result[3].Should().Be(false); // 80k < 75k = false
    }

    [Fact]
    public void Or_CombinesTwoPredicates()
    {
        var df = Df();
        var expr = (Col("Age") < Lit(28)) | (Col("Salary") > Lit(75_000));
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(true);  // 25 < 28
        result[1].Should().Be(false);
        result[2].Should().Be(false);
        result[3].Should().Be(true);  // 80k > 75k
    }

    [Fact]
    public void Not_InvertsPredicate()
    {
        var df = Df();
        var expr = !(Col("Age") > Lit(30));
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(true);  // !(25>30) = true
        result[1].Should().Be(true);  // !(30>30) = true
        result[2].Should().Be(false); // !(35>30) = false
    }

    [Fact]
    public void Eq_ComparesEquality()
    {
        var df = Df();
        var expr = Col("Age").Eq(Lit(30));
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(false);
        result[1].Should().Be(true);
        result[2].Should().Be(false);
    }

    [Fact]
    public void Neq_ComparesInequality()
    {
        var df = Df();
        var expr = Col("Age").Neq(Lit(30));
        var result = (Column<bool>)expr.Evaluate(df);

        result[0].Should().Be(true);
        result[1].Should().Be(false);
        result[2].Should().Be(true);
    }

    [Fact]
    public void Filter_WithLogicalExpression()
    {
        var df = Df();
        var result = df.Filter((Col("Age") > Lit(28)) & (Col("Salary") < Lit(75_000)));

        result.RowCount.Should().Be(2); // Age 30 + 35
    }

    [Fact]
    public void ComplexLogic_ThreeConditions()
    {
        var df = Df();
        // Age > 25 AND (Salary < 65000 OR Salary > 75000)
        var expr = (Col("Age") > Lit(25)) & ((Col("Salary") < Lit(65_000)) | (Col("Salary") > Lit(75_000)));
        var result = df.Filter(expr);

        result.RowCount.Should().Be(2); // Age 30 (60k<65k) and Age 40 (80k>75k)
    }
}
