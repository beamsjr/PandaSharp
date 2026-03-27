using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Expressions;

public class ExprEdgeCaseTests
{
    [Fact]
    public void DivideByZero_InExpression_ReturnsNull()
    {
        var df = new DataFrame(
            new Column<double>("A", [10.0, 20.0]),
            new Column<double>("B", [0.0, 5.0])
        );

        var result = (Col("A") / Col("B")).Evaluate(df);
        result.IsNull(0).Should().BeTrue(); // 10/0
        result.GetObject(1).Should().Be(4.0);
    }

    [Fact]
    public void NestedExpressions_DeepChain()
    {
        var df = new DataFrame(new Column<double>("X", [2.0, 3.0]));
        // ((X + 1) * 2) - X = X + 2
        var expr = (Col("X") + Lit(1.0)) * Lit(2.0) - Col("X");
        var result = expr.Evaluate(df);

        result.GetObject(0).Should().Be(4.0); // (2+1)*2 - 2 = 4
        result.GetObject(1).Should().Be(5.0); // (3+1)*2 - 3 = 5
    }

    [Fact]
    public void Expression_AllNulls_Column()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("X", [null, null, null])
        );

        var result = (Col("X") + Lit(1.0)).Evaluate(df);
        for (int i = 0; i < 3; i++)
            result.IsNull(i).Should().BeTrue();
    }

    [Fact]
    public void Expression_SingleRow()
    {
        var df = new DataFrame(new Column<double>("X", [42.0]));
        var result = (Col("X") * Lit(2.0)).Evaluate(df);
        result.Length.Should().Be(1);
        result.GetObject(0).Should().Be(84.0);
    }

    [Fact]
    public void When_AllTrue()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var result = When(Col("X") > Lit(0)).Then(Lit(1.0)).Otherwise(Lit(0.0)).Evaluate(df);
        for (int i = 0; i < 3; i++)
            result.GetObject(i).Should().Be(1.0);
    }

    [Fact]
    public void When_AllFalse()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var result = When(Col("X") > Lit(100)).Then(Lit(1.0)).Otherwise(Lit(0.0)).Evaluate(df);
        for (int i = 0; i < 3; i++)
            result.GetObject(i).Should().Be(0.0);
    }

    [Fact]
    public void Coalesce_SingleExpr()
    {
        var df = new DataFrame(new Column<double>("X", [1.0, 2.0]));
        var result = Coalesce(Col("X")).Evaluate(df);
        result.GetObject(0).Should().Be(1.0);
    }

    [Fact]
    public void Cast_StringToDouble_HandlesNonNumeric()
    {
        var df = new DataFrame(new StringColumn("X", ["1.5", "abc", "3.0"]));
        var result = Col("X").Cast<double>().Evaluate(df);
        // "abc" can't convert → null
        ((Column<double>)result)[0].Should().Be(1.5);
        result.IsNull(1).Should().BeTrue();
        ((Column<double>)result)[2].Should().Be(3.0);
    }

    [Fact]
    public void ConcatStr_EmptyParts()
    {
        var df = new DataFrame(new StringColumn("X", ["hello"]));
        var result = ConcatStr(Col("X")).Evaluate(df);
        ((StringColumn)result)[0].Should().Be("hello");
    }

    [Fact]
    public void Alias_PreservesNameThroughChain()
    {
        var df = new DataFrame(new Column<double>("X", [1.0]));
        var expr = (Col("X") * Lit(2.0)).Alias("DoubleX");
        var result = expr.Evaluate(df);
        result.Name.Should().Be("DoubleX");
    }
}
