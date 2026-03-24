using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Expressions;

namespace PandaSharp.Tests.Unit.Expressions;

public class EvalTests
{
    private DataFrame CreateTestDF()
    {
        return new DataFrame(
            new Column<int>("id", [1, 2, 3, 4, 5]),
            new Column<double>("price", [10.0, 20.0, 30.0, 40.0, 50.0]),
            new Column<double>("quantity", [5.0, 3.0, 2.0, 1.0, 4.0]),
            new StringColumn("name", ["Alice", "Bob", "Charlie", "Diana", "Eve"])
        );
    }

    // ===== ExprParser =====

    [Fact]
    public void Parse_ColumnReference()
    {
        var expr = ExprParser.Parse("price");
        expr.Should().BeOfType<ColExpr>();
        expr.Name.Should().Be("price");
    }

    [Fact]
    public void Parse_IntLiteral()
    {
        var expr = ExprParser.Parse("42");
        expr.Should().BeOfType<LitExpr<int>>();
    }

    [Fact]
    public void Parse_DoubleLiteral()
    {
        var expr = ExprParser.Parse("3.14");
        expr.Should().BeOfType<LitExpr<double>>();
    }

    [Fact]
    public void Parse_Addition()
    {
        var expr = ExprParser.Parse("price + quantity");
        expr.Should().BeOfType<BinaryExpr>();
    }

    [Fact]
    public void Parse_Multiplication()
    {
        var expr = ExprParser.Parse("price * quantity");
        expr.Should().BeOfType<BinaryExpr>();
    }

    [Fact]
    public void Parse_Comparison()
    {
        var expr = ExprParser.Parse("price > 30");
        expr.Should().BeOfType<ComparisonExpr>();
    }

    [Fact]
    public void Parse_LogicalAnd()
    {
        var expr = ExprParser.Parse("price > 20 and quantity < 4");
        expr.Should().BeOfType<LogicalExpr>();
    }

    [Fact]
    public void Parse_LogicalOr()
    {
        var expr = ExprParser.Parse("price < 15 or price > 45");
        expr.Should().BeOfType<LogicalExpr>();
    }

    [Fact]
    public void Parse_Parentheses()
    {
        var expr = ExprParser.Parse("(price + 10) * quantity");
        expr.Should().BeOfType<BinaryExpr>();
    }

    [Fact]
    public void Parse_Not()
    {
        var expr = ExprParser.Parse("not price > 30");
        expr.Should().BeOfType<NotExpr>();
    }

    [Fact]
    public void Parse_StringLiteral()
    {
        var expr = ExprParser.Parse("name == 'Alice'");
        expr.Should().BeOfType<ComparisonExpr>();
    }

    [Fact]
    public void Parse_ComplexExpression()
    {
        // Should not throw
        var expr = ExprParser.Parse("(price * quantity) >= 100 and quantity > 1");
        expr.Should().NotBeNull();
    }

    [Fact]
    public void Parse_OperatorPrecedence()
    {
        // * binds tighter than +
        var expr = ExprParser.Parse("price + quantity * 2");
        // Should be price + (quantity * 2), not (price + quantity) * 2
        expr.Should().BeOfType<BinaryExpr>();
    }

    // ===== Eval on DataFrame =====

    [Fact]
    public void Eval_FilterByComparison()
    {
        var df = CreateTestDF();
        var result = df.Eval("price > 30");

        result.RowCount.Should().Be(2); // price=40, price=50
    }

    [Fact]
    public void Eval_FilterByCompound()
    {
        var df = CreateTestDF();
        var result = df.Eval("price >= 20 and quantity <= 3");

        result.RowCount.Should().Be(3); // (20,3), (30,2), (40,1)
    }

    [Fact]
    public void Eval_Assignment_AddsColumn()
    {
        var df = CreateTestDF();
        var result = df.Eval("total = price * quantity");

        result.ColumnNames.Should().Contain("total");
        // price=10 * quantity=5 = 50
        result.GetColumn<double>("total")[0].Should().Be(50);
    }

    [Fact]
    public void Eval_ArithmeticExpression()
    {
        var df = CreateTestDF();
        var result = df.Eval("discount = price * 0.9");

        result.GetColumn<double>("discount")[0].Should().BeApproximately(9.0, 0.001);
    }

    [Fact]
    public void Eval_FilterWithParentheses()
    {
        var df = CreateTestDF();
        var result = df.Eval("(price > 15) and (quantity > 2)");

        result.RowCount.Should().Be(2); // (20,3) and (50,4)
    }

    [Fact]
    public void Eval_Subtraction()
    {
        var df = CreateTestDF();
        var result = df.Eval("diff = price - quantity");

        result.GetColumn<double>("diff")[0].Should().Be(5); // 10-5
    }

    [Fact]
    public void Eval_Division()
    {
        var df = CreateTestDF();
        var result = df.Eval("ratio = price / quantity");

        result.GetColumn<double>("ratio")[0].Should().Be(2); // 10/5
    }

    [Fact]
    public void Eval_NegativeNumber()
    {
        var expr = ExprParser.Parse("-5");
        expr.Should().NotBeNull();
    }

    // ===== EvalColumn =====

    [Fact]
    public void EvalColumn_ReturnsColumn()
    {
        var df = CreateTestDF();
        var col = df.EvalColumn("price * quantity");

        col.Should().NotBeNull();
        col.Length.Should().Be(5);
    }

    // ===== Error handling =====

    [Fact]
    public void Parse_InvalidExpression_Throws()
    {
        var act = () => ExprParser.Parse("@@invalid");
        act.Should().Throw<FormatException>();
    }

    [Fact]
    public void Parse_UnmatchedParen_Throws()
    {
        var act = () => ExprParser.Parse("(price + 5");
        act.Should().Throw<FormatException>();
    }
}
