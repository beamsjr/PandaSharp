using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit.Expressions;

public class StringExprTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "CHARLIE"]),
        new Column<int>("Age", [25, 30, 35])
    );

    [Fact]
    public void Str_Upper()
    {
        var result = Col("Name").Str.Upper().Evaluate(Df());
        ((StringColumn)result)[0].Should().Be("ALICE");
        ((StringColumn)result)[1].Should().Be("BOB");
    }

    [Fact]
    public void Str_Lower()
    {
        var result = Col("Name").Str.Lower().Evaluate(Df());
        ((StringColumn)result)[2].Should().Be("charlie");
    }

    [Fact]
    public void Str_Trim()
    {
        var df = new DataFrame(new StringColumn("S", ["  hi  ", " world "]));
        var result = Col("S").Str.Trim().Evaluate(df);
        ((StringColumn)result)[0].Should().Be("hi");
    }

    [Fact]
    public void Str_Len()
    {
        var result = Col("Name").Str.Len().Evaluate(Df());
        ((Column<int>)result)[0].Should().Be(5); // "Alice"
        ((Column<int>)result)[1].Should().Be(3); // "Bob"
    }

    [Fact]
    public void Str_Contains()
    {
        var result = Col("Name").Str.Contains("li").Evaluate(Df());
        ((Column<bool>)result)[0].Should().Be(true);  // Alice
        ((Column<bool>)result)[1].Should().Be(false);
    }

    [Fact]
    public void Str_StartsWith()
    {
        var result = Col("Name").Str.StartsWith("Al").Evaluate(Df());
        ((Column<bool>)result)[0].Should().Be(true);
        ((Column<bool>)result)[1].Should().Be(false);
    }

    [Fact]
    public void Str_Replace()
    {
        var result = Col("Name").Str.Replace("li", "LI").Evaluate(Df());
        ((StringColumn)result)[0].Should().Be("ALIce");
    }

    [Fact]
    public void Str_Slice()
    {
        var result = Col("Name").Str.Slice(0, 3).Evaluate(Df());
        ((StringColumn)result)[0].Should().Be("Ali");
        ((StringColumn)result)[1].Should().Be("Bob");
    }

    [Fact]
    public void Str_NullPropagation()
    {
        var df = new DataFrame(new StringColumn("S", ["hello", null, "world"]));
        var result = Col("S").Str.Upper().Evaluate(df);
        ((StringColumn)result)[1].Should().BeNull();
    }

    [Fact]
    public void Str_InWithColumn()
    {
        var df = Df();
        var result = df.WithColumn(Col("Name").Str.Upper(), "UpperName");
        result.ColumnNames.Should().Contain("UpperName");
        result.GetStringColumn("UpperName")[0].Should().Be("ALICE");
    }

    [Fact]
    public void Str_InFilter()
    {
        var df = Df();
        var result = df.Filter(Col("Name").Str.Contains("o"));
        result.RowCount.Should().Be(1); // Bob
        result.GetStringColumn("Name")[0].Should().Be("Bob");
    }
}
