using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Expressions;

public class DateTimeExprTests
{
    private static DataFrame Df() => new(
        new Column<DateTime>("Date", [
            new DateTime(2024, 1, 15),
            new DateTime(2024, 6, 30),
            new DateTime(2024, 12, 31)
        ]),
        new Column<int>("Value", [10, 20, 30])
    );

    [Fact]
    public void Dt_Year()
    {
        var result = (Column<int>)Col("Date").Dt.Year().Evaluate(Df());
        result[0].Should().Be(2024);
    }

    [Fact]
    public void Dt_Month()
    {
        var result = (Column<int>)Col("Date").Dt.Month().Evaluate(Df());
        result[0].Should().Be(1);
        result[1].Should().Be(6);
        result[2].Should().Be(12);
    }

    [Fact]
    public void Dt_Day()
    {
        var result = (Column<int>)Col("Date").Dt.Day().Evaluate(Df());
        result[0].Should().Be(15);
    }

    [Fact]
    public void Dt_Quarter()
    {
        var result = (Column<int>)Col("Date").Dt.Quarter().Evaluate(Df());
        result[0].Should().Be(1);
        result[1].Should().Be(2);
        result[2].Should().Be(4);
    }

    [Fact]
    public void Dt_DayOfYear()
    {
        var result = (Column<int>)Col("Date").Dt.DayOfYear().Evaluate(Df());
        result[0].Should().Be(15); // Jan 15
    }

    [Fact]
    public void Dt_WithColumn_CreatesYearColumn()
    {
        var df = Df();
        var result = df.WithColumn(Col("Date").Dt.Year(), "Year");
        result.ColumnNames.Should().Contain("Year");
        result.GetColumn<int>("Year")[0].Should().Be(2024);
    }

    [Fact]
    public void Dt_NullPropagation()
    {
        var df = new DataFrame(
            Column<DateTime>.FromNullable("Date", [new DateTime(2024, 1, 1), null])
        );
        var result = (Column<int>)Col("Date").Dt.Month().Evaluate(df);
        result[0].Should().Be(1);
        result[1].Should().BeNull();
    }
}
