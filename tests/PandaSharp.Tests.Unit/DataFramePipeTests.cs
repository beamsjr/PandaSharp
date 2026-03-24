using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit;

public class DataFramePipeTests
{
    [Fact]
    public void Pipe_ChainsTransformations()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        static DataFrame AddSenior(DataFrame df) =>
            df.Filter(df.GetColumn<int>("Age").Gt(28));

        static DataFrame UpperNames(DataFrame df) =>
            df.WithColumn(Col("Name").Str.Upper(), "NameUpper");

        var result = df.Pipe(AddSenior).Pipe(UpperNames);

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("NameUpper");
        result.GetStringColumn("NameUpper")[0].Should().Be("BOB");
    }

    [Fact]
    public void Pipe_WithArgument()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));

        static DataFrame MultiplyColumn(DataFrame df, string colName) =>
            df.WithColumn(Col(colName) * Lit(10), $"{colName}_10x");

        var result = df.Pipe(MultiplyColumn, "X");

        result.ColumnNames.Should().Contain("X_10x");
        result.GetColumn<double>("X_10x")[0].Should().Be(10.0);
    }

    [Fact]
    public void WithColumns_AddsMultipleComputedColumns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<double>("Salary", [50_000, 60_000])
        );

        var result = df.WithColumns(
            (Col("Salary") * Lit(1.1)).Alias("Raised"),
            Col("Name").Str.Upper().Alias("UpperName")
        );

        result.ColumnNames.Should().Contain("Raised");
        result.ColumnNames.Should().Contain("UpperName");
        result.GetStringColumn("UpperName")[0].Should().Be("ALICE");
    }
}
