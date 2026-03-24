using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using PandaSharp.Lazy;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit.Lazy;

public class LazyWithColumnsTests
{
    [Fact]
    public void WithColumns_MultipleComputedColumns()
    {
        var df = new DataFrame(
            new Column<double>("Price", [10.0, 20.0, 30.0]),
            new Column<int>("Qty", [5, 3, 8])
        );

        var result = df.Lazy()
            .WithColumns(
                (Col("Price") * Lit(1.1), "PriceUp"),
                (Col("Price") * Col("Qty"), "Total")
            )
            .Collect();

        result.ColumnNames.Should().Contain("PriceUp");
        result.ColumnNames.Should().Contain("Total");
        result.GetColumn<double>("Total")[0].Should().Be(50.0);
    }

    [Fact]
    public void WithColumns_ThenFilter()
    {
        var df = new DataFrame(
            new Column<double>("Value", [10.0, 50.0, 90.0])
        );

        var result = df.Lazy()
            .WithColumns(
                (Col("Value") * Lit(2.0), "Doubled")
            )
            .Filter(Col("Doubled") > Lit(50.0))
            .Collect();

        result.RowCount.Should().Be(2); // 100 and 180
    }
}
