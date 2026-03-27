using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using Cortex.Lazy;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Lazy;

public class LazyGroupByTests
{
    private static DataFrame Df() => new(
        new StringColumn("Dept", ["Sales", "Eng", "Sales", "Eng", "Sales"]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
    );

    [Fact]
    public void Lazy_GroupBy_Sum()
    {
        var result = Df().Lazy()
            .GroupBy("Dept")
            .Sum()
            .Collect();

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Dept");
        result.ColumnNames.Should().Contain("Salary");
    }

    [Fact]
    public void Lazy_GroupBy_Mean()
    {
        var result = Df().Lazy()
            .GroupBy("Dept")
            .Mean()
            .Collect();

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Lazy_GroupBy_Count()
    {
        var result = Df().Lazy()
            .GroupBy("Dept")
            .Count()
            .Collect();

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Lazy_Filter_Then_GroupBy()
    {
        var result = Df().Lazy()
            .Filter(Col("Salary") > Lit(55_000))
            .GroupBy("Dept")
            .Sum()
            .Collect();

        // Sales: 75000 + 91000 = 166000
        // Eng: 62000 + 58000 = 120000
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Lazy_GroupBy_Explain()
    {
        var plan = Df().Lazy()
            .Filter(Col("Salary") > Lit(50_000))
            .GroupBy("Dept")
            .Mean()
            .Explain();

        plan.Should().Contain("GroupBy");
        plan.Should().Contain("Filter");
        plan.Should().Contain("Dept");
    }

    [Fact]
    public void StaticConcat_Works()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new Column<int>("A", [3, 4]));

        var result = DataFrame.Concat(df1, df2);
        result.RowCount.Should().Be(4);
        result.GetColumn<int>("A")[2].Should().Be(3);
    }
}
