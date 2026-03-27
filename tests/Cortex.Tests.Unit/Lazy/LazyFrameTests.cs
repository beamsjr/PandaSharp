using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using Cortex.Lazy;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Lazy;

public class LazyFrameTests
{
    private static DataFrame SampleDf() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        new Column<int>("Age", [25, 30, 35, 28, 42]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
    );

    [Fact]
    public void Lazy_Collect_ReturnsSameData()
    {
        var df = SampleDf();
        var result = df.Lazy().Collect();

        result.RowCount.Should().Be(5);
        result.ColumnNames.Should().Equal(df.ColumnNames);
    }

    [Fact]
    public void Lazy_Select_ProjectsColumns()
    {
        var result = SampleDf().Lazy()
            .Select("Name", "Salary")
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["Name", "Salary"]);
    }

    [Fact]
    public void Lazy_Filter_FiltersRows()
    {
        var result = SampleDf().Lazy()
            .Filter(Col("Age") > Lit(30))
            .Collect();

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Lazy_Sort_SortsRows()
    {
        var result = SampleDf().Lazy()
            .Sort("Salary", ascending: false)
            .Collect();

        result.GetColumn<double>("Salary")[0].Should().Be(91_000);
    }

    [Fact]
    public void Lazy_WithColumn_AddsColumn()
    {
        var result = SampleDf().Lazy()
            .WithColumn(Col("Salary") * Lit(1.1), "NewSalary")
            .Collect();

        result.ColumnNames.Should().Contain("NewSalary");
        result.GetColumn<double>("NewSalary")[0].Should().BeApproximately(55_000, 1);
    }

    [Fact]
    public void Lazy_Head_TakesFirstN()
    {
        var result = SampleDf().Lazy()
            .Head(3)
            .Collect();

        result.RowCount.Should().Be(3);
    }

    [Fact]
    public void Lazy_Chain_MultipleOps()
    {
        var result = SampleDf().Lazy()
            .Filter(Col("Age") > Lit(25))
            .Sort("Salary", ascending: false)
            .Select("Name", "Salary")
            .Head(2)
            .Collect();

        result.RowCount.Should().Be(2);
        result.ColumnCount.Should().Be(2);
        result.GetStringColumn("Name")[0].Should().Be("Eve"); // highest salary
    }

    [Fact]
    public void Lazy_Explain_ReturnsReadablePlan()
    {
        var plan = SampleDf().Lazy()
            .Filter(Col("Age") > Lit(30))
            .Sort("Salary")
            .Explain();

        plan.Should().Contain("Filter");
        plan.Should().Contain("Sort");
        plan.Should().Contain("Scan");
    }

    [Fact]
    public void Lazy_Optimizer_PushesFilterBeforeSort()
    {
        // Filter after Sort should get pushed before Sort in optimized plan
        var lazy = SampleDf().Lazy()
            .Sort("Salary")
            .Filter(Col("Age") > Lit(30));

        var plan = lazy.Explain();

        // After optimization, Sort should appear after Filter in the plan
        int filterPos = plan.IndexOf("Filter");
        int sortPos = plan.IndexOf("Sort");
        sortPos.Should().BeLessThan(filterPos); // Sort is outer (later), Filter is inner (earlier/pushed down)
    }
}
