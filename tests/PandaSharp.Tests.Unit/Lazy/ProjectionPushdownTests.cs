using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Lazy;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit.Lazy;

public class ProjectionPushdownTests
{
    private DataFrame CreateWideDF(int rows = 100, int cols = 20)
    {
        var columns = new List<IColumn>();
        for (int c = 0; c < cols; c++)
        {
            var values = new double[rows];
            for (int r = 0; r < rows; r++)
                values[r] = r * (c + 1);
            columns.Add(new Column<double>($"col{c}", values));
        }
        return new DataFrame(columns);
    }

    [Fact]
    public void ProjectionPushdown_SelectThroughSort()
    {
        var df = CreateWideDF();
        var result = df.Lazy()
            .Sort("col0")
            .Select("col0", "col1")
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["col0", "col1"]);
        result.RowCount.Should().Be(100);
    }

    [Fact]
    public void ProjectionPushdown_SelectThroughFilter()
    {
        var df = CreateWideDF();
        var result = df.Lazy()
            .Filter(Col("col5") > Lit(50))
            .Select("col0", "col1")
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["col0", "col1"]);
        result.RowCount.Should().BeLessThan(100);
    }

    [Fact]
    public void ProjectionPushdown_SelectThroughHead()
    {
        var df = CreateWideDF();
        var result = df.Lazy()
            .Head(10)
            .Select("col0", "col1")
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.RowCount.Should().Be(10);
    }

    [Fact]
    public void ProjectionPushdown_ChainedSelectFilterSort()
    {
        var df = CreateWideDF();
        var result = df.Lazy()
            .Filter(Col("col5") > Lit(50))
            .Sort("col3", ascending: false)
            .Select("col0", "col1")
            .Head(5)
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["col0", "col1"]);
        result.RowCount.Should().Be(5);
    }

    [Fact]
    public void ProjectionPushdown_ExplainShowsPushedSelect()
    {
        var df = CreateWideDF();
        var explanation = df.Lazy()
            .Sort("col0")
            .Select("col0", "col1")
            .Explain();

        // The Select should appear below the Sort in the optimized plan
        explanation.Should().Contain("Select");
        explanation.Should().Contain("Sort");
    }

    [Fact]
    public void ProjectionPushdown_SelectOverSelect_Merges()
    {
        var df = CreateWideDF();
        var result = df.Lazy()
            .Select("col0", "col1", "col2")
            .Select("col0", "col1")
            .Collect();

        result.ColumnCount.Should().Be(2);
    }

    [Fact]
    public void ProjectionPushdown_FilterColumnIncludedThenPruned()
    {
        // Filter on col5, but only select col0, col1
        // The optimizer should include col5 for the filter, then prune it
        var df = CreateWideDF();
        var result = df.Lazy()
            .Filter(Col("col5") > Lit(200))
            .Select("col0", "col1")
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().NotContain("col5"); // pruned after filter
    }

    [Fact]
    public void ProjectionPushdown_SortColumnIncludedThenPruned()
    {
        // Sort by col3, but only select col0
        var df = CreateWideDF();
        var result = df.Lazy()
            .Sort("col3")
            .Select("col0")
            .Collect();

        result.ColumnCount.Should().Be(1);
        result.ColumnNames.Should().Equal(["col0"]);
        result.ColumnNames.Should().NotContain("col3"); // pruned after sort
    }

    [Fact]
    public void ProjectionPushdown_PreservesCorrectness_LargeChain()
    {
        var df = CreateWideDF(1000, 50);

        // Complex chain: filter on col10, sort by col20, select col0 and col1, head 100
        var sequential = df
            .Filter(row => (double)row["col10"]! > 500)
            .Sort("col20")
            .Select("col0", "col1")
            .Head(100);

        var lazy = df.Lazy()
            .Filter(Col("col10") > Lit(500))
            .Sort("col20")
            .Select("col0", "col1")
            .Head(100)
            .Collect();

        lazy.RowCount.Should().Be(sequential.RowCount);
        lazy.ColumnCount.Should().Be(2);

        // Verify values match
        for (int i = 0; i < lazy.RowCount; i++)
        {
            lazy.GetColumn<double>("col0")[i]
                .Should().Be(sequential.GetColumn<double>("col0")[i]);
        }
    }
}
