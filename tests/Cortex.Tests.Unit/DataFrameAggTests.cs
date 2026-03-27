using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameAggTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
        new Column<int>("Age", [25, 30, 35]),
        new Column<double>("Salary", [50_000, 60_000, 70_000])
    );

    [Fact]
    public void Agg_Sum()
    {
        var result = Df().Agg(vals => vals.Sum());
        result.RowCount.Should().Be(1);
        result.GetColumn<double>("Age")[0].Should().Be(90);
        result.GetColumn<double>("Salary")[0].Should().Be(180_000);
        result.ColumnNames.Should().NotContain("Name"); // non-numeric excluded
    }

    [Fact]
    public void Agg_Average()
    {
        var result = Df().Agg(vals => vals.Average());
        result.GetColumn<double>("Age")[0].Should().Be(30);
        result.GetColumn<double>("Salary")[0].Should().Be(60_000);
    }

    [Fact]
    public void Agg_Max()
    {
        var result = Df().Agg(vals => vals.Max());
        result.GetColumn<double>("Age")[0].Should().Be(35);
    }

    [Fact]
    public void Agg_Count()
    {
        var result = Df().Agg(vals => vals.Count());
        result.GetColumn<double>("Age")[0].Should().Be(3);
    }

    [Fact]
    public void IsEmpty_True()
    {
        new DataFrame().IsEmpty.Should().BeTrue();
        new DataFrame(new Column<int>("A", Array.Empty<int>())).IsEmpty.Should().BeTrue();
    }

    [Fact]
    public void IsEmpty_False()
    {
        Df().IsEmpty.Should().BeFalse();
    }

    [Fact]
    public void Clip_DataFrame()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 50.0, 100.0]),
            new Column<int>("B", [5, 50, 95]),
            new StringColumn("C", ["x", "y", "z"]) // unchanged
        );

        var result = df.Clip(10, 90);

        result.GetColumn<double>("A")[0].Should().Be(10);
        result.GetColumn<double>("A")[1].Should().Be(50);
        result.GetColumn<double>("A")[2].Should().Be(90);
        result.GetColumn<int>("B")[0].Should().Be(10);
        result.GetStringColumn("C")[0].Should().Be("x"); // untouched
    }

    [Fact]
    public void ApplyColumns_TransformsEachColumn()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new Column<int>("B", [4, 5, 6])
        );

        var result = df.ApplyColumns(col => col.Clone($"{col.Name}_copy"));

        result.ColumnNames.Should().Equal(["A_copy", "B_copy"]);
    }
}
