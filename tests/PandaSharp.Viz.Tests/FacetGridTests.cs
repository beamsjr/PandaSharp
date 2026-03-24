using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Viz;
using PandaSharp.Viz.Charts;

namespace PandaSharp.Viz.Tests;

public class FacetGridTests
{
    [Fact]
    public void FacetGrid_ProducesOneChartPerGroup()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "West", "East", "West"]),
            new Column<double>("Month", [1, 1, 2, 2]),
            new Column<double>("Revenue", [100, 200, 150, 250])
        );

        var html = df.FacetGrid("Region").Line("Month", "Revenue").ToHtmlString();

        html.Should().Contain("Region = East");
        html.Should().Contain("Region = West");
        html.Should().Contain("facet_0");
        html.Should().Contain("facet_1");
    }

    [Fact]
    public void FacetGrid_Bar()
    {
        var df = new DataFrame(
            new StringColumn("Type", ["A", "B", "A", "B"]),
            new StringColumn("Item", ["X", "X", "Y", "Y"]),
            new Column<double>("Count", [10, 20, 30, 40])
        );

        var html = df.FacetGrid("Type").Bar("Item", "Count").ToHtmlString();
        html.Should().Contain("Type = A");
        html.Should().Contain("bar");
    }

    [Fact]
    public void FacetGrid_WithTitle()
    {
        var df = new DataFrame(
            new StringColumn("G", ["A", "B"]),
            new Column<double>("V", [1, 2])
        );

        var html = df.FacetGrid("G").Histogram("V").Title("My Facets").ToHtmlString();
        html.Should().Contain("My Facets");
    }

    [Fact]
    public void Subplots_CombinesCharts()
    {
        var df = new DataFrame(
            new StringColumn("Cat", ["A", "B", "C"]),
            new Column<double>("Val", [10, 20, 30])
        );

        var chart1 = df.Viz().Bar("Cat", "Val").Title("Chart 1");
        var chart2 = df.Viz().Scatter("Val", "Val").Title("Chart 2");

        var html = SubplotBuilder.Grid(chart1, chart2).Cols(2).Title("Dashboard").ToHtmlString();

        html.Should().Contain("subplot_0");
        html.Should().Contain("subplot_1");
        html.Should().Contain("Dashboard");
    }
}
