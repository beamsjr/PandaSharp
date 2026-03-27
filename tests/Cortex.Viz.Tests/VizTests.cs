using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Viz;
using Cortex.Viz.Themes;

namespace Cortex.Viz.Tests;

public class VizTests
{
    private static DataFrame SampleDf() => new(
        new StringColumn("Category", ["A", "B", "C", "D"]),
        new Column<double>("Sales", [100, 200, 150, 300]),
        new Column<double>("Profit", [20, 50, 30, 80]),
        new Column<int>("Year", [2021, 2022, 2023, 2024])
    );

    private static DataFrame ScatterDf() => new(
        new Column<double>("X", [1, 2, 3, 4, 5, 6, 7, 8]),
        new Column<double>("Y", [2, 4, 3, 5, 4, 6, 5, 7]),
        new StringColumn("Group", ["A", "A", "B", "B", "A", "A", "B", "B"]),
        new Column<double>("Size", [10, 20, 15, 25, 12, 18, 22, 30])
    );

    // ===== Bar chart =====

    [Fact]
    public void Bar_ProducesValidHtml()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales").ToHtmlString();

        html.Should().Contain("plotly");
        html.Should().Contain("bar");
        html.Should().Contain("Plotly.newPlot");
    }

    [Fact]
    public void Bar_WithTitle()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales")
            .Title("Revenue by Category")
            .ToHtmlString();

        html.Should().Contain("Revenue by Category");
    }

    [Fact]
    public void Bar_Horizontal()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales", horizontal: true).ToHtmlString();
        html.Should().Contain("\"orientation\":\"h\"");
    }

    [Fact]
    public void Bar_GroupedByColor()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "West", "East", "West"]),
            new StringColumn("Product", ["A", "A", "B", "B"]),
            new Column<double>("Sales", [100, 200, 150, 250])
        );

        var html = df.Viz().Bar("Product", "Sales", color: "Region").ToHtmlString();
        html.Should().Contain("East");
        html.Should().Contain("West");
    }

    // ===== Line chart =====

    [Fact]
    public void Line_Basic()
    {
        var html = SampleDf().Viz().Line("Year", "Sales").ToHtmlString();
        html.Should().Contain("scatter");
        html.Should().Contain("lines");
    }

    [Fact]
    public void Line_WithMarkers()
    {
        var html = SampleDf().Viz().Line("Year", "Sales", markers: true).ToHtmlString();
        html.Should().Contain("lines+markers");
    }

    [Fact]
    public void Line_MultiSeries()
    {
        var html = ScatterDf().Viz().Line("X", "Y", color: "Group").ToHtmlString();
        html.Should().Contain("\"name\":\"A\"");
        html.Should().Contain("\"name\":\"B\"");
    }

    // ===== Scatter =====

    [Fact]
    public void Scatter_Basic()
    {
        var html = ScatterDf().Viz().Scatter("X", "Y").ToHtmlString();
        html.Should().Contain("markers");
    }

    [Fact]
    public void Scatter_WithColorAndSize()
    {
        var html = ScatterDf().Viz()
            .Scatter("X", "Y", color: "Group", size: "Size")
            .ToHtmlString();

        html.Should().Contain("\"name\":\"A\"");
        html.Should().Contain("\"size\"");
    }

    [Fact]
    public void Scatter_WebGL_ForLargeData()
    {
        var n = 100;
        var df = new DataFrame(
            new Column<double>("X", Enumerable.Range(0, n).Select(i => (double)i).ToArray()),
            new Column<double>("Y", Enumerable.Range(0, n).Select(i => (double)i * 2).ToArray())
        );

        var html = df.Viz().Scatter("X", "Y", webgl: true).ToHtmlString();
        html.Should().Contain("scattergl");
    }

    // ===== Histogram =====

    [Fact]
    public void Histogram_Basic()
    {
        var html = SampleDf().Viz().Histogram("Sales").ToHtmlString();
        html.Should().Contain("histogram");
    }

    [Fact]
    public void Histogram_WithBins()
    {
        var html = SampleDf().Viz().Histogram("Sales", bins: 20).ToHtmlString();
        html.Should().Contain("\"nbinsx\":20");
    }

    [Fact]
    public void Histogram_Density()
    {
        var html = SampleDf().Viz().Histogram("Sales", density: true).ToHtmlString();
        html.Should().Contain("probability density");
    }

    // ===== Box plot =====

    [Fact]
    public void Box_Basic()
    {
        var html = ScatterDf().Viz().Box(y: "Y").ToHtmlString();
        html.Should().Contain("box");
    }

    [Fact]
    public void Box_Grouped()
    {
        var html = ScatterDf().Viz().Box(y: "Y", color: "Group").ToHtmlString();
        html.Should().Contain("\"name\":\"A\"");
    }

    // ===== Heatmap =====

    [Fact]
    public void Heatmap_Basic()
    {
        var df = new DataFrame(
            new Column<double>("Col1", [1, 2, 3]),
            new Column<double>("Col2", [4, 5, 6]),
            new Column<double>("Col3", [7, 8, 9])
        );

        var html = df.Viz().Heatmap(["Col1", "Col2", "Col3"]).ToHtmlString();
        html.Should().Contain("heatmap");
        html.Should().Contain("\"z\"");
    }

    // ===== Pie =====

    [Fact]
    public void Pie_Basic()
    {
        var html = SampleDf().Viz().Pie("Category", "Sales").ToHtmlString();
        html.Should().Contain("pie");
    }

    // ===== Area =====

    [Fact]
    public void Area_Basic()
    {
        var html = SampleDf().Viz().Area("Year", "Sales").ToHtmlString();
        html.Should().Contain("tozeroy");
    }

    // ===== Customization =====

    [Fact]
    public void Customization_AxisLabels()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales")
            .XLabel("Product Category")
            .YLabel("Revenue ($)")
            .ToHtmlString();

        html.Should().Contain("Product Category");
        html.Should().Contain("Revenue ($)");
    }

    [Fact]
    public void Customization_Size()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales")
            .Size(1200, 600)
            .ToHtmlString();

        html.Should().Contain("\"width\":1200");
        html.Should().Contain("\"height\":600");
    }

    [Fact]
    public void Customization_DarkTheme()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales")
            .Theme(VizTheme.Dark)
            .ToHtmlString();

        html.Should().Contain("plotly_dark");
    }

    [Fact]
    public void Customization_NoLegend()
    {
        var html = SampleDf().Viz().Bar("Category", "Sales")
            .Legend(false)
            .ToHtmlString();

        html.Should().Contain("\"showlegend\":false");
    }

    // ===== Export =====

    [Fact]
    public void ToHtml_WritesFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"viz_test_{Guid.NewGuid():N}.html");
        try
        {
            SampleDf().Viz().Bar("Category", "Sales").ToHtml(path);
            File.Exists(path).Should().BeTrue();
            File.ReadAllText(path).Should().Contain("plotly");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ToHtmlFragment_NoFullPage()
    {
        var fragment = SampleDf().Viz().Bar("Category", "Sales").ToHtmlFragment();
        fragment.Should().Contain("<div id=\"chart\">");
        fragment.Should().Contain("Plotly.newPlot");
        fragment.Should().NotContain("<!DOCTYPE");
    }

    [Fact]
    public void ToHtmlFragment_CustomDivId()
    {
        var fragment = SampleDf().Viz().Bar("Category", "Sales").ToHtmlFragment("myChart");
        fragment.Should().Contain("id=\"myChart\"");
    }

    // ===== Chaining =====

    [Fact]
    public void FullChain_Works()
    {
        var html = SampleDf().Viz()
            .Scatter("Year", "Sales")
            .Title("Sales Over Time")
            .XLabel("Year")
            .YLabel("Sales ($)")
            .Size(1000, 500)
            .Theme(VizTheme.Dark)
            .Legend(true)
            .ToHtmlString();

        html.Should().Contain("Sales Over Time");
        html.Should().Contain("plotly_dark");
        html.Should().Contain("\"width\":1000");
    }

    // ===== Edge cases =====

    [Fact]
    public void EmptyDataFrame_ProducesValidHtml()
    {
        var df = new DataFrame(
            new StringColumn("X", Array.Empty<string?>()),
            new Column<double>("Y", Array.Empty<double>())
        );
        var html = df.Viz().Bar("X", "Y").ToHtmlString();
        html.Should().Contain("Plotly.newPlot");
    }

    [Fact]
    public void SingleRow_Works()
    {
        var df = new DataFrame(
            new StringColumn("X", ["Only"]),
            new Column<double>("Y", [42.0])
        );
        var html = df.Viz().Bar("X", "Y").ToHtmlString();
        html.Should().Contain("42");
    }
}
