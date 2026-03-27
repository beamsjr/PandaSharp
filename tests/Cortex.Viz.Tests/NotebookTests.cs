using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Viz;

namespace Cortex.Viz.Tests;

public class NotebookTests
{
    [Fact]
    public void ToNotebookHtml_ContainsPlotlyAndChart()
    {
        var df = new DataFrame(
            new StringColumn("X", ["A", "B"]),
            new Column<double>("Y", [10, 20])
        );

        var html = df.Viz().Bar("X", "Y").ToNotebookHtml();

        html.Should().Contain("plotly");
        html.Should().Contain("Plotly.newPlot");
        html.Should().NotContain("<!DOCTYPE"); // fragment, not full page
    }

    [Fact]
    public void FacetGrid_ToNotebookHtml()
    {
        var df = new DataFrame(
            new StringColumn("G", ["A", "B"]),
            new Column<double>("V", [1, 2])
        );

        var html = df.FacetGrid("G").Histogram("V").ToNotebookHtml();
        html.Should().Contain("plotly");
        html.Should().Contain("facet");
    }
}
