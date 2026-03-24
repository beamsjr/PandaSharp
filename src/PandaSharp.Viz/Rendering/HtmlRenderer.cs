using PandaSharp.Viz.Charts;

namespace PandaSharp.Viz.Rendering;

/// <summary>
/// Renders a ChartSpec as self-contained HTML with embedded Plotly.js.
/// </summary>
public static class HtmlRenderer
{
    private const string PlotlyCdn = "https://cdn.plot.ly/plotly-2.35.2.min.js";

    /// <summary>Generate a complete self-contained HTML page.</summary>
    public static string Render(ChartSpec spec)
    {
        var traces = PlotlySerializer.SerializeTraces(spec);
        var layout = PlotlySerializer.SerializeLayout(spec);
        var plotCall = BuildPlotCall("chart", traces, layout, spec);

        return $$"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{spec.Layout.Title ?? "PandaSharp Chart"}}</title>
            <script src="{{PlotlyCdn}}"></script>
        </head>
        <body>
            <div id="chart"></div>
            <script>
                {{plotCall}}
            </script>
        </body>
        </html>
        """;
    }

    /// <summary>Generate just the div + script for embedding (no full HTML page).</summary>
    public static string RenderFragment(ChartSpec spec, string divId = "chart")
    {
        var traces = PlotlySerializer.SerializeTraces(spec);
        var layout = PlotlySerializer.SerializeLayout(spec);
        var plotCall = BuildPlotCall(divId, traces, layout, spec);

        return $$"""
        <div id="{{divId}}"></div>
        <script>
            {{plotCall}}
        </script>
        """;
    }

    private static string BuildPlotCall(string divId, string traces, string layout, ChartSpec spec)
    {
        if (spec.Frames.Count > 0)
        {
            var frames = PlotlySerializer.SerializeFrames(spec);
            return $"Plotly.newPlot('{divId}', {traces}, {layout}, {{responsive: true}}).then(function(){{ Plotly.addFrames('{divId}', {frames}); }});";
        }
        return $"Plotly.newPlot('{divId}', {traces}, {layout}, {{responsive: true}});";
    }
}
