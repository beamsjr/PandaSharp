using Cortex.Viz.Charts;
using Cortex.Viz.Themes;

namespace Cortex.Viz.Rendering;

/// <summary>
/// Renders a ChartSpec as self-contained HTML with embedded D3.js.
/// Drop-in replacement for HtmlRenderer (Plotly-based).
/// </summary>
public static class D3HtmlRenderer
{
    private const string D3Cdn = "https://cdn.jsdelivr.net/npm/d3@7";

    /// <summary>Generate a complete self-contained HTML page.</summary>
    public static string Render(ChartSpec spec)
    {
        var script = D3Renderer.Render(spec, "chart");
        var themeClass = spec.Layout.Template ?? VizTheme.Default;

        return $$"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{{spec.Layout.Title ?? "Cortex Chart"}}</title>
            <script src="{{D3Cdn}}"></script>
            <style>
                {{VizTheme.GetCss()}}
                body { margin: 0; display: flex; justify-content: center; padding: 20px; background: var(--chart-bg, #fafafa); color: var(--text-color, #333); }
                #chart { background: var(--chart-bg, #fff); border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
                #chart text { fill: var(--text-color, #333); }
                #chart line, #chart .domain { stroke: var(--axis-color, #333); }
            </style>
        </head>
        <body class="{{themeClass}}">
            <div id="chart"></div>
            <script>
                {{script}}
            </script>
        </body>
        </html>
        """;
    }

    /// <summary>Generate just the div + script for embedding (no full HTML page).</summary>
    public static string RenderFragment(ChartSpec spec, string divId = "chart")
    {
        var script = D3Renderer.Render(spec, divId);

        return $$"""
        <div id="{{divId}}"></div>
        <script>
            {{script}}
        </script>
        """;
    }

    /// <summary>The D3 CDN URL, for use by other renderers that need to include it in their HTML head.</summary>
    public static string CdnUrl => D3Cdn;

    /// <summary>Theme CSS for embedding in composite pages (SubplotBuilder, StoryBoard).</summary>
    public static string ThemeCss => VizTheme.GetCss();
}
