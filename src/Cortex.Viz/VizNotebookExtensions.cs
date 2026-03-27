namespace Cortex.Viz;

/// <summary>
/// .NET Interactive / Polyglot Notebook integration for Cortex.Viz.
/// When displayed in a notebook, VizBuilder renders as interactive HTML.
/// </summary>
public static class VizNotebookExtensions
{
    /// <summary>
    /// Get the HTML content suitable for notebook display.
    /// In .NET Interactive, implement IHtmlContent or use display().
    /// </summary>
    public static string ToNotebookHtml(this VizBuilder viz)
    {
        // Return just the fragment — notebooks handle the Plotly CDN via RequireJS
        return $"""
        <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
        {viz.ToHtmlFragment($"viz_{Guid.NewGuid():N}")}
        """;
    }

    /// <summary>
    /// Get the HTML for a FacetGrid in notebook context.
    /// </summary>
    public static string ToNotebookHtml(this Charts.FacetGridBuilder facet)
    {
        // Strip the full HTML wrapper, return just the body content
        var full = facet.ToHtmlString();
        int bodyStart = full.IndexOf("<body>") + 6;
        int bodyEnd = full.IndexOf("</body>");
        if (bodyStart > 6 && bodyEnd > bodyStart)
        {
            return $"""
            <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
            {full[bodyStart..bodyEnd]}
            """;
        }
        return full;
    }
}
