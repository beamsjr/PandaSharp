using System.Text;
using System.Text.RegularExpressions;
using PandaSharp;
using PandaSharp.Viz.Charts;

namespace PandaSharp.Viz.Rendering;

/// <summary>
/// Renders a StoryBoard as a single-page narrative HTML document.
/// </summary>
public static class StoryBoardRenderer
{
    private static int _chartCounter;

    public static string Render(List<StorySection> sections, string? title, string? author, StoryTheme theme)
    {
        _chartCounter = 0;
        var sb = new StringBuilder();
        var isDark = theme == StoryTheme.Dark;

        // HTML head
        sb.AppendLine("<!DOCTYPE html>");
        sb.AppendLine("<html lang='en'>");
        sb.AppendLine("<head>");
        sb.AppendLine("<meta charset='utf-8'>");
        sb.AppendLine("<meta name='viewport' content='width=device-width, initial-scale=1'>");
        sb.AppendLine($"<title>{Escape(title ?? "PandaSharp Report")}</title>");
        sb.AppendLine("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>");
        sb.AppendLine("<style>");
        sb.AppendLine(GetCss());
        sb.AppendLine("</style>");
        sb.AppendLine("</head>");
        sb.AppendLine($"<body class='{(isDark ? "story-dark" : "story-light")}'>");
        sb.AppendLine("<div class='story-container'>");

        // Author/date meta
        if (author is not null)
        {
            sb.AppendLine($"<div class='story-meta'>{Escape(author)} &middot; {DateTime.Now:MMMM d, yyyy}</div>");
        }

        // Render each section
        foreach (var section in sections)
            RenderSection(sb, section, isDark);

        // Footer
        sb.AppendLine("<footer class='story-footer'>Generated with PandaSharp</footer>");
        sb.AppendLine("</div>");
        sb.AppendLine("</body></html>");

        return sb.ToString();
    }

    private static void RenderSection(StringBuilder sb, StorySection section, bool isDark)
    {
        switch (section)
        {
            case TitleSection t:
                var tag = t.Level switch { 1 => "h1", 2 => "h2", _ => "h3" };
                var cls = t.Level == 1 ? " class='story-title'" : "";
                sb.AppendLine($"<{tag}{cls}>{Escape(t.Text)}</{tag}>");
                break;

            case TextSection t:
                foreach (var para in t.Content.Split("\n\n", StringSplitOptions.RemoveEmptyEntries))
                    sb.AppendLine($"<p>{FormatInlineMarkup(para.Trim())}</p>");
                break;

            case ChartSection c:
                var divId = $"story_chart_{_chartCounter++}";
                // Apply dark theme to chart if needed
                if (isDark && string.IsNullOrEmpty(c.Spec.Layout.Template))
                    c.Spec.Layout.Template = "plotly_dark";
                // Override layout to be responsive
                c.Spec.Layout.Extra["responsive"] = true;
                var fragment = HtmlRenderer.RenderFragment(c.Spec, divId);
                sb.AppendLine($"<div class='story-chart'>{fragment}</div>");
                if (c.Caption is not null)
                    sb.AppendLine($"<p class='chart-caption'>{Escape(c.Caption)}</p>");
                break;

            case StatsSection s:
                sb.AppendLine("<div class='stats-row'>");
                foreach (var (label, value) in s.Items)
                {
                    sb.AppendLine("<div class='stat-card'>");
                    sb.AppendLine($"<div class='stat-value'>{Escape(value)}</div>");
                    sb.AppendLine($"<div class='stat-label'>{Escape(label)}</div>");
                    sb.AppendLine("</div>");
                }
                sb.AppendLine("</div>");
                break;

            case TableSection t:
                if (t.Caption is not null)
                    sb.AppendLine($"<p class='table-caption'>{Escape(t.Caption)}</p>");
                sb.AppendLine("<div class='story-table'>");
                RenderTable(sb, t.Data, t.MaxRows);
                sb.AppendLine("</div>");
                break;

            case CalloutSection c:
                var calloutClass = c.Style switch
                {
                    CalloutStyle.Warning => "callout-warning",
                    CalloutStyle.Success => "callout-success",
                    CalloutStyle.Note => "callout-note",
                    _ => "callout-info"
                };
                sb.AppendLine($"<div class='callout {calloutClass}'>{FormatInlineMarkup(c.Content)}</div>");
                break;

            case DividerSection:
                sb.AppendLine("<hr class='story-divider'>");
                break;

            case RowSection r:
                sb.AppendLine("<div class='story-row'>");
                foreach (var child in r.Children)
                {
                    sb.AppendLine("<div class='row-cell'>");
                    RenderSection(sb, child, isDark);
                    sb.AppendLine("</div>");
                }
                sb.AppendLine("</div>");
                break;
        }
    }

    private static void RenderTable(StringBuilder sb, DataFrame df, int maxRows)
    {
        int rows = Math.Min(df.RowCount, maxRows);
        sb.AppendLine("<table>");
        sb.AppendLine("<thead><tr>");
        foreach (var name in df.ColumnNames)
            sb.AppendLine($"<th>{Escape(name)}</th>");
        sb.AppendLine("</tr></thead>");
        sb.AppendLine("<tbody>");
        for (int r = 0; r < rows; r++)
        {
            sb.AppendLine("<tr>");
            foreach (var name in df.ColumnNames)
            {
                var val = df[name].GetObject(r);
                var text = val switch
                {
                    null => "<span class='null'>null</span>",
                    double d => d.ToString("G6"),
                    float f => f.ToString("G6"),
                    _ => Escape(val.ToString() ?? "")
                };
                sb.AppendLine($"<td>{text}</td>");
            }
            sb.AppendLine("</tr>");
        }
        sb.AppendLine("</tbody></table>");
        if (df.RowCount > maxRows)
            sb.AppendLine($"<p class='table-truncated'>Showing {rows} of {df.RowCount:N0} rows</p>");
    }

    private static string FormatInlineMarkup(string text)
    {
        // **bold** → <strong>
        text = Regex.Replace(text, @"\*\*(.+?)\*\*", "<strong>$1</strong>");
        // *italic* → <em>
        text = Regex.Replace(text, @"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", "<em>$1</em>");
        // `code` → <code>
        text = Regex.Replace(text, @"`(.+?)`", "<code>$1</code>");
        return text;
    }

    private static string Escape(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;").Replace("\"", "&quot;");

    private static string GetCss() => """
    :root {
        --bg: #ffffff; --fg: #1a1a2e; --fg-muted: #6c757d;
        --card-bg: #f8f9fa; --card-border: #e9ecef; --accent: #0f3460;
        --callout-info: #3498db; --callout-warn: #f39c12; --callout-success: #27ae60; --callout-note: #8e44ad;
        --table-border: #dee2e6; --table-stripe: #f8f9fa; --code-bg: #f1f3f5;
        --divider: #dee2e6;
    }
    body.story-dark {
        --bg: #1a1a2e; --fg: #e8e8e8; --fg-muted: #8899aa;
        --card-bg: #16213e; --card-border: #0f3460; --accent: #5dade2;
        --table-border: #2c3e50; --table-stripe: #16213e; --code-bg: #2c3e50;
        --divider: #2c3e50;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: var(--bg); color: var(--fg); font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; line-height: 1.6; }
    .story-container { max-width: 960px; margin: 0 auto; padding: 40px 24px; }
    .story-meta { text-align: center; color: var(--fg-muted); font-size: 0.9em; margin-bottom: 32px; }
    h1.story-title { text-align: center; font-size: 2.2em; font-weight: 700; color: var(--accent); margin-bottom: 8px; }
    h2 { font-size: 1.6em; font-weight: 600; margin: 40px 0 16px; padding-bottom: 8px; border-bottom: 2px solid var(--accent); }
    h3 { font-size: 1.2em; font-weight: 600; margin: 24px 0 12px; }
    p { margin: 12px 0; font-size: 1.05em; line-height: 1.7; }
    code { background: var(--code-bg); padding: 2px 6px; border-radius: 3px; font-size: 0.9em; }

    .stats-row { display: flex; gap: 16px; flex-wrap: wrap; margin: 24px 0; }
    .stat-card { flex: 1; min-width: 140px; background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 10px; padding: 20px; text-align: center; }
    .stat-value { font-size: 1.8em; font-weight: 700; color: var(--accent); }
    .stat-label { font-size: 0.85em; color: var(--fg-muted); margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px; }

    .story-chart { margin: 24px 0; border-radius: 8px; overflow: hidden; }
    .chart-caption { text-align: center; font-size: 0.9em; color: var(--fg-muted); font-style: italic; margin-top: -8px; }

    .callout { border-left: 4px solid var(--callout-info); padding: 16px 20px; margin: 20px 0; background: var(--card-bg); border-radius: 0 8px 8px 0; font-size: 1em; }
    .callout-warning { border-left-color: var(--callout-warn); }
    .callout-success { border-left-color: var(--callout-success); }
    .callout-note { border-left-color: var(--callout-note); }

    .story-table { margin: 20px 0; overflow-x: auto; }
    .story-table table { width: 100%; border-collapse: collapse; font-size: 0.92em; }
    .story-table th { background: var(--card-bg); font-weight: 600; text-align: left; padding: 10px 12px; border-bottom: 2px solid var(--table-border); }
    .story-table td { padding: 8px 12px; border-bottom: 1px solid var(--table-border); }
    .story-table tbody tr:nth-child(even) { background: var(--table-stripe); }
    .story-table .null { color: var(--fg-muted); font-style: italic; }
    .table-caption { font-weight: 600; font-size: 1em; margin-bottom: 4px; }
    .table-truncated { font-size: 0.85em; color: var(--fg-muted); text-align: center; margin-top: 8px; }

    .story-divider { border: none; border-top: 1px solid var(--divider); margin: 40px 0; }

    .story-row { display: flex; gap: 24px; margin: 24px 0; }
    .row-cell { flex: 1; min-width: 0; }

    .story-footer { text-align: center; color: var(--fg-muted); font-size: 0.8em; margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--divider); }

    @media (max-width: 640px) {
        .stats-row { flex-direction: column; }
        .story-row { flex-direction: column; }
        .story-container { padding: 20px 16px; }
        h1.story-title { font-size: 1.6em; }
    }
    """;
}
