using PandaSharp;

namespace PandaSharp.Viz.Charts;

/// <summary>
/// Combine multiple VizBuilder charts into a grid layout.
/// Usage: Subplots.Grid(chart1, chart2, chart3).Cols(2).ToHtmlString()
/// </summary>
public class SubplotBuilder
{
    private readonly List<VizBuilder> _charts = new();
    private int _cols = 2;
    private string? _title;

    private SubplotBuilder() { }

    public static SubplotBuilder Grid(params VizBuilder[] charts)
    {
        var builder = new SubplotBuilder();
        builder._charts.AddRange(charts);
        return builder;
    }

    public SubplotBuilder Cols(int cols) { _cols = cols; return this; }
    public SubplotBuilder Title(string title) { _title = title; return this; }

    public string ToHtmlString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'>");
        sb.AppendLine("<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>");
        sb.AppendLine("<style>body{font-family:sans-serif;} .subplot-grid{display:grid;gap:10px;} .subplot-cell{}</style>");
        sb.AppendLine("</head><body>");
        if (_title is not null) sb.AppendLine($"<h2>{_title}</h2>");
        sb.AppendLine($"<div class='subplot-grid' style='grid-template-columns:repeat({_cols},1fr);'>");

        for (int i = 0; i < _charts.Count; i++)
        {
            var divId = $"subplot_{i}";
            sb.AppendLine($"<div class='subplot-cell'>{_charts[i].ToHtmlFragment(divId)}</div>");
        }

        sb.AppendLine("</div></body></html>");
        return sb.ToString();
    }

    public void ToHtml(string path) => File.WriteAllText(path, ToHtmlString());
}
