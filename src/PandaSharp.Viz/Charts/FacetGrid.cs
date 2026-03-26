using PandaSharp;

namespace PandaSharp.Viz.Charts;

/// <summary>
/// Create a grid of charts, one per unique value in a facet column.
/// Usage: df.Viz().FacetGrid("Region").Line("Month", "Revenue").ToHtmlString()
/// </summary>
public class FacetGridBuilder
{
    private readonly DataFrame _df;
    private readonly string _facetColumn;
    private readonly List<Action<VizBuilder>> _chartBuilders = new();
    private string? _title;
    private int _colWrap = 3;
    private int _cellWidth = 400;
    private int _cellHeight = 300;
    private string? _theme;

    internal FacetGridBuilder(DataFrame df, string facetColumn)
    {
        _df = df;
        _facetColumn = facetColumn;
    }

    public FacetGridBuilder Line(string x, string y) { _chartBuilders.Add(v => v.Line(x, y)); return this; }
    public FacetGridBuilder Bar(string x, string y) { _chartBuilders.Add(v => v.Bar(x, y)); return this; }
    public FacetGridBuilder Scatter(string x, string y) { _chartBuilders.Add(v => v.Scatter(x, y)); return this; }
    public FacetGridBuilder Histogram(string column, int? bins = null) { _chartBuilders.Add(v => v.Histogram(column, bins)); return this; }
    public FacetGridBuilder Box(string? x = null, string? y = null) { _chartBuilders.Add(v => v.Box(x, y)); return this; }

    public FacetGridBuilder Title(string title) { _title = title; return this; }
    public FacetGridBuilder ColWrap(int cols) { _colWrap = cols; return this; }
    public FacetGridBuilder CellSize(int width, int height) { _cellWidth = width; _cellHeight = height; return this; }
    public FacetGridBuilder Theme(string theme) { _theme = theme; return this; }

    public string ToHtmlString()
    {
        var facetCol = _df[_facetColumn];
        var uniqueValues = new List<string>();
        var seen = new HashSet<string>();
        for (int i = 0; i < facetCol.Length; i++)
        {
            var val = facetCol.GetObject(i)?.ToString() ?? "null";
            if (seen.Add(val)) uniqueValues.Add(val);
        }

        var sb = new System.Text.StringBuilder();
        sb.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'>");
        sb.AppendLine($"<script src='{Rendering.D3HtmlRenderer.CdnUrl}'></script>");
        if (_title is not null) sb.AppendLine($"<title>{_title}</title>");
        sb.AppendLine("<style>body{font-family:sans-serif;} .facet-grid{display:flex;flex-wrap:wrap;gap:10px;} .facet-cell{}</style>");
        sb.AppendLine("</head><body>");
        if (_title is not null) sb.AppendLine($"<h2>{_title}</h2>");
        sb.AppendLine("<div class='facet-grid'>");

        for (int i = 0; i < uniqueValues.Count; i++)
        {
            var facetValue = uniqueValues[i];
            var mask = new bool[_df.RowCount];
            for (int r = 0; r < _df.RowCount; r++)
                mask[r] = (facetCol.GetObject(r)?.ToString() ?? "null") == facetValue;
            var subset = _df.Filter(mask);

            var vizBuilder = new VizBuilder(subset);
            foreach (var builder in _chartBuilders) builder(vizBuilder);
            vizBuilder.Title($"{_facetColumn} = {facetValue}")
                .Size(_cellWidth, _cellHeight);
            if (_theme is not null) vizBuilder.Theme(_theme);

            var divId = $"facet_{i}";
            sb.AppendLine($"<div class='facet-cell'>{vizBuilder.ToHtmlFragment(divId)}</div>");
        }

        sb.AppendLine("</div></body></html>");
        return sb.ToString();
    }

    public void ToHtml(string path) => File.WriteAllText(path, ToHtmlString());

    public void Show()
    {
        var path = Path.Combine(Path.GetTempPath(), $"pandasharp_facet_{Guid.NewGuid():N}.html");
        ToHtml(path);
        if (OperatingSystem.IsMacOS()) System.Diagnostics.Process.Start("open", path);
        else if (OperatingSystem.IsWindows()) System.Diagnostics.Process.Start(new System.Diagnostics.ProcessStartInfo(path) { UseShellExecute = true });
    }
}
