using System.Text;
using System.Text.Json;
using PandaSharp.Viz.Charts;

namespace PandaSharp.Viz.Rendering;

/// <summary>
/// Serializes ChartSpec into Plotly.js JSON data and layout objects.
/// </summary>
internal static class PlotlySerializer
{
    public static string SerializeTraces(ChartSpec spec)
    {
        var traces = new List<Dictionary<string, object?>>();

        foreach (var trace in spec.Traces)
        {
            var t = new Dictionary<string, object?> { ["type"] = trace.Type };

            if (trace.Mode is not null) t["mode"] = trace.Mode;
            if (trace.Name is not null) t["name"] = trace.Name;
            if (trace.Orientation is not null) t["orientation"] = trace.Orientation;

            // X axis
            if (trace.XLabels is not null)
                t["x"] = trace.XLabels;
            else if (trace.X is not null)
                t["x"] = trace.X;

            // Y axis
            if (trace.YLabels is not null)
                t["y"] = trace.YLabels;
            else if (trace.Y is not null)
                t["y"] = trace.Y;

            // Text
            if (trace.Text is not null) t["text"] = trace.Text;

            // Marker
            if (trace.MarkerSize is not null || trace.MarkerColor is not null)
            {
                var marker = new Dictionary<string, object?>();
                if (trace.MarkerSize is not null) marker["size"] = trace.MarkerSize;
                if (trace.MarkerColor is not null) marker["color"] = trace.MarkerColor;
                t["marker"] = marker;
            }

            // Histogram bins
            if (trace.NBinsX.HasValue)
                t["nbinsx"] = trace.NBinsX.Value;

            // Heatmap Z data
            if (trace.Z is not null && trace.ZRows.HasValue && trace.ZCols.HasValue)
            {
                var zMatrix = new double[trace.ZRows.Value][];
                for (int r = 0; r < trace.ZRows.Value; r++)
                {
                    zMatrix[r] = new double[trace.ZCols.Value];
                    for (int c = 0; c < trace.ZCols.Value; c++)
                        zMatrix[r][c] = trace.Z[r * trace.ZCols.Value + c];
                }
                t["z"] = zMatrix;
                if (trace.XLabels is not null) t["x"] = trace.XLabels;
                if (trace.YLabels is not null) t["y"] = trace.YLabels;
            }

            // Extra properties
            foreach (var (key, val) in trace.Extra)
                t[key] = val;

            traces.Add(t);
        }

        return JsonSerializer.Serialize(traces, JsonOpts);
    }

    public static string SerializeLayout(ChartSpec spec)
    {
        var layout = new Dictionary<string, object?>();

        if (spec.Layout.Title is not null) layout["title"] = spec.Layout.Title;
        layout["width"] = spec.Layout.Width;
        layout["height"] = spec.Layout.Height;
        layout["showlegend"] = spec.Layout.ShowLegend;

        if (spec.Layout.XAxisTitle is not null)
            layout["xaxis"] = new Dictionary<string, object?> { ["title"] = spec.Layout.XAxisTitle };
        if (spec.Layout.YAxisTitle is not null)
            layout["yaxis"] = new Dictionary<string, object?> { ["title"] = spec.Layout.YAxisTitle };
        if (spec.Layout.Barmode is not null)
            layout["barmode"] = spec.Layout.Barmode;
        if (spec.Layout.Template is not null)
            layout["template"] = spec.Layout.Template;

        foreach (var (key, val) in spec.Layout.Extra)
            layout[key] = val;

        return JsonSerializer.Serialize(layout, JsonOpts);
    }

    public static string SerializeFrames(ChartSpec spec)
    {
        var frames = new List<Dictionary<string, object?>>();
        foreach (var frame in spec.Frames)
        {
            var f = new Dictionary<string, object?> { ["name"] = frame.Name };
            var data = new List<Dictionary<string, object?>>();
            foreach (var trace in frame.Data)
            {
                var t = new Dictionary<string, object?>();
                if (trace.XLabels is not null) t["x"] = trace.XLabels;
                else if (trace.X is not null) t["x"] = trace.X;
                if (trace.YLabels is not null) t["y"] = trace.YLabels;
                else if (trace.Y is not null) t["y"] = trace.Y;
                if (trace.MarkerSize is not null)
                    t["marker"] = new Dictionary<string, object?> { ["size"] = trace.MarkerSize };
                if (trace.Text is not null) t["text"] = trace.Text;
                data.Add(t);
            }
            f["data"] = data;
            frames.Add(f);
        }
        return JsonSerializer.Serialize(frames, JsonOpts);
    }

    private static readonly JsonSerializerOptions JsonOpts = new()
    {
        WriteIndented = false,
        Encoder = System.Text.Encodings.Web.JavaScriptEncoder.UnsafeRelaxedJsonEscaping
    };
}
