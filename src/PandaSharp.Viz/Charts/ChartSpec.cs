namespace PandaSharp.Viz.Charts;

/// <summary>
/// Internal representation of a chart before rendering.
/// Holds Plotly trace data + layout configuration.
/// </summary>
public class ChartSpec
{
    public List<TraceSpec> Traces { get; } = new();
    public LayoutSpec Layout { get; set; } = new();
    public List<FrameSpec> Frames { get; } = new();
    public string? AnimationColumn { get; set; }
}

public class TraceSpec
{
    public string Type { get; set; } = "scatter";
    public string? Mode { get; set; }
    public string? Name { get; set; }
    public double[]? X { get; set; }
    public double[]? Y { get; set; }
    public string[]? XLabels { get; set; }
    public string[]? YLabels { get; set; }
    public string[]? Text { get; set; }
    public double[]? MarkerSize { get; set; }
    public string[]? MarkerColor { get; set; }
    public string? Orientation { get; set; } // "h" for horizontal
    public int? NBinsX { get; set; }
    public double[]? Z { get; set; } // for heatmap (flattened)
    public int? ZRows { get; set; }
    public int? ZCols { get; set; }
    public Dictionary<string, object?> Extra { get; } = new();

    // Source column names for animation frame re-extraction
    internal string? XColumn { get; set; }
    internal string? YColumn { get; set; }
    internal string? SizeColumn { get; set; }
    internal string? TextColumn { get; set; }
    internal bool XIsString { get; set; }
    internal bool YIsString { get; set; }
}

public class LayoutSpec
{
    public string? Title { get; set; }
    public string? XAxisTitle { get; set; }
    public string? YAxisTitle { get; set; }
    public int Width { get; set; } = 800;
    public int Height { get; set; } = 500;
    public string? Barmode { get; set; } // "group", "stack"
    public bool ShowLegend { get; set; } = true;
    public string? Template { get; set; } // "plotly_dark", "plotly_white", etc.
    public Dictionary<string, object?> Extra { get; } = new();
}

/// <summary>
/// Represents a single animation frame in a Plotly animation.
/// </summary>
public class FrameSpec
{
    public string Name { get; set; } = "";
    public List<TraceSpec> Data { get; } = new();
}
