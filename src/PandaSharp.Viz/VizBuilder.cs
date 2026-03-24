using System.Diagnostics;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Viz.Charts;
using PandaSharp.Viz.Rendering;
using PandaSharp.Viz.Themes;

namespace PandaSharp.Viz;

/// <summary>
/// Fluent API for building interactive Plotly.js charts from DataFrames.
/// Usage: df.Viz.Bar("Category", "Sales").Title("Revenue").Show()
/// </summary>
public class VizBuilder
{
    private readonly DataFrame _df;
    private readonly ChartSpec _spec = new();

    internal VizBuilder(DataFrame df) => _df = df;

    // ===== Chart types =====

    /// <summary>Bar chart.</summary>
    public VizBuilder Bar(string x, string y, string? color = null, bool horizontal = false, string? barmode = null)
    {
        if (color is not null)
        {
            foreach (var group in GetGroups(color))
            {
                _spec.Traces.Add(new TraceSpec
                {
                    Type = "bar",
                    Name = group.Key,
                    XLabels = horizontal ? null : GetStringValues(group.Df, x),
                    YLabels = horizontal ? GetStringValues(group.Df, x) : null,
                    X = horizontal ? GetDoubleValues(group.Df, y) : null,
                    Y = horizontal ? null : GetDoubleValues(group.Df, y),
                    Orientation = horizontal ? "h" : null,
                    XColumn = x, YColumn = y,
                    XIsString = !horizontal, YIsString = horizontal,
                });
            }
            _spec.Layout.Barmode = barmode ?? "group";
        }
        else
        {
            _spec.Traces.Add(new TraceSpec
            {
                Type = "bar",
                XLabels = horizontal ? null : GetStringValues(_df, x),
                YLabels = horizontal ? GetStringValues(_df, x) : null,
                X = horizontal ? GetDoubleValues(_df, y) : null,
                Y = horizontal ? null : GetDoubleValues(_df, y),
                Orientation = horizontal ? "h" : null,
                XColumn = x, YColumn = y,
                XIsString = !horizontal, YIsString = horizontal,
            });
        }
        return this;
    }

    /// <summary>Line chart.</summary>
    public VizBuilder Line(string x, string y, string? color = null, bool markers = false)
    {
        var mode = markers ? "lines+markers" : "lines";
        if (color is not null)
        {
            foreach (var group in GetGroups(color))
            {
                _spec.Traces.Add(new TraceSpec
                {
                    Type = "scatter", Mode = mode, Name = group.Key,
                    X = GetDoubleValues(group.Df, x),
                    Y = GetDoubleValues(group.Df, y),
                    XColumn = x, YColumn = y,
                });
            }
        }
        else
        {
            _spec.Traces.Add(new TraceSpec
            {
                Type = "scatter", Mode = mode,
                X = GetDoubleValues(_df, x),
                Y = GetDoubleValues(_df, y),
                XColumn = x, YColumn = y,
            });
        }
        return this;
    }

    /// <summary>Scatter plot with optional color/size encoding.</summary>
    public VizBuilder Scatter(string x, string y, string? color = null, string? size = null,
        string? text = null, bool webgl = false)
    {
        var traceType = webgl || _df.RowCount > 50_000 ? "scattergl" : "scatter";

        if (color is not null)
        {
            foreach (var group in GetGroups(color))
            {
                var trace = new TraceSpec
                {
                    Type = traceType, Mode = "markers", Name = group.Key,
                    X = GetDoubleValues(group.Df, x),
                    Y = GetDoubleValues(group.Df, y),
                    XColumn = x, YColumn = y, SizeColumn = size, TextColumn = text,
                };
                if (size is not null) trace.MarkerSize = GetDoubleValues(group.Df, size);
                if (text is not null) trace.Text = GetStringValues(group.Df, text);
                _spec.Traces.Add(trace);
            }
        }
        else
        {
            var trace = new TraceSpec
            {
                Type = traceType, Mode = "markers",
                X = GetDoubleValues(_df, x),
                Y = GetDoubleValues(_df, y),
                XColumn = x, YColumn = y, SizeColumn = size, TextColumn = text,
            };
            if (size is not null) trace.MarkerSize = GetDoubleValues(_df, size);
            if (text is not null) trace.Text = GetStringValues(_df, text);
            _spec.Traces.Add(trace);
        }
        return this;
    }

    /// <summary>Histogram.</summary>
    public VizBuilder Histogram(string column, int? bins = null, bool density = false)
    {
        var trace = new TraceSpec
        {
            Type = "histogram",
            X = GetDoubleValues(_df, column),
        };
        if (bins.HasValue) trace.NBinsX = bins.Value;
        if (density) trace.Extra["histnorm"] = "probability density";
        _spec.Traces.Add(trace);
        return this;
    }

    /// <summary>Box plot.</summary>
    public VizBuilder Box(string? x = null, string? y = null, string? color = null)
    {
        if (color is not null)
        {
            foreach (var group in GetGroups(color))
            {
                _spec.Traces.Add(new TraceSpec
                {
                    Type = "box", Name = group.Key,
                    X = x is not null ? GetDoubleValues(group.Df, x) : null,
                    XLabels = x is not null ? GetStringValues(group.Df, x) : null,
                    Y = y is not null ? GetDoubleValues(group.Df, y) : null,
                });
            }
        }
        else
        {
            _spec.Traces.Add(new TraceSpec
            {
                Type = "box",
                X = x is not null ? GetDoubleValues(_df, x) : null,
                XLabels = x is not null ? GetStringValues(_df, x) : null,
                Y = y is not null ? GetDoubleValues(_df, y) : null,
            });
        }
        return this;
    }

    /// <summary>Heatmap from a 2D data matrix.</summary>
    public VizBuilder Heatmap(string[] valueColumns, string[]? xLabels = null, string[]? yLabels = null)
    {
        int rows = _df.RowCount;
        int cols = valueColumns.Length;
        var z = new double[rows * cols];
        for (int c = 0; c < cols; c++)
        {
            var col = _df[valueColumns[c]];
            for (int r = 0; r < rows; r++)
                z[r * cols + c] = col.IsNull(r) ? double.NaN : Convert.ToDouble(col.GetObject(r));
        }

        _spec.Traces.Add(new TraceSpec
        {
            Type = "heatmap",
            Z = z, ZRows = rows, ZCols = cols,
            XLabels = xLabels ?? valueColumns,
            YLabels = yLabels,
        });
        return this;
    }

    /// <summary>Pie chart.</summary>
    public VizBuilder Pie(string labels, string values)
    {
        _spec.Traces.Add(new TraceSpec
        {
            Type = "pie",
            Extra =
            {
                ["labels"] = GetStringValues(_df, labels),
                ["values"] = GetDoubleValues(_df, values),
            }
        });
        return this;
    }

    /// <summary>Area chart (filled line).</summary>
    public VizBuilder Area(string x, string y, string? color = null, bool stacked = false)
    {
        if (color is not null)
        {
            foreach (var group in GetGroups(color))
            {
                _spec.Traces.Add(new TraceSpec
                {
                    Type = "scatter", Mode = "lines",
                    Name = group.Key,
                    X = GetDoubleValues(group.Df, x),
                    Y = GetDoubleValues(group.Df, y),
                    Extra = { ["fill"] = stacked ? "tonexty" : "tozeroy" }
                });
            }
        }
        else
        {
            _spec.Traces.Add(new TraceSpec
            {
                Type = "scatter", Mode = "lines",
                X = GetDoubleValues(_df, x),
                Y = GetDoubleValues(_df, y),
                Extra = { ["fill"] = "tozeroy" }
            });
        }
        return this;
    }

    // ===== Animation =====

    /// <summary>
    /// Animate the chart over a column's unique values (e.g. .Animate("Year") for time-based transitions).
    /// Call after setting the chart type (Bar, Line, Scatter, etc.).
    /// Each unique value in the animation column becomes a frame.
    /// </summary>
    public VizBuilder Animate(string column, int frameDurationMs = 500, int transitionDurationMs = 300)
    {
        _spec.AnimationColumn = column;

        // Get sorted unique frame values
        var col = _df[column];
        var frameValues = new SortedSet<string>();
        for (int i = 0; i < _df.RowCount; i++)
        {
            var val = col.GetObject(i)?.ToString();
            if (val is not null) frameValues.Add(val);
        }

        // Group rows by frame value
        var frameGroups = new Dictionary<string, List<int>>();
        foreach (var v in frameValues) frameGroups[v] = new List<int>();
        for (int i = 0; i < _df.RowCount; i++)
        {
            var val = col.GetObject(i)?.ToString();
            if (val is not null && frameGroups.TryGetValue(val, out var list))
                list.Add(i);
        }

        // Build frames using stored column names from trace construction
        foreach (var frameVal in frameValues)
        {
            var indices = frameGroups[frameVal].ToArray();
            var frameDf = new DataFrame(_df.ColumnNames.Select(n => _df[n].TakeRows(indices)));
            var frame = new Charts.FrameSpec { Name = frameVal };

            foreach (var origTrace in _spec.Traces)
            {
                var ft = new Charts.TraceSpec
                {
                    Type = origTrace.Type,
                    Mode = origTrace.Mode,
                    Name = origTrace.Name,
                    Orientation = origTrace.Orientation,
                    XColumn = origTrace.XColumn,
                    YColumn = origTrace.YColumn,
                    SizeColumn = origTrace.SizeColumn,
                    TextColumn = origTrace.TextColumn,
                    XIsString = origTrace.XIsString,
                    YIsString = origTrace.YIsString,
                };

                if (origTrace.XColumn is not null)
                {
                    if (origTrace.XIsString)
                        ft.XLabels = GetStringValues(frameDf, origTrace.XColumn);
                    else
                        ft.X = GetDoubleValues(frameDf, origTrace.XColumn);
                }
                if (origTrace.YColumn is not null)
                {
                    if (origTrace.YIsString)
                        ft.YLabels = GetStringValues(frameDf, origTrace.YColumn);
                    else
                        ft.Y = GetDoubleValues(frameDf, origTrace.YColumn);
                }
                if (origTrace.SizeColumn is not null)
                    ft.MarkerSize = GetDoubleValues(frameDf, origTrace.SizeColumn);
                if (origTrace.TextColumn is not null)
                    ft.Text = GetStringValues(frameDf, origTrace.TextColumn);

                foreach (var (key, val) in origTrace.Extra)
                    ft.Extra[key] = val;

                frame.Data.Add(ft);
            }

            _spec.Frames.Add(frame);
        }

        // Set up initial traces to show first frame
        if (frameValues.Count > 0)
        {
            var firstFrame = _spec.Frames[0];
            _spec.Traces.Clear();
            foreach (var t in firstFrame.Data)
                _spec.Traces.Add(t);
        }

        // Add slider and play/pause buttons to layout
        _spec.Layout.Extra["updatemenus"] = new object[]
        {
            new Dictionary<string, object?>
            {
                ["type"] = "buttons",
                ["showactive"] = false,
                ["x"] = 0.0,
                ["y"] = 0.0,
                ["xanchor"] = "right",
                ["yanchor"] = "top",
                ["pad"] = new Dictionary<string, int> { ["t"] = 60, ["r"] = 20 },
                ["buttons"] = new object[]
                {
                    new Dictionary<string, object?>
                    {
                        ["label"] = "▶ Play",
                        ["method"] = "animate",
                        ["args"] = new object?[]
                        {
                            null,
                            new Dictionary<string, object?>
                            {
                                ["fromcurrent"] = true,
                                ["frame"] = new Dictionary<string, int> { ["duration"] = frameDurationMs },
                                ["transition"] = new Dictionary<string, int> { ["duration"] = transitionDurationMs },
                            }
                        }
                    },
                    new Dictionary<string, object?>
                    {
                        ["label"] = "⏸ Pause",
                        ["method"] = "animate",
                        ["args"] = new object?[]
                        {
                            new object[] { null },
                            new Dictionary<string, object?>
                            {
                                ["mode"] = "immediate",
                                ["frame"] = new Dictionary<string, int> { ["duration"] = 0 },
                                ["transition"] = new Dictionary<string, int> { ["duration"] = 0 },
                            }
                        }
                    }
                }
            }
        };

        _spec.Layout.Extra["sliders"] = new object[]
        {
            new Dictionary<string, object?>
            {
                ["active"] = 0,
                ["pad"] = new Dictionary<string, int> { ["t"] = 50 },
                ["steps"] = frameValues.Select(v => new Dictionary<string, object?>
                {
                    ["label"] = v,
                    ["method"] = "animate",
                    ["args"] = new object?[]
                    {
                        new object[] { v },
                        new Dictionary<string, object?>
                        {
                            ["mode"] = "immediate",
                            ["frame"] = new Dictionary<string, int> { ["duration"] = frameDurationMs },
                            ["transition"] = new Dictionary<string, int> { ["duration"] = transitionDurationMs },
                        }
                    }
                }).ToArray(),
            }
        };

        return this;
    }

    // ===== Customization =====

    public VizBuilder Title(string title) { _spec.Layout.Title = title; return this; }
    public VizBuilder XLabel(string label) { _spec.Layout.XAxisTitle = label; return this; }
    public VizBuilder YLabel(string label) { _spec.Layout.YAxisTitle = label; return this; }
    public VizBuilder Width(int w) { _spec.Layout.Width = w; return this; }
    public VizBuilder Height(int h) { _spec.Layout.Height = h; return this; }
    public VizBuilder Size(int w, int h) { _spec.Layout.Width = w; _spec.Layout.Height = h; return this; }
    public VizBuilder Theme(string theme) { _spec.Layout.Template = theme; return this; }
    public VizBuilder Legend(bool show) { _spec.Layout.ShowLegend = show; return this; }

    // ===== Export =====

    /// <summary>Generate self-contained HTML string.</summary>
    public string ToHtmlString() => HtmlRenderer.Render(_spec);

    /// <summary>Generate embeddable HTML fragment (div + script).</summary>
    public string ToHtmlFragment(string divId = "chart") => HtmlRenderer.RenderFragment(_spec, divId);

    /// <summary>Write to an HTML file.</summary>
    public void ToHtml(string path) => File.WriteAllText(path, ToHtmlString());

    /// <summary>Open the chart in the default browser.</summary>
    public void Show()
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"pandasharp_viz_{Guid.NewGuid():N}.html");
        ToHtml(tempPath);

        if (OperatingSystem.IsMacOS())
            Process.Start("open", tempPath);
        else if (OperatingSystem.IsLinux())
            Process.Start("xdg-open", tempPath);
        else if (OperatingSystem.IsWindows())
            Process.Start(new ProcessStartInfo(tempPath) { UseShellExecute = true });
    }

    /// <summary>
    /// Export the chart as a PNG image file.
    /// Requires Node.js with the 'puppeteer' package installed.
    /// </summary>
    public void ToPng(string path, int? width = null, int? height = null, int scale = 2)
        => StaticExporter.Export(_spec, path, "png", width ?? _spec.Layout.Width, height ?? _spec.Layout.Height, scale);

    /// <summary>
    /// Export the chart as an SVG image file.
    /// Requires Node.js with the 'puppeteer' package installed.
    /// </summary>
    public void ToSvg(string path, int? width = null, int? height = null)
        => StaticExporter.Export(_spec, path, "svg", width ?? _spec.Layout.Width, height ?? _spec.Layout.Height, 1);

    /// <summary>
    /// Export the chart as PNG bytes.
    /// Requires Node.js with the 'puppeteer' package installed.
    /// </summary>
    public byte[] ToPngBytes(int? width = null, int? height = null, int scale = 2)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"pandasharp_export_{Guid.NewGuid():N}.png");
        try
        {
            ToPng(tempPath, width, height, scale);
            return File.ReadAllBytes(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    /// <summary>
    /// Export the chart as an SVG string.
    /// Requires Node.js with the 'puppeteer' package installed.
    /// </summary>
    public string ToSvgString(int? width = null, int? height = null)
    {
        var tempPath = Path.Combine(Path.GetTempPath(), $"pandasharp_export_{Guid.NewGuid():N}.svg");
        try
        {
            ToSvg(tempPath, width, height);
            return File.ReadAllText(tempPath);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    /// <summary>Access the raw chart spec for advanced customization.</summary>
    public ChartSpec Spec => _spec;

    // ===== Data extraction helpers =====

    private static double[] GetDoubleValues(DataFrame df, string column)
    {
        var col = df[column];
        var result = new double[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            result[i] = col.IsNull(i) ? double.NaN : Convert.ToDouble(col.GetObject(i));
        return result;
    }

    private static string[] GetStringValues(DataFrame df, string column)
    {
        var col = df[column];
        var result = new string[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            result[i] = col.GetObject(i)?.ToString() ?? "";
        return result;
    }

    private List<(string Key, DataFrame Df)> GetGroups(string colorColumn)
    {
        var col = _df[colorColumn];
        var groups = new Dictionary<string, List<int>>();
        var order = new List<string>();
        for (int i = 0; i < _df.RowCount; i++)
        {
            var key = col.GetObject(i)?.ToString() ?? "null";
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<int>();
                groups[key] = list;
                order.Add(key);
            }
            list.Add(i);
        }

        return order.Select(key =>
        {
            var idx = groups[key].ToArray();
            var groupDf = new DataFrame(_df.ColumnNames.Select(n => _df[n].TakeRows(idx)));
            return (key, groupDf);
        }).ToList();
    }
}
