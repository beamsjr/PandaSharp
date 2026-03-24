using PandaSharp;
using PandaSharp.Column;
using ScottPlot;

namespace PandaSharp.Plot;

public static class PlotExtensions
{
    /// <summary>
    /// Access the plot builder for a DataFrame.
    /// Usage: df.Plot().Bar(x: "Category", y: "Value").Save("chart.png")
    /// </summary>
    public static PlotBuilder Plot(this DataFrame df) => new(df);
}

public class PlotBuilder
{
    private readonly DataFrame _df;

    internal PlotBuilder(DataFrame df) => _df = df;

    public PlotResult Scatter(string x, string y, string? title = null)
    {
        var plt = new ScottPlot.Plot();
        var xs = ToDoubleArray(_df[x]);
        var ys = ToDoubleArray(_df[y]);
        plt.Add.Scatter(xs, ys);
        if (title is not null) plt.Title(title);
        plt.XLabel(x);
        plt.YLabel(y);
        return new PlotResult(plt);
    }

    public PlotResult Line(string x, string y, string? title = null)
    {
        var plt = new ScottPlot.Plot();
        var xs = ToDoubleArray(_df[x]);
        var ys = ToDoubleArray(_df[y]);
        plt.Add.Scatter(xs, ys);
        if (title is not null) plt.Title(title);
        plt.XLabel(x);
        plt.YLabel(y);
        return new PlotResult(plt);
    }

    public PlotResult Bar(string x, string y, string? title = null)
    {
        var plt = new ScottPlot.Plot();
        var ys = ToDoubleArray(_df[y]);
        var bars = new List<ScottPlot.Bar>();
        for (int i = 0; i < ys.Length; i++)
            bars.Add(new ScottPlot.Bar { Position = i, Value = ys[i] });
        plt.Add.Bars(bars.ToArray());

        // Set tick labels from x column
        var labels = new string[_df.RowCount];
        var xCol = _df[x];
        for (int i = 0; i < _df.RowCount; i++)
            labels[i] = xCol.GetObject(i)?.ToString() ?? "";

        var ticks = new ScottPlot.Tick[labels.Length];
        for (int i = 0; i < labels.Length; i++)
            ticks[i] = new ScottPlot.Tick(i, labels[i]);
        plt.Axes.Bottom.TickGenerator = new ScottPlot.TickGenerators.NumericManual(ticks);

        if (title is not null) plt.Title(title);
        return new PlotResult(plt);
    }

    public PlotResult Histogram(string column, int bins = 20, string? title = null)
    {
        var plt = new ScottPlot.Plot();
        var values = ToDoubleArray(_df[column]).Where(v => !double.IsNaN(v)).ToArray();
        if (values.Length == 0) return new PlotResult(plt);

        double min = values.Min();
        double max = values.Max();
        double binWidth = (max - min) / bins;
        if (binWidth == 0) binWidth = 1;

        var counts = new int[bins];
        foreach (var v in values)
        {
            int idx = Math.Min((int)((v - min) / binWidth), bins - 1);
            counts[idx]++;
        }

        var bars = new ScottPlot.Bar[bins];
        for (int i = 0; i < bins; i++)
        {
            bars[i] = new ScottPlot.Bar
            {
                Position = min + (i + 0.5) * binWidth,
                Value = counts[i],
                Size = binWidth * 0.9
            };
        }
        plt.Add.Bars(bars);
        if (title is not null) plt.Title(title);
        plt.XLabel(column);
        plt.YLabel("Count");
        return new PlotResult(plt);
    }

    private static double[] ToDoubleArray(IColumn col)
    {
        var result = new double[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.IsNull(i) ? double.NaN : Convert.ToDouble(col.GetObject(i));
        return result;
    }
}

public class PlotResult
{
    public ScottPlot.Plot Plot { get; }

    internal PlotResult(ScottPlot.Plot plot) => Plot = plot;

    public PlotResult Title(string title) { Plot.Title(title); return this; }
    public PlotResult XLabel(string label) { Plot.XLabel(label); return this; }
    public PlotResult YLabel(string label) { Plot.YLabel(label); return this; }

    public void SavePng(string path, int width = 800, int height = 600)
        => Plot.SavePng(path, width, height);

    public void SaveSvg(string path, int width = 800, int height = 600)
        => Plot.SaveSvg(path, width, height);
}
