using System.Diagnostics;
using PandaSharp;
using PandaSharp.ML.Models;
using PandaSharp.Viz.Rendering;

namespace PandaSharp.Viz.Charts;

// ===== Section model =====

public abstract record StorySection;
public record TitleSection(string Text, int Level = 1) : StorySection;
public record TextSection(string Content) : StorySection;
public record ChartSection(ChartSpec Spec, string? Caption = null) : StorySection;
public record StatsSection(IReadOnlyList<(string Label, string Value)> Items) : StorySection;
public record TableSection(DataFrame Data, int MaxRows = 20, string? Caption = null) : StorySection;
public record CalloutSection(string Content, CalloutStyle Style = CalloutStyle.Info) : StorySection;
public record DividerSection() : StorySection;
public record RawHtmlSection(string Html, string? Caption = null) : StorySection;
public record RowSection(IReadOnlyList<StorySection> Children) : StorySection;

public enum CalloutStyle { Info, Warning, Success, Note }
public enum StoryTheme { Light, Dark }

// ===== Builder =====

/// <summary>
/// Narrative data report builder — combines charts, text, statistics, and tables
/// into a single interactive HTML page. Like a data journalism article.
///
/// Usage:
///   StoryBoard.Create("My Report")
///       .Stats(("Rows", "1M"), ("Avg", "42.5"))
///       .Chart(df, v => v.Line("X", "Y").Title("Trend"))
///       .Text("The data shows a clear upward trend.")
///       .Table(df.Head(10))
///       .ToHtml("report.html");
/// </summary>
public class StoryBoard
{
    private readonly List<StorySection> _sections = new();
    private string? _title;
    private string? _author;
    private StoryTheme _theme = StoryTheme.Light;

    private StoryBoard() { }

    /// <summary>Create a new StoryBoard with a title.</summary>
    public static StoryBoard Create(string title)
    {
        var sb = new StoryBoard();
        sb._title = title;
        sb._sections.Add(new TitleSection(title, 1));
        return sb;
    }

    // ===== Metadata =====

    public StoryBoard Author(string author) { _author = author; return this; }
    public StoryBoard Theme(StoryTheme theme) { _theme = theme; return this; }

    // ===== Content sections =====

    /// <summary>Add a section heading (h2).</summary>
    public StoryBoard Section(string heading)
    {
        _sections.Add(new TitleSection(heading, 2));
        return this;
    }

    /// <summary>Add a subsection heading (h3).</summary>
    public StoryBoard Subsection(string heading)
    {
        _sections.Add(new TitleSection(heading, 3));
        return this;
    }

    /// <summary>Add narrative text. Supports **bold**, *italic*, and `code` inline markup.</summary>
    public StoryBoard Text(string content)
    {
        _sections.Add(new TextSection(content));
        return this;
    }

    /// <summary>Add a chart by configuring a VizBuilder via lambda.</summary>
    public StoryBoard Chart(DataFrame df, Action<VizBuilder> configure, string? caption = null)
    {
        var viz = new VizBuilder(df);
        configure(viz);
        _sections.Add(new ChartSection(viz.Spec, caption));
        return this;
    }

    /// <summary>Add a pre-built chart.</summary>
    public StoryBoard Chart(VizBuilder viz, string? caption = null)
    {
        _sections.Add(new ChartSection(viz.Spec, caption));
        return this;
    }

    /// <summary>Add a decision tree visualization from a regression tree.</summary>
    public StoryBoard Tree(DecisionTreeRegressor tree, string[]? featureNames = null, int maxDepth = 0, string? caption = null)
    {
        var divId = $"tree_{Guid.NewGuid():N}";
        _sections.Add(new RawHtmlSection(TreeVisualizer.ToFragment(tree, featureNames, maxDepth, divId), caption));
        return this;
    }

    /// <summary>Add a decision tree visualization from a classification tree.</summary>
    public StoryBoard Tree(DecisionTreeClassifier tree, string[]? featureNames = null, int maxDepth = 0, string? caption = null)
    {
        var divId = $"tree_{Guid.NewGuid():N}";
        _sections.Add(new RawHtmlSection(TreeVisualizer.ToFragment(tree, featureNames, maxDepth, divId), caption));
        return this;
    }

    /// <summary>Add a decision tree visualization from a random forest (specific tree index).</summary>
    public StoryBoard Tree(RandomForestRegressor forest, int treeIndex, string[]? featureNames = null, int maxDepth = 3, string? caption = null)
    {
        var divId = $"tree_{Guid.NewGuid():N}";
        _sections.Add(new RawHtmlSection(TreeVisualizer.ToFragment(forest, treeIndex, featureNames, maxDepth, divId), caption));
        return this;
    }

    /// <summary>Add a row of statistics cards.</summary>
    public StoryBoard Stats(params (string Label, string Value)[] items)
    {
        _sections.Add(new StatsSection(items));
        return this;
    }

    /// <summary>Add a data table.</summary>
    public StoryBoard Table(DataFrame df, int maxRows = 20, string? caption = null)
    {
        _sections.Add(new TableSection(df, maxRows, caption));
        return this;
    }

    /// <summary>Add a callout/highlight box.</summary>
    public StoryBoard Callout(string content, CalloutStyle style = CalloutStyle.Info)
    {
        _sections.Add(new CalloutSection(content, style));
        return this;
    }

    /// <summary>Add a grid of charts (SubplotBuilder) inline in the report.</summary>
    public StoryBoard Grid(SubplotBuilder grid, string? caption = null)
    {
        var id = $"grid_{Guid.NewGuid():N}";
        _sections.Add(new RawHtmlSection(grid.ToHtmlFragment(id), caption));
        return this;
    }

    /// <summary>Embed raw HTML content (e.g. a tree visualizer or custom D3 diagram).</summary>
    public StoryBoard Embed(string html, string? caption = null)
    {
        _sections.Add(new RawHtmlSection(html, caption));
        return this;
    }

    /// <summary>Add a horizontal divider.</summary>
    public StoryBoard Divider()
    {
        _sections.Add(new DividerSection());
        return this;
    }

    /// <summary>Add sections side-by-side in a row.</summary>
    public StoryBoard Row(params Action<StoryBoard>[] builders)
    {
        var children = new List<StorySection>();
        foreach (var b in builders)
        {
            var sub = new StoryBoard();
            b(sub);
            children.AddRange(sub._sections);
        }
        _sections.Add(new RowSection(children));
        return this;
    }

    // ===== Output =====

    /// <summary>Render to a complete HTML string.</summary>
    public string ToHtmlString() => StoryBoardRenderer.Render(_sections, _title, _author, _theme);

    /// <summary>Write to an HTML file.</summary>
    public void ToHtml(string path) => File.WriteAllText(path, ToHtmlString());

    /// <summary>Write to a temp file and open in the default browser.</summary>
    public void Show()
    {
        var path = Path.Combine(Path.GetTempPath(), $"pandasharp_story_{Guid.NewGuid():N}.html");
        ToHtml(path);
        Process.Start(new ProcessStartInfo(path) { UseShellExecute = true });
    }
}
