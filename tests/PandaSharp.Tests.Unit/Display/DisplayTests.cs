using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Display;

namespace PandaSharp.Tests.Unit.Display;

public class DisplayTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob"]),
        new Column<int>("Age", [25, 30]),
        new Column<double>("Score", [95.5, 87.3])
    );

    [Fact]
    public void ToMarkdown_ProducesValidTable()
    {
        var md = Df().ToMarkdown();

        md.Should().Contain("| Name | Age | Score |");
        md.Should().Contain("| :--- | ---: | ---: |"); // string left, numbers right
        md.Should().Contain("| Alice | 25 | 95.5 |");
        md.Should().Contain("| Bob | 30 | 87.3 |");
    }

    [Fact]
    public void ToMarkdown_EmptyDataFrame()
    {
        var md = new DataFrame().ToMarkdown();
        md.Should().Contain("empty");
    }

    [Fact]
    public void ToMarkdown_Truncates()
    {
        var names = Enumerable.Range(0, 100).Select(i => $"Name{i}").ToArray();
        var df = new DataFrame(new StringColumn("Name", names));

        var md = df.ToMarkdown(maxRows: 5);
        md.Should().Contain("95 more rows");
    }

    [Fact]
    public void ToMarkdown_NullsAsEmpty()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );
        var md = df.ToMarkdown();
        md.Should().Contain("| 1 |");
        md.Should().Contain("|  |"); // null = empty
    }

    [Fact]
    public void ToString_ConsoleFormat()
    {
        var output = Df().ToString();
        output.Should().Contain("┌");
        output.Should().Contain("Alice");
        output.Should().Contain("2 rows x 3 columns");
    }

    [Fact]
    public void ToHtml_ProducesTable()
    {
        var html = Df().ToHtml();
        html.Should().Contain("<table");
        html.Should().Contain("Alice");
        html.Should().Contain("2 rows");
    }
}
