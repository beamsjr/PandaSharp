using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Interactive;

namespace PandaSharp.Tests.Unit.Display;

public class InteractiveFormatterTests
{
    [Fact]
    public void FormatDataFrame_ProducesStyledHtml()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35]),
            new Column<double>("Salary", [50_000.0, 62_000.0, 75_000.0])
        );

        var html = DataFrameKernelExtension.FormatDataFrame(df);

        // Structure
        html.Should().Contain("<table");
        html.Should().Contain("<thead>");
        html.Should().Contain("<tbody>");
        html.Should().Contain("</table>");

        // Column headers with dtype annotations
        html.Should().Contain("Name");
        html.Should().Contain("string");
        html.Should().Contain("Age");
        html.Should().Contain("int32");
        html.Should().Contain("Salary");
        html.Should().Contain("float64");

        // Data values
        html.Should().Contain("Alice");
        html.Should().Contain("Bob");
        html.Should().Contain("25");
        html.Should().Contain("50,000"); // formatted with N0

        // Row count footer
        html.Should().Contain("3 rows");
        html.Should().Contain("3 columns");

        // Styling
        html.Should().Contain("font-family");
        html.Should().Contain("border");
    }

    [Fact]
    public void FormatDataFrame_NullsRenderedWithStyle()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );

        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("null");
        html.Should().Contain("italic"); // null styling
    }

    [Fact]
    public void FormatDataFrame_TruncatesLargeDataFrame()
    {
        var values = Enumerable.Range(0, 100).ToArray();
        var df = new DataFrame(new Column<int>("X", values));

        DataFrameKernelExtension.MaxRows = 25;
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("100 rows");
        html.Should().Contain("showing first 25");
        html.Should().Contain("..."); // truncation indicator
    }

    [Fact]
    public void FormatDataFrame_TruncatesWideDataFrame()
    {
        var cols = new List<IColumn>();
        for (int i = 0; i < 30; i++)
            cols.Add(new Column<int>($"Col{i}", [1]));
        var df = new DataFrame(cols);

        DataFrameKernelExtension.MaxColumns = 20;
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("30 columns");
    }

    [Fact]
    public void FormatDataFrame_AlternatingRowColors()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("#ffffff"); // even rows
        html.Should().Contain("#f8f9fa"); // odd rows
    }

    [Fact]
    public void FormatDataFrame_NumericRightAligned()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25])
        );
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        // Name should be left-aligned, Age should be right-aligned
        html.Should().Contain("text-align:left");
        html.Should().Contain("text-align:right");
    }

    [Fact]
    public void FormatDataFrame_EmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", System.Array.Empty<int>())
        );
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("0 rows");
        html.Should().Contain("<table");
    }

    [Fact]
    public void FormatDataFrame_DateTimeFormatting()
    {
        var df = new DataFrame(
            new Column<DateTime>("Date", [new DateTime(2024, 1, 15), new DateTime(2024, 6, 30, 14, 30, 0)])
        );
        var html = DataFrameKernelExtension.FormatDataFrame(df);

        html.Should().Contain("2024-01-15"); // date-only: no time
        html.Should().Contain("2024-06-30 14:30:00"); // with time
    }
}
