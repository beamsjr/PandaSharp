using FluentAssertions;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class HtmlTableTests
{
    [Fact]
    public void ReadHtml_SimpleTable()
    {
        var html = """
        <html><body>
        <table>
            <tr><th>Name</th><th>Age</th></tr>
            <tr><td>Alice</td><td>25</td></tr>
            <tr><td>Bob</td><td>30</td></tr>
        </table>
        </body></html>
        """;

        var tables = HtmlTableReader.ReadHtml(html);
        tables.Should().HaveCount(1);

        var df = tables[0];
        df.RowCount.Should().Be(2);
        df.ColumnNames.Should().Contain("Name");
        df.ColumnNames.Should().Contain("Age");
        df.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void ReadHtml_MultipleTables()
    {
        var html = """
        <table><tr><th>A</th></tr><tr><td>1</td></tr></table>
        <table><tr><th>B</th></tr><tr><td>2</td></tr></table>
        """;

        var tables = HtmlTableReader.ReadHtml(html);
        tables.Should().HaveCount(2);
    }

    [Fact]
    public void ReadHtml_SpecificTableIndex()
    {
        var html = """
        <table><tr><th>First</th></tr><tr><td>1</td></tr></table>
        <table><tr><th>Second</th></tr><tr><td>2</td></tr></table>
        """;

        var tables = HtmlTableReader.ReadHtml(html, tableIndex: 1);
        tables.Should().HaveCount(1);
        tables[0].ColumnNames.Should().Contain("Second");
    }

    [Fact]
    public void ReadHtml_NumericInference()
    {
        var html = """
        <table>
            <tr><th>Value</th><th>Name</th></tr>
            <tr><td>100</td><td>Alice</td></tr>
            <tr><td>200</td><td>Bob</td></tr>
        </table>
        """;

        var df = HtmlTableReader.ReadHtml(html)[0];
        df["Value"].DataType.Should().Be(typeof(int));
        df["Name"].DataType.Should().Be(typeof(string));
    }

    [Fact]
    public void ReadHtml_EmptyTable()
    {
        var html = "<table></table>";
        var tables = HtmlTableReader.ReadHtml(html);
        tables.Should().HaveCount(1);
        tables[0].RowCount.Should().Be(0);
    }
}
