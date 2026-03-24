using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameConversionTests
{
    private DataFrame CreateTestDF() => new(
        new Column<int>("Id", [1, 2, 3]),
        new Column<double>("Value", [1.5, 2.5, 3.5]),
        new StringColumn("Name", ["Alice", "Bob", "Charlie"])
    );

    // ===== ToDictionary =====

    [Fact]
    public void ToDictionary_ListOrient()
    {
        var df = CreateTestDF();
        var dict = df.ToDictionary("list");

        dict.Should().ContainKey("Id");
        dict.Should().ContainKey("Value");
        dict.Should().ContainKey("Name");

        var ids = (object?[])dict["Id"]!;
        ids.Should().HaveCount(3);
        ids[0].Should().Be(1);
    }

    [Fact]
    public void ToDictionary_DictOrient()
    {
        var df = CreateTestDF();
        var dict = df.ToDictionary("dict");

        var idDict = (Dictionary<int, object?>)dict["Id"]!;
        idDict[0].Should().Be(1);
        idDict[2].Should().Be(3);
    }

    [Fact]
    public void ToDictionary_SplitOrient()
    {
        var df = CreateTestDF();
        var dict = df.ToDictionary("split");

        dict.Should().ContainKey("columns");
        dict.Should().ContainKey("data");
        dict.Should().ContainKey("index");

        var columns = (string[])dict["columns"]!;
        columns.Should().Equal(["Id", "Value", "Name"]);

        var data = (object?[][])dict["data"]!;
        data.Should().HaveCount(3);
    }

    [Fact]
    public void ToDictionary_DefaultIsList()
    {
        var df = CreateTestDF();
        var dict = df.ToDictionary();
        dict.Should().ContainKey("Id");
        ((object?[])dict["Id"]!).Should().HaveCount(3);
    }

    [Fact]
    public void ToDictionary_InvalidOrient_Throws()
    {
        var df = CreateTestDF();
        var act = () => df.ToDictionary("invalid");
        act.Should().Throw<ArgumentException>();
    }

    // ===== ToRecordDicts =====

    [Fact]
    public void ToRecordDicts_ReturnsListOfDicts()
    {
        var df = CreateTestDF();
        var records = df.ToRecordDicts();

        records.Should().HaveCount(3);
        records[0]["Name"].Should().Be("Alice");
        records[1]["Id"].Should().Be(2);
    }

    // ===== ToLatex =====

    [Fact]
    public void ToLatex_ContainsTableStructure()
    {
        var df = CreateTestDF();
        var latex = df.ToLatex();

        latex.Should().Contain("\\begin{tabular}");
        latex.Should().Contain("\\end{tabular}");
        latex.Should().Contain("\\toprule");
        latex.Should().Contain("\\midrule");
        latex.Should().Contain("\\bottomrule");
        latex.Should().Contain("Id");
        latex.Should().Contain("Alice");
    }

    [Fact]
    public void ToLatex_TruncatesLargeDataFrame()
    {
        var ids = Enumerable.Range(0, 100).ToArray();
        var df = new DataFrame(new Column<int>("x", ids));
        var latex = df.ToLatex(maxRows: 5);

        latex.Should().Contain("more rows");
    }

    [Fact]
    public void ToLatex_ColumnsSeparatedByAmpersand()
    {
        var df = CreateTestDF();
        var latex = df.ToLatex();
        latex.Should().Contain("&"); // LaTeX column separator
    }

    // ===== Empty DataFrame =====

    [Fact]
    public void ToDictionary_EmptyDataFrame()
    {
        var df = new DataFrame(new Column<int>("x", Array.Empty<int>()));
        var dict = df.ToDictionary();
        ((object?[])dict["x"]!).Should().BeEmpty();
    }

    [Fact]
    public void ToRecordDicts_EmptyDataFrame()
    {
        var df = new DataFrame(new Column<int>("x", Array.Empty<int>()));
        df.ToRecordDicts().Should().BeEmpty();
    }
}
