using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameIndexTests
{
    [Fact]
    public void SetIndex_RemovesColumnAndSetsIndex()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        var indexed = df.SetIndex("Name");

        indexed.ColumnNames.Should().NotContain("Name");
        indexed.ColumnCount.Should().Be(1); // only Age
        indexed.IndexName.Should().Be("Name");
    }

    [Fact]
    public void ResetIndex_RestoresColumn()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var indexed = df.SetIndex("Name");
        var reset = indexed.ResetIndex();

        reset.ColumnNames.Should().Contain("Name");
        reset.ColumnNames.Should().Contain("Age");
        reset.ColumnCount.Should().Be(2);
        reset.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void ResetIndex_NoIndex_ReturnsSame()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2]));
        var result = df.ResetIndex();
        result.ColumnCount.Should().Be(1);
    }

    [Fact]
    public void SetIndex_PreservesData()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["x", "y"]),
            new Column<int>("Val", [10, 20])
        );

        var indexed = df.SetIndex("Id");
        indexed.GetColumn<int>("Val")[0].Should().Be(10);
        indexed.GetColumn<int>("Val")[1].Should().Be(20);
    }
}
