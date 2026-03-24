using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Accessors;

public class StringSplitTests
{
    [Fact]
    public void Split_ReturnsDataFrameOfParts()
    {
        var col = new StringColumn("Full", ["Alice Smith", "Bob Jones", "Charlie Brown"]);
        var result = col.Str.Split(" ");

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["Full_0", "Full_1"]);
        result.GetStringColumn("Full_0")[0].Should().Be("Alice");
        result.GetStringColumn("Full_1")[0].Should().Be("Smith");
    }

    [Fact]
    public void Split_UnevenParts_PadsWithNull()
    {
        var col = new StringColumn("Data", ["a,b,c", "x,y"]);
        var result = col.Str.Split(",");

        result.ColumnCount.Should().Be(3); // max parts = 3
        result.GetStringColumn("Data_2")[0].Should().Be("c");
        result.GetStringColumn("Data_2")[1].Should().BeNull(); // "x,y" only has 2 parts
    }

    [Fact]
    public void Split_WithNulls()
    {
        var col = new StringColumn("Data", ["a-b", null, "c-d"]);
        var result = col.Str.Split("-");

        result.ColumnCount.Should().Be(2);
        result.GetStringColumn("Data_0")[1].Should().BeNull();
    }

    [Fact]
    public void Split_MaxParts()
    {
        var col = new StringColumn("Data", ["a,b,c,d"]);
        var result = col.Str.Split(",", maxParts: 2);

        result.ColumnCount.Should().Be(2);
        result.GetStringColumn("Data_0")[0].Should().Be("a");
        result.GetStringColumn("Data_1")[0].Should().Be("b,c,d"); // rest unsplit
    }
}
