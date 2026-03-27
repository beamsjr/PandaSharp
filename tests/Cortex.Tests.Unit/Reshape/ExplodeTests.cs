using FluentAssertions;
using Cortex.Column;
using Cortex.Reshape;

namespace Cortex.Tests.Unit.Reshape;

public class ExplodeTests
{
    [Fact]
    public void Explode_SplitsCommaSeparated()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new StringColumn("Tags", ["a,b,c", "x,y"])
        );

        var result = df.Explode("Tags");

        result.RowCount.Should().Be(5); // 3 + 2
        result.GetStringColumn("Tags")[0].Should().Be("a");
        result.GetStringColumn("Tags")[1].Should().Be("b");
        result.GetStringColumn("Tags")[2].Should().Be("c");
        result.GetStringColumn("Tags")[3].Should().Be("x");
        result.GetStringColumn("Tags")[4].Should().Be("y");

        // Name should be repeated
        result.GetStringColumn("Name")[0].Should().Be("Alice");
        result.GetStringColumn("Name")[2].Should().Be("Alice");
        result.GetStringColumn("Name")[3].Should().Be("Bob");
    }

    [Fact]
    public void Explode_HandlesNulls()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["A", "B"]),
            new StringColumn("Values", ["1,2", null])
        );

        var result = df.Explode("Values");

        result.RowCount.Should().Be(3); // 2 + 1 (null stays as 1 row)
        result.GetStringColumn("Values")[2].Should().BeNull();
    }

    [Fact]
    public void Explode_CustomSeparator()
    {
        var df = new DataFrame(
            new StringColumn("Data", ["a|b|c"])
        );

        var result = df.Explode("Data", separator: "|");

        result.RowCount.Should().Be(3);
        result.GetStringColumn("Data")[1].Should().Be("b");
    }

    [Fact]
    public void Explode_PreservesNumericColumns()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2]),
            new StringColumn("Items", ["a,b", "c"])
        );

        var result = df.Explode("Items");

        result.RowCount.Should().Be(3);
        result.GetColumn<int>("Id")[0].Should().Be(1);
        result.GetColumn<int>("Id")[1].Should().Be(1);
        result.GetColumn<int>("Id")[2].Should().Be(2);
    }
}
