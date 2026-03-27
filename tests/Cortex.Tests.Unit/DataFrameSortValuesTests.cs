using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameSortValuesTests
{
    [Fact]
    public void SortValues_SingleColumn_Works()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Charlie", "Alice", "Bob"]),
            new Column<int>("Age", [35, 25, 30])
        );

        var result = df.SortValues("Age");
        result.GetStringColumn("Name")[0].Should().Be("Alice");
        result.GetStringColumn("Name")[2].Should().Be("Charlie");
    }

    [Fact]
    public void SortValues_MultiColumn_Works()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["B", "A", "A", "B"]),
            new Column<int>("Age", [30, 25, 35, 20])
        );

        var result = df.SortValues(("Dept", true), ("Age", true));
        result.GetStringColumn("Dept")[0].Should().Be("A");
        result.GetColumn<int>("Age")[0].Should().Be(25);
    }

    [Fact]
    public void SortValues_Descending()
    {
        var df = new DataFrame(new Column<int>("X", [1, 3, 2]));
        var result = df.SortValues("X", ascending: false);
        result.GetColumn<int>("X")[0].Should().Be(3);
    }
}
