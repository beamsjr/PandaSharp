using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameWhereTests
{
    [Fact]
    public void Where_FiltersByColumnPredicate()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        var result = df.Where("Age", val => (int)val! > 28);

        result.RowCount.Should().Be(2);
        result.GetStringColumn("Name")[0].Should().Be("Bob");
        result.GetStringColumn("Name")[1].Should().Be("Charlie");
    }

    [Fact]
    public void Where_WithStringColumn()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        var result = df.Where("Name", val => val?.ToString()?.StartsWith("C") == true);

        result.RowCount.Should().Be(1);
        result.GetStringColumn("Name")[0].Should().Be("Charlie");
    }

    [Fact]
    public void Where_WithNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3, null, 5])
        );

        var result = df.Where("A", val => val is not null && (int)val > 2);

        result.RowCount.Should().Be(2);
    }
}
