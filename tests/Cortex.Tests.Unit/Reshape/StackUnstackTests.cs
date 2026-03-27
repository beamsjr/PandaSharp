using FluentAssertions;
using Cortex.Column;
using Cortex.Reshape;

namespace Cortex.Tests.Unit.Reshape;

public class StackUnstackTests
{
    [Fact]
    public void Stack_ConvertsWideToLong()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<double>("Math", [90.0, 85.0]),
            new Column<double>("English", [80.0, 95.0])
        );

        var stacked = df.Stack("Name");

        stacked.RowCount.Should().Be(4); // 2 names * 2 subjects
        stacked.ColumnNames.Should().Contain("variable");
        stacked.ColumnNames.Should().Contain("value");
    }

    [Fact]
    public void Unstack_ConvertsLongToWide()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Alice", "Bob", "Bob"]),
            new StringColumn("Subject", ["Math", "Eng", "Math", "Eng"]),
            new Column<double>("Score", [90.0, 80.0, 85.0, 95.0])
        );

        var unstacked = df.Unstack(indexColumn: "Name", columnColumn: "Subject", valueColumn: "Score");

        unstacked.RowCount.Should().Be(2);
        unstacked.ColumnNames.Should().Contain("Math");
        unstacked.ColumnNames.Should().Contain("Eng");
    }

    [Fact]
    public void Stack_Unstack_RoundTrips()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["A", "B"]),
            new Column<double>("X", [1.0, 2.0]),
            new Column<double>("Y", [3.0, 4.0])
        );

        var stacked = df.Stack("Id", varName: "Var", valName: "Val");
        stacked.RowCount.Should().Be(4);

        var unstacked = stacked.Unstack(indexColumn: "Id", columnColumn: "Var", valueColumn: "Val");
        unstacked.RowCount.Should().Be(2);
        unstacked.ColumnNames.Should().Contain("X");
        unstacked.ColumnNames.Should().Contain("Y");
    }
}
