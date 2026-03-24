using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameMultiSortTests
{
    [Fact]
    public void Sort_MultiColumn_PrimaryThenSecondary()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["Sales", "Eng", "Sales", "Eng"]),
            new Column<int>("Age", [30, 25, 25, 35]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana"])
        );

        var sorted = df.Sort(("Dept", true), ("Age", true));

        // Eng before Sales (alpha), then by Age asc within each
        sorted.GetStringColumn("Dept")[0].Should().Be("Eng");
        sorted.GetColumn<int>("Age")[0].Should().Be(25); // Bob
        sorted.GetColumn<int>("Age")[1].Should().Be(35); // Diana
        sorted.GetStringColumn("Dept")[2].Should().Be("Sales");
        sorted.GetColumn<int>("Age")[2].Should().Be(25); // Charlie
        sorted.GetColumn<int>("Age")[3].Should().Be(30); // Alice
    }

    [Fact]
    public void Sort_MultiColumn_MixedDirection()
    {
        var df = new DataFrame(
            new StringColumn("Group", ["A", "A", "B", "B"]),
            new Column<int>("Value", [1, 2, 1, 2])
        );

        var sorted = df.Sort(("Group", true), ("Value", false));

        sorted.GetStringColumn("Group")[0].Should().Be("A");
        sorted.GetColumn<int>("Value")[0].Should().Be(2); // A, desc
        sorted.GetColumn<int>("Value")[1].Should().Be(1);
        sorted.GetStringColumn("Group")[2].Should().Be("B");
        sorted.GetColumn<int>("Value")[2].Should().Be(2); // B, desc
    }
}
