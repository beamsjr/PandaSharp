using FluentAssertions;
using Cortex.Column;
using Cortex.Index;

namespace Cortex.Tests.Unit.Indexing;

public class MultiIndexTests
{
    [Fact]
    public void MultiIndex_FromArrays()
    {
        var mi = MultiIndex.FromArrays(
            ("Region", new StringColumn("Region", ["East", "East", "West", "West"])),
            ("Year", new Column<int>("Year", [2023, 2024, 2023, 2024]))
        );

        mi.NLevels.Should().Be(2);
        mi.Length.Should().Be(4);
        mi.Names.Should().Equal(["Region", "Year"]);
    }

    [Fact]
    public void MultiIndex_GetLevel()
    {
        var mi = MultiIndex.FromArrays(
            ("A", new StringColumn("A", ["x", "y", "x"])),
            ("B", new Column<int>("B", [1, 2, 3]))
        );

        mi.GetLevel(0, 0).Should().Be("x");
        mi.GetLevel(1, 2).Should().Be(3);
    }

    [Fact]
    public void MultiIndex_GetLocations()
    {
        var mi = MultiIndex.FromArrays(
            ("Key", new StringColumn("Key", ["A", "B", "A", "C", "A"]))
        );

        var locs = mi.GetLocations(0, "A");
        locs.Should().Equal([0, 2, 4]);
    }

    [Fact]
    public void DataFrame_SetIndex_MultiColumn()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "West", "East"]),
            new Column<int>("Year", [2023, 2023, 2024]),
            new Column<double>("Sales", [100, 200, 150])
        );

        var indexed = df.SetIndex("Region", "Year");

        indexed.ColumnNames.Should().NotContain("Region");
        indexed.ColumnNames.Should().NotContain("Year");
        indexed.ColumnNames.Should().Contain("Sales");
        indexed.MultiIndex.Should().NotBeNull();
        indexed.MultiIndex!.NLevels.Should().Be(2);
    }

    [Fact]
    public void DataFrame_ResetIndex_MultiColumn()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "West"]),
            new Column<int>("Year", [2023, 2024]),
            new Column<double>("Sales", [100, 200])
        );

        var reset = df.SetIndex("Region", "Year").ResetIndex();

        reset.ColumnNames.Should().Contain("Region");
        reset.ColumnNames.Should().Contain("Year");
        reset.ColumnNames.Should().Contain("Sales");
        reset.RowCount.Should().Be(2);
    }

    [Fact]
    public void At_ScalarAccess()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        df.At[0, "Name"].Should().Be("Alice");
        df.At[1, "Age"].Should().Be(30);
    }

    [Fact]
    public void At_OutOfRange_Throws()
    {
        var df = new DataFrame(new Column<int>("X", [1]));
        var act = () => df.At[5, "X"];
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void IAt_ScalarAccess()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        df.IAt[0, 0].Should().Be("Alice"); // row 0, col 0
        df.IAt[1, 1].Should().Be(30);      // row 1, col 1
    }

    [Fact]
    public void IAt_OutOfRange_Throws()
    {
        var df = new DataFrame(new Column<int>("X", [1]));
        var act = () => df.IAt[0, 5];
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void Xs_ByColumn()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "West", "East"]),
            new Column<int>("Year", [2023, 2023, 2024]),
            new Column<double>("Sales", [100, 200, 150])
        );

        var result = df.Xs("Region", "East");

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().NotContain("Region");
        result.ColumnNames.Should().Contain("Year");
        result.ColumnNames.Should().Contain("Sales");
    }

    [Fact]
    public void Xs_ByIndex()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice"]),
            new Column<double>("Score", [90, 85, 95])
        );

        var indexed = df.SetIndex("Name");
        var result = indexed.Xs("Alice");

        result.RowCount.Should().Be(2);
        result.GetColumn<double>("Score")[0].Should().Be(90);
        result.GetColumn<double>("Score")[1].Should().Be(95);
    }

    [Fact]
    public void Xs_NoIndex_Throws()
    {
        var df = new DataFrame(new Column<int>("X", [1]));
        var act = () => df.Xs("value");
        act.Should().Throw<InvalidOperationException>();
    }
}
