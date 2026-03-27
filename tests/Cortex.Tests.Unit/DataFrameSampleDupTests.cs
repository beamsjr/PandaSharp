using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameSampleDupTests
{
    // -- SampleFrac --

    [Fact]
    public void SampleFrac_Half()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var result = df.SampleFrac(0.5, seed: 42);
        result.RowCount.Should().Be(50);
    }

    [Fact]
    public void SampleFrac_Zero()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        df.SampleFrac(0.0).RowCount.Should().Be(0);
    }

    [Fact]
    public void SampleFrac_One()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        df.SampleFrac(1.0).RowCount.Should().Be(3);
    }

    [Fact]
    public void SampleFrac_OutOfRange_Throws()
    {
        var df = new DataFrame(new Column<int>("X", [1]));
        var act = () => df.SampleFrac(1.5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // -- Duplicated --

    [Fact]
    public void Duplicated_ReturnsMask()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice", "Charlie", "Bob"]),
            new Column<int>("Age", [25, 30, 25, 35, 30])
        );

        var mask = df.Duplicated();
        mask.Should().Equal([false, false, true, false, true]);
    }

    [Fact]
    public void Duplicated_Subset()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice"]),
            new Column<int>("Age", [25, 30, 99]) // different age but same name
        );

        var mask = df.Duplicated("Name");
        mask.Should().Equal([false, false, true]); // third row is dup by Name
    }

    [Fact]
    public void Duplicated_NoDups()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        df.Duplicated().Should().Equal([false, false, false]);
    }

    [Fact]
    public void Duplicated_CombinedWithFilter()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice"]),
            new Column<int>("Val", [1, 2, 1])
        );

        // Keep only non-duplicates (same as DropDuplicates but via mask)
        var mask = df.Duplicated().Not();
        var unique = df.Filter(mask);
        unique.RowCount.Should().Be(2);
    }

    // -- ToRecords --

    [Fact]
    public void ToRecords_ReturnsObjectArrays()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var records = df.ToRecords();
        records.Length.Should().Be(2);
        records[0][0].Should().Be("Alice");
        records[0][1].Should().Be(25);
        records[1][0].Should().Be("Bob");
    }

    [Fact]
    public void ToRecords_WithNulls()
    {
        var df = new DataFrame(Column<int>.FromNullable("X", [1, null, 3]));
        var records = df.ToRecords();
        records[1][0].Should().BeNull();
    }
}
