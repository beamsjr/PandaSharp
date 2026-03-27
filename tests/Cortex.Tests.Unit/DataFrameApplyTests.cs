using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameApplyTests
{
    [Fact]
    public void Apply_Numeric_CreatesNewColumn()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var result = df.Apply(row => (double)row.Get<int>("Age")! * 2, "DoubleAge");

        result.ColumnNames.Should().Contain("DoubleAge");
        result.GetColumn<double>("DoubleAge")[0].Should().Be(50.0);
        result.GetColumn<double>("DoubleAge")[1].Should().Be(60.0);
    }

    [Fact]
    public void Apply_String_CreatesNewColumn()
    {
        var df = new DataFrame(
            new StringColumn("First", ["Alice", "Bob"]),
            new StringColumn("Last", ["Smith", "Jones"])
        );

        var result = df.Apply(row => $"{row.GetString("First")} {row.GetString("Last")}", "Full");

        result.GetStringColumn("Full")[0].Should().Be("Alice Smith");
        result.GetStringColumn("Full")[1].Should().Be("Bob Jones");
    }

    [Fact]
    public void Apply_WithNulls_HandlesGracefully()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );

        // Accessing null will throw, Apply catches and returns null
        var result = df.Apply(row => row.Get<int>("A")!.Value * 10, "B");

        result.GetColumn<int>("B")[0].Should().Be(10);
        result["B"].IsNull(1).Should().BeTrue();
        result.GetColumn<int>("B")[2].Should().Be(30);
    }

    [Fact]
    public void Apply_OriginalUnchanged()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2]));
        var result = df.Apply<int>(row => row.Get<int>("X")!.Value + 100, "Y");

        df.ColumnCount.Should().Be(1);
        result.ColumnCount.Should().Be(2);
    }
}
