using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class StringColumnTests
{
    [Fact]
    public void Constructor_SetsProperties()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie"]);

        col.Name.Should().Be("Name");
        col.Length.Should().Be(3);
        col.NullCount.Should().Be(0);
        col.DataType.Should().Be(typeof(string));
    }

    [Fact]
    public void Indexer_ReturnsValues()
    {
        var col = new StringColumn("Name", ["Alice", "Bob"]);

        col[0].Should().Be("Alice");
        col[1].Should().Be("Bob");
    }

    [Fact]
    public void NullTracking_Works()
    {
        var col = new StringColumn("Name", ["Alice", null, "Charlie"]);

        col.NullCount.Should().Be(1);
        col.IsNull(0).Should().BeFalse();
        col.IsNull(1).Should().BeTrue();
        col[1].Should().BeNull();
    }

    [Fact]
    public void Filter_ReturnsMaskedRows()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie"]);
        bool[] mask = [true, false, true];

        var filtered = (StringColumn)col.Filter(mask);

        filtered.Length.Should().Be(2);
        filtered[0].Should().Be("Alice");
        filtered[1].Should().Be("Charlie");
    }

    [Fact]
    public void Slice_ReturnsSubset()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana"]);
        var sliced = (StringColumn)col.Slice(1, 2);

        sliced.Length.Should().Be(2);
        sliced[0].Should().Be("Bob");
        sliced[1].Should().Be("Charlie");
    }

    [Fact]
    public void Contains_ReturnsMask()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie"]);
        var mask = col.Contains("li");

        mask.Should().Equal([true, false, true]);
    }

    [Fact]
    public void Eq_ReturnsMask()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Alice"]);
        var mask = col.Eq("Alice");

        mask.Should().Equal([true, false, true]);
    }

    [Fact]
    public void TakeRows_ReturnsSpecifiedRows()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie"]);
        int[] indices = [2, 0];

        var taken = (StringColumn)col.TakeRows(indices);

        taken.Length.Should().Be(2);
        taken[0].Should().Be("Charlie");
        taken[1].Should().Be("Alice");
    }
}
