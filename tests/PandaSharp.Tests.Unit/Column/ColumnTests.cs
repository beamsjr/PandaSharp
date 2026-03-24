using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class ColumnTests
{
    [Fact]
    public void Constructor_WithArray_CreatesColumnWithCorrectLength()
    {
        var col = new Column<int>("Age", [25, 30, 35]);

        col.Name.Should().Be("Age");
        col.Length.Should().Be(3);
        col.NullCount.Should().Be(0);
        col.DataType.Should().Be(typeof(int));
    }

    [Fact]
    public void Indexer_ReturnsCorrectValues()
    {
        var col = new Column<int>("Age", [25, 30, 35]);

        col[0].Should().Be(25);
        col[1].Should().Be(30);
        col[2].Should().Be(35);
    }

    [Fact]
    public void Constructor_WithNullables_TracksNulls()
    {
        var col = Column<int>.FromNullable("Age", [25, null, 35]);

        col.Length.Should().Be(3);
        col.NullCount.Should().Be(1);
        col[0].Should().Be(25);
        col[1].Should().BeNull();
        col[2].Should().Be(35);
        col.IsNull(0).Should().BeFalse();
        col.IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void Slice_ReturnsSubset()
    {
        var col = new Column<int>("Age", [10, 20, 30, 40, 50]);
        var sliced = (Column<int>)col.Slice(1, 3);

        sliced.Length.Should().Be(3);
        sliced[0].Should().Be(20);
        sliced[1].Should().Be(30);
        sliced[2].Should().Be(40);
    }

    [Fact]
    public void Filter_ReturnsMaskedRows()
    {
        var col = new Column<int>("Age", [10, 20, 30, 40, 50]);
        bool[] mask = [true, false, true, false, true];

        var filtered = (Column<int>)col.Filter(mask);

        filtered.Length.Should().Be(3);
        filtered[0].Should().Be(10);
        filtered[1].Should().Be(30);
        filtered[2].Should().Be(50);
    }

    [Fact]
    public void TakeRows_ReturnsSpecifiedRows()
    {
        var col = new Column<int>("Age", [10, 20, 30, 40, 50]);
        int[] indices = [4, 2, 0];

        var taken = (Column<int>)col.TakeRows(indices);

        taken.Length.Should().Be(3);
        taken[0].Should().Be(50);
        taken[1].Should().Be(30);
        taken[2].Should().Be(10);
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        var col = new Column<int>("Age", [10, 20]);
        var clone = (Column<int>)col.Clone("NewName");

        clone.Name.Should().Be("NewName");
        clone[0].Should().Be(10);
    }

    [Fact]
    public void GetObject_ReturnsBoxedValueOrNull()
    {
        var col = Column<int>.FromNullable("Age", [25, null, 35]);

        col.GetObject(0).Should().Be(25);
        col.GetObject(1).Should().BeNull();
        col.GetObject(2).Should().Be(35);
    }

    [Fact]
    public void Count_ExcludesNulls()
    {
        var col = Column<int>.FromNullable("Age", [25, null, 35, null]);
        col.Count().Should().Be(2);
    }
}
