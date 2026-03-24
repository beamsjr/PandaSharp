using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class UniqueTests
{
    [Fact]
    public void Unique_Int_ReturnsDistinct()
    {
        var col = new Column<int>("X", [1, 2, 2, 3, 1, 3, 4]);
        var result = col.Unique();
        result.Length.Should().Be(4);
    }

    [Fact]
    public void Unique_Double()
    {
        var col = new Column<double>("X", [1.0, 1.0, 2.0, 2.0]);
        var result = col.Unique();
        result.Length.Should().Be(2);
    }

    [Fact]
    public void Unique_String()
    {
        var col = new StringColumn("S", ["a", "b", "a", "c", "b"]);
        var result = col.Unique();
        result.Length.Should().Be(3);
    }

    [Fact]
    public void Unique_WithNulls_ExcludesNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 2, null, 1]);
        var result = col.Unique();
        result.Length.Should().Be(2); // 1 and 2, nulls excluded
    }

    [Fact]
    public void Unique_AllSame_ReturnsSingle()
    {
        var col = new Column<int>("X", [5, 5, 5, 5]);
        col.Unique().Length.Should().Be(1);
    }

    [Fact]
    public void Unique_Empty_ReturnsEmpty()
    {
        var col = new Column<int>("X", Array.Empty<int>());
        col.Unique().Length.Should().Be(0);
    }

    [Fact]
    public void Unique_StringWithNulls()
    {
        var col = new StringColumn("S", ["a", null, "b", null, "a"]);
        var result = col.Unique();
        result.Length.Should().Be(2); // "a" and "b"
    }
}
