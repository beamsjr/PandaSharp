using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Storage;

namespace PandaSharp.Tests.Unit.Storage;

public class MutableDataFrameTests
{
    [Fact]
    public void ToMutable_CreatesWritableView()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var mdf = df.ToMutable();
        mdf.RowCount.Should().Be(3);
        mdf.ColumnCount.Should().Be(2);
    }

    [Fact]
    public void SetValue_ModifiesValueWithoutAffectingOriginal()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var mdf = df.ToMutable();

        mdf.SetValue<int>("A", 1, 99);

        // Original unchanged
        df.GetColumn<int>("A")[1].Should().Be(2);

        // Mutable changed
        var result = mdf.ToDataFrame();
        result.GetColumn<int>("A")[1].Should().Be(99);
    }

    [Fact]
    public void SetValue_ToNull()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var mdf = df.ToMutable();

        mdf.SetValue<int>("A", 1, null);

        var result = mdf.ToDataFrame();
        result["A"].IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void SetStringValue_ModifiesString()
    {
        var df = new DataFrame(new StringColumn("Name", ["Alice", "Bob"]));
        var mdf = df.ToMutable();

        mdf.SetStringValue("Name", 0, "Charlie");

        var result = mdf.ToDataFrame();
        result.GetStringColumn("Name")[0].Should().Be("Charlie");

        // Original unchanged
        df.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void SetValue_ThrowsOnMissingColumn()
    {
        var df = new DataFrame(new Column<int>("A", [1]));
        var mdf = df.ToMutable();

        var act = () => mdf.SetValue<int>("Missing", 0, 1);
        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void SetValue_ThrowsOnOutOfRange()
    {
        var df = new DataFrame(new Column<int>("A", [1]));
        var mdf = df.ToMutable();

        var act = () => mdf.SetValue<int>("A", 5, 1);
        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void ToDataFrame_Freezes()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2]));
        var mdf = df.ToMutable();
        mdf.SetValue<int>("A", 0, 10);

        var frozen = mdf.ToDataFrame();
        frozen.GetColumn<int>("A")[0].Should().Be(10);
        frozen.GetColumn<int>("A")[1].Should().Be(2);
    }
}
