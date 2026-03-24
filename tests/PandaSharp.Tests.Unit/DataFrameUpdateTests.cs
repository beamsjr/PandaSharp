using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameUpdateTests
{
    [Fact]
    public void UpdateColumn_TransformsInPlace()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var result = df.UpdateColumn("Age", col =>
        {
            var ages = (Column<int>)col;
            return ages.Add(5);
        });

        result.GetColumn<int>("Age")[0].Should().Be(30);
        result.GetColumn<int>("Age")[1].Should().Be(35);
        result.GetStringColumn("Name")[0].Should().Be("Alice"); // untouched
    }

    [Fact]
    public void UpdateColumn_OriginalUnchanged()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var updated = df.UpdateColumn("X", col => ((Column<int>)col).Add(10));

        df.GetColumn<int>("X")[0].Should().Be(1); // original
        updated.GetColumn<int>("X")[0].Should().Be(11); // updated
    }

    [Fact]
    public void ReplaceColumn_SwapsColumn()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var newA = new Column<double>("A", [10.5, 20.5, 30.5]);
        var result = df.ReplaceColumn("A", newA);

        result["A"].DataType.Should().Be(typeof(double));
        result.GetColumn<double>("A")[0].Should().Be(10.5);
    }

    [Fact]
    public void ReplaceColumn_WrongLength_Throws()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var act = () => df.ReplaceColumn("A", new Column<int>("A", [1, 2]));
        act.Should().Throw<ArgumentException>();
    }
}
