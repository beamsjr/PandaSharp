using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameCombineTests
{
    [Fact]
    public void Combine_AddsTwoDataFrames()
    {
        var df1 = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<double>("B", [10.0, 20.0, 30.0])
        );
        var df2 = new DataFrame(
            new Column<double>("A", [0.5, 0.5, 0.5]),
            new Column<double>("B", [1.0, 1.0, 1.0])
        );

        var result = df1.Combine(df2, (a, b) => a + b);

        result.GetColumn<double>("A")[0].Should().Be(1.5);
        result.GetColumn<double>("B")[2].Should().Be(31.0);
    }

    [Fact]
    public void Combine_MaxOfTwo()
    {
        var df1 = new DataFrame(new Column<double>("X", [1.0, 5.0, 3.0]));
        var df2 = new DataFrame(new Column<double>("X", [4.0, 2.0, 6.0]));

        var result = df1.Combine(df2, (a, b) =>
            a.HasValue && b.HasValue ? Math.Max(a.Value, b.Value) : a ?? b);

        result.GetColumn<double>("X")[0].Should().Be(4.0);
        result.GetColumn<double>("X")[1].Should().Be(5.0);
        result.GetColumn<double>("X")[2].Should().Be(6.0);
    }

    [Fact]
    public void Combine_WithNulls()
    {
        var df1 = new DataFrame(Column<double>.FromNullable("A", [1.0, null, 3.0]));
        var df2 = new DataFrame(new Column<double>("A", [10.0, 20.0, 30.0]));

        var result = df1.Combine(df2, (a, b) => a.HasValue && b.HasValue ? a + b : null);

        result.GetColumn<double>("A")[0].Should().Be(11.0);
        result["A"].IsNull(1).Should().BeTrue();
        result.GetColumn<double>("A")[2].Should().Be(33.0);
    }

    [Fact]
    public void Combine_MismatchedRows_Throws()
    {
        var df1 = new DataFrame(new Column<double>("A", [1.0, 2.0]));
        var df2 = new DataFrame(new Column<double>("A", [1.0]));

        var act = () => df1.Combine(df2, (a, b) => a + b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Combine_NonNumericColumnsPreserved()
    {
        var df1 = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<double>("Score", [90.0, 80.0])
        );
        var df2 = new DataFrame(
            new StringColumn("Name", ["X", "Y"]),
            new Column<double>("Score", [5.0, 10.0])
        );

        var result = df1.Combine(df2, (a, b) => a + b);

        result.GetStringColumn("Name")[0].Should().Be("Alice"); // left preserved
        result.GetColumn<double>("Score")[0].Should().Be(95.0);
    }
}
