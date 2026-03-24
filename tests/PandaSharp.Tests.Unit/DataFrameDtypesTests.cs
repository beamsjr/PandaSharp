using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameDtypesTests
{
    [Fact]
    public void Dtypes_ReturnsCorrectTypes()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25]),
            new Column<double>("Score", [95.0]),
            new Column<bool>("Active", [true])
        );

        var dtypes = df.Dtypes;

        dtypes["Name"].Should().Be(typeof(string));
        dtypes["Age"].Should().Be(typeof(int));
        dtypes["Score"].Should().Be(typeof(double));
        dtypes["Active"].Should().Be(typeof(bool));
    }

    [Fact]
    public void Dtypes_EmptyDataFrame()
    {
        var df = new DataFrame();
        df.Dtypes.Count.Should().Be(0);
    }

    [Fact]
    public void Dtypes_CategoricalColumn()
    {
        var df = new DataFrame(
            new CategoricalColumn("Cat", ["a", "b", "a"])
        );
        df.Dtypes["Cat"].Should().Be(typeof(string));
    }
}
