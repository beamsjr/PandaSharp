using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameConstructionTests
{
    [Fact]
    public void FromDictionary_CreatesDataFrame()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = new string?[] { "Alice", "Bob", "Charlie" },
            ["Age"] = new int[] { 25, 30, 35 },
            ["Score"] = new double[] { 90.5, 85.0, 92.3 }
        });

        df.RowCount.Should().Be(3);
        df.ColumnNames.Should().Contain("Name");
        df.ColumnNames.Should().Contain("Age");
        df.ColumnNames.Should().Contain("Score");
        df.GetStringColumn("Name")[0].Should().Be("Alice");
        df.GetColumn<int>("Age")[1].Should().Be(30);
    }

    [Fact]
    public void FromDictionary_BoolColumn()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Active"] = new bool[] { true, false, true }
        });

        df.GetColumn<bool>("Active")[0].Should().Be(true);
    }

    [Fact]
    public void FromDictionary_UnsupportedType_Throws()
    {
        var act = () => DataFrame.FromDictionary(new()
        {
            ["X"] = new decimal[] { 1.0m }
        });
        act.Should().Throw<ArgumentException>();
    }
}
