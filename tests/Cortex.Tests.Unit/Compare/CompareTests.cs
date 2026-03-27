using FluentAssertions;
using Cortex.Column;
// DataFrameCompareExtensions is now in Cortex namespace

namespace Cortex.Tests.Unit.Compare;

public class CompareTests
{
    [Fact]
    public void Compare_IdenticalDataFrames_ReturnsEmpty()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var df2 = new DataFrame(new Column<int>("A", [1, 2, 3]));

        var result = df1.Compare(df2);
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Compare_DifferentValues_ReturnsDiffRows()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );
        var df2 = new DataFrame(
            new Column<int>("A", [1, 99, 3]),
            new StringColumn("B", ["x", "y", "changed"])
        );

        var result = df1.Compare(df2);
        result.RowCount.Should().Be(2); // rows 1 and 2 differ
        result.GetColumn<int>("row")[0].Should().Be(1);
        result.GetColumn<int>("row")[1].Should().Be(2);
    }

    [Fact]
    public void Compare_DifferentLengths_FlagsExtraRows()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2, 3, 4]));
        var df2 = new DataFrame(new Column<int>("A", [1, 2]));

        var result = df1.Compare(df2);
        // Rows 2 and 3 are extra in df1
        result.RowCount.Should().BeGreaterThanOrEqualTo(2);
    }

    [Fact]
    public void Compare_ShowsSelfAndOtherValues()
    {
        var df1 = new DataFrame(new StringColumn("Name", ["Alice", "Bob"]));
        var df2 = new DataFrame(new StringColumn("Name", ["Alice", "Charlie"]));

        var result = df1.Compare(df2);
        result.ColumnNames.Should().Contain("Name_self");
        result.ColumnNames.Should().Contain("Name_other");
        result.GetStringColumn("Name_self")[0].Should().Be("Bob");
        result.GetStringColumn("Name_other")[0].Should().Be("Charlie");
    }
}
