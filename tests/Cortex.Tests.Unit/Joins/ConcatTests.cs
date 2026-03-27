using FluentAssertions;
using Cortex.Column;
using Cortex.Concat;

namespace Cortex.Tests.Unit.Joins;

public class ConcatTests
{
    [Fact]
    public void Concat_Rows_CombinesDataFrames()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2]),
            new StringColumn("B", ["x", "y"])
        );
        var df2 = new DataFrame(
            new Column<int>("A", [3, 4]),
            new StringColumn("B", ["z", "w"])
        );

        var result = ConcatExtensions.Concat(df1, df2);

        result.RowCount.Should().Be(4);
        result.GetColumn<int>("A")[2].Should().Be(3);
        result.GetStringColumn("B")[3].Should().Be("w");
    }

    [Fact]
    public void Concat_Rows_MissingColumnsFilledWithNull()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2]),
            new StringColumn("B", ["x", "y"])
        );
        var df2 = new DataFrame(
            new Column<int>("A", [3]),
            new StringColumn("C", ["z"])
        );

        var result = ConcatExtensions.Concat(df1, df2);

        result.RowCount.Should().Be(3);
        result.ColumnNames.Should().Equal(["A", "B", "C"]);

        // df2 didn't have B, so row 2 should be null
        result.GetStringColumn("B")[2].Should().BeNull();
        // df1 didn't have C, so rows 0,1 should be null
        result.GetStringColumn("C")[0].Should().BeNull();
    }

    [Fact]
    public void Concat_Columns_CombinesSideBySide()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new StringColumn("B", ["x", "y"]));

        var result = ConcatExtensions.Concat(1, df1, df2);

        result.RowCount.Should().Be(2);
        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["A", "B"]);
    }

    [Fact]
    public void Concat_Columns_DuplicateNamesGetSuffix()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new Column<int>("A", [3, 4]));

        var result = ConcatExtensions.Concat(1, df1, df2);

        result.ColumnCount.Should().Be(2);
        result.ColumnNames[0].Should().Be("A");
        result.ColumnNames[1].Should().Be("A_1");
    }

    [Fact]
    public void Concat_Columns_MismatchedRows_Throws()
    {
        var df1 = new DataFrame(new Column<int>("A", [1, 2]));
        var df2 = new DataFrame(new Column<int>("B", [3]));

        var act = () => ConcatExtensions.Concat(1, df1, df2);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Concat_Empty_ReturnsEmpty()
    {
        var result = ConcatExtensions.Concat();
        result.ColumnCount.Should().Be(0);
    }

    [Fact]
    public void Concat_Single_ReturnsSame()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2]));
        var result = ConcatExtensions.Concat(df);
        result.RowCount.Should().Be(2);
    }
}
