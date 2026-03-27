using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Joins;
using Cortex.Missing;
using Cortex.Statistics;

namespace Cortex.Tests.Unit;

/// <summary>
/// Edge case tests for robustness: empty DataFrames, single rows, all-nulls, etc.
/// </summary>
public class EdgeCaseTests
{
    // -- Empty DataFrame --

    [Fact]
    public void Empty_DataFrame_HasZeroRows()
    {
        var df = new DataFrame();
        df.RowCount.Should().Be(0);
        df.ColumnCount.Should().Be(0);
        df.ToString().Should().Contain("empty");
    }

    [Fact]
    public void Head_OnEmpty_ReturnsEmpty()
    {
        var df = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        df.Head(5).RowCount.Should().Be(0);
    }

    [Fact]
    public void Tail_OnEmpty_ReturnsEmpty()
    {
        var df = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        df.Tail(5).RowCount.Should().Be(0);
    }

    [Fact]
    public void Filter_AllFalse_ReturnsEmpty()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var result = df.Filter([false, false, false]);
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Filter_AllTrue_ReturnsSame()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var result = df.Filter([true, true, true]);
        result.RowCount.Should().Be(3);
    }

    // -- Single row --

    [Fact]
    public void SingleRow_DataFrame_Works()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25])
        );

        df.RowCount.Should().Be(1);
        df.Head(1).RowCount.Should().Be(1);
        df.Sort("Age").RowCount.Should().Be(1);
        df.DropDuplicates().RowCount.Should().Be(1);
    }

    [Fact]
    public void SingleRow_Aggregation()
    {
        var col = new Column<double>("X", [42.0]);
        col.Sum().Should().Be(42.0);
        col.Mean().Should().Be(42.0);
        col.Min().Should().Be(42.0);
        col.Max().Should().Be(42.0);
        col.Median().Should().Be(42.0);
    }

    // -- All nulls --

    [Fact]
    public void AllNulls_Column_Aggregation()
    {
        var col = Column<double>.FromNullable("X", [null, null, null]);
        col.Sum().Should().Be(0); // sum of no values is 0
        col.Mean().Should().BeNull();
        col.Min().Should().BeNull();
        col.Max().Should().BeNull();
        col.Count().Should().Be(0);
    }

    [Fact]
    public void AllNulls_DropNa_RemovesAll()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [null, null, null])
        );
        df.DropNa().RowCount.Should().Be(0);
    }

    [Fact]
    public void AllNulls_FillNa_FillsAll()
    {
        var col = Column<int>.FromNullable("X", [null, null, null]);
        var filled = col.FillNa(0);
        filled[0].Should().Be(0);
        filled[1].Should().Be(0);
        filled[2].Should().Be(0);
        filled.NullCount.Should().Be(0);
    }

    // -- Large DataFrame --

    [Fact]
    public void LargeDataFrame_BasicOperations()
    {
        int n = 100_000;
        var values = new double[n];
        for (int i = 0; i < n; i++) values[i] = i;

        var df = new DataFrame(new Column<double>("X", values));
        df.RowCount.Should().Be(n);
        df.GetColumn<double>("X").Sum().Should().Be((double)n * (n - 1) / 2);
        df.Head(5).RowCount.Should().Be(5);
        df.Tail(5).RowCount.Should().Be(5);

        // Filter
        var filtered = df.Filter(df.GetColumn<double>("X").Gt(99_990));
        filtered.RowCount.Should().Be(9); // 99991..99999
    }

    // -- Join edge cases --

    [Fact]
    public void Join_EmptyRight_InnerReturnsEmpty()
    {
        var left = new DataFrame(new Column<int>("Id", [1, 2]));
        var right = new DataFrame(new Column<int>("Id", Array.Empty<int>()));
        var result = left.Join(right, "Id");
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Join_EmptyLeft_InnerReturnsEmpty()
    {
        var left = new DataFrame(new Column<int>("Id", Array.Empty<int>()));
        var right = new DataFrame(new Column<int>("Id", [1, 2]));
        var result = left.Join(right, "Id");
        result.RowCount.Should().Be(0);
    }

    // -- Transpose --

    [Fact]
    public void Transpose_SwapsRowsAndColumns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var t = df.Transpose();

        t.RowCount.Should().Be(2); // 2 original columns
        t.ColumnCount.Should().Be(3); // column + 2 original rows
        t.GetStringColumn("column")[0].Should().Be("Name");
        t.GetStringColumn("column")[1].Should().Be("Age");
        t.GetStringColumn("row_0")[0].Should().Be("Alice");
        t.GetStringColumn("row_0")[1].Should().Be("25");
    }

    [Fact]
    public void Transpose_SingleColumn()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var t = df.Transpose();
        t.RowCount.Should().Be(1);
        t.ColumnCount.Should().Be(4); // column + 3 rows
    }

    // -- GroupBy edge cases --

    [Fact]
    public void GroupBy_SingleGroup_ReturnsSingleRow()
    {
        var df = new DataFrame(
            new StringColumn("Key", ["A", "A", "A"]),
            new Column<int>("Val", [1, 2, 3])
        );
        var result = df.GroupBy("Key").Sum();
        result.RowCount.Should().Be(1);
    }

    [Fact]
    public void GroupBy_EachRowIsGroup_ReturnsAllRows()
    {
        var df = new DataFrame(
            new Column<int>("Key", [1, 2, 3]),
            new Column<int>("Val", [10, 20, 30])
        );
        var result = df.GroupBy("Key").Sum();
        result.RowCount.Should().Be(3);
    }
}
