using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameFinalTests
{
    // -- Typed ValueCounts --

    [Fact]
    public void ValueCounts_IntColumn()
    {
        var col = new Column<int>("X", [1, 2, 2, 3, 3, 3]);
        var vc = col.ValueCounts();

        vc.RowCount.Should().Be(3);
        vc.GetColumn<int>("X")[0].Should().Be(3); // most frequent
        vc.GetColumn<int>("count")[0].Should().Be(3);
    }

    [Fact]
    public void ValueCounts_DoubleColumn()
    {
        var col = new Column<double>("X", [1.0, 1.0, 2.0]);
        var vc = col.ValueCounts();

        vc.RowCount.Should().Be(2);
        vc.GetColumn<double>("X")[0].Should().Be(1.0);
        vc.GetColumn<int>("count")[0].Should().Be(2);
    }

    [Fact]
    public void ValueCounts_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 1, null]);
        var vc = col.ValueCounts();
        vc.RowCount.Should().Be(1); // just "1"
        vc.GetColumn<int>("count")[0].Should().Be(2);
    }

    // -- Summary --

    [Fact]
    public void Summary_ReturnsCompactOverview()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30]),
            Column<double>.FromNullable("Score", [90.0, null])
        );

        var summary = df.Summary();

        summary.Should().Contain("2 rows");
        summary.Should().Contain("3 columns");
        summary.Should().Contain("Name");
        summary.Should().Contain("string");
        summary.Should().Contain("int32");
        summary.Should().Contain("1 nulls"); // Score has 1 null
    }

    [Fact]
    public void Summary_EmptyDataFrame()
    {
        var summary = new DataFrame().Summary();
        summary.Should().Contain("0 rows");
    }

    // -- Comprehensive API smoke test --

    [Fact]
    public void SmokeTest_AllMajorAPIs()
    {
        // Construction
        var df = DataFrame.FromDictionary(new()
        {
            ["Name"] = new string?[] { "Alice", "Bob", "Charlie" },
            ["Age"] = new int[] { 25, 30, 35 },
            ["Score"] = new double[] { 90.0, 85.0, 70.0 }
        });
        df.RowCount.Should().Be(3);

        // Properties
        df.Shape.Should().Be((3, 3));
        df.IsEmpty.Should().BeFalse();
        df.Dtypes.Count.Should().Be(3);

        // Selection
        df.Head(2).RowCount.Should().Be(2);
        df.Tail(2).RowCount.Should().Be(2);
        df.Select("Name", "Age").ColumnCount.Should().Be(2);
        df.NumericOnly().ColumnCount.Should().Be(2);

        // Filtering
        df.Query("Age > 28").RowCount.Should().Be(2);
        df.Where("Age", v => (int)v! > 28).RowCount.Should().Be(2);
        df.Filter(df.GetColumn<int>("Age").Gt(28)).RowCount.Should().Be(2);

        // Sorting
        df.Sort("Age").GetColumn<int>("Age")[0].Should().Be(25);
        df.SortValues("Age", ascending: false).GetColumn<int>("Age")[0].Should().Be(35);
        df.Nlargest(1, "Score").GetColumn<double>("Score")[0].Should().Be(90.0);
        df.Nsmallest(1, "Score").GetColumn<double>("Score")[0].Should().Be(70.0);

        // Mutation
        var assigned = df.Assign("Flag", new Column<bool>("Flag", [true, false, true]));
        assigned.ColumnCount.Should().Be(4);
        df.DropColumn("Score").ColumnCount.Should().Be(2);
        df.RenameColumn("Age", "Years").ColumnNames.Should().Contain("Years");

        // Aggregation
        df.GetColumn<double>("Score").Sum().Should().Be(245.0);
        df.GetColumn<double>("Score").Mean().Should().Be(245.0 / 3);
        df.GetColumn<double>("Score").ArgMax().Should().Be(0);

        // Display
        df.ToString().Should().Contain("Alice");
        df.ToHtml().Should().Contain("<table");
        df.ToMarkdown().Should().Contain("|");
        df.Summary().Should().Contain("3 rows");

        // Copy/Compare
        var copy = df.Copy();
        df.ContentEquals(copy).Should().BeTrue();

        // Iteration
        df.Iterrows().Count().Should().Be(3);
        df.Itertuples().Count().Should().Be(3);
        df.Itercolumns().Count().Should().Be(3);
    }
}
