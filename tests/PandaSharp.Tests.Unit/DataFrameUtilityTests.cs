using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameUtilityTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Alice", "Bob"]),
        new Column<int>("Age", [25, 30, 35, 25, 30]),
        new Column<double>("Score", [90.0, 85.0, 70.0, 90.0, 85.0])
    );

    // -- DropDuplicates --

    [Fact]
    public void DropDuplicates_AllColumns_RemovesDuplicateRows()
    {
        var result = Df().DropDuplicates();
        result.RowCount.Should().Be(3); // Alice/25/90, Bob/30/85, Charlie/35/70
    }

    [Fact]
    public void DropDuplicates_Subset_RemovesBySubset()
    {
        var result = Df().DropDuplicates("Name");
        result.RowCount.Should().Be(3); // Alice, Bob, Charlie (first occurrence kept)
        result.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void DropDuplicates_NoDuplicates_ReturnsSame()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        df.DropDuplicates().RowCount.Should().Be(3);
    }

    [Fact]
    public void DropDuplicates_TwoStringColumns_ParallelDictEncoding()
    {
        var df = new DataFrame(
            new StringColumn("City", ["NYC", "LA", "NYC", "LA", "NYC", "Chicago"]),
            new StringColumn("Dept", ["Sales", "Eng", "Eng", "Eng", "Sales", "Sales"]),
            new Column<int>("Val", [1, 2, 3, 4, 5, 6])
        );

        var result = df.DropDuplicates("City", "Dept");

        // Unique (City, Dept) pairs: (NYC,Sales), (LA,Eng), (NYC,Eng), (Chicago,Sales)
        result.RowCount.Should().Be(4);

        // Verify first occurrences are kept (rows 0, 1, 2, 5)
        var cities = result.GetStringColumn("City");
        var depts = result.GetStringColumn("Dept");
        var vals = result.GetColumn<int>("Val");

        // Row 0: NYC, Sales, 1
        cities[0].Should().Be("NYC");
        depts[0].Should().Be("Sales");
        vals[0].Should().Be(1);

        // Row 1: LA, Eng, 2
        cities[1].Should().Be("LA");
        depts[1].Should().Be("Eng");
        vals[1].Should().Be(2);

        // Row 2: NYC, Eng, 3 (not row 4 which is NYC, Sales duplicate)
        cities[2].Should().Be("NYC");
        depts[2].Should().Be("Eng");
        vals[2].Should().Be(3);

        // Row 3: Chicago, Sales, 6
        cities[3].Should().Be("Chicago");
        depts[3].Should().Be("Sales");
        vals[3].Should().Be(6);
    }

    // -- RenameColumns --

    [Fact]
    public void RenameColumns_Dictionary()
    {
        var result = Df().RenameColumns(new() { ["Name"] = "Person", ["Age"] = "Years" });

        result.ColumnNames.Should().Contain("Person");
        result.ColumnNames.Should().Contain("Years");
        result.ColumnNames.Should().Contain("Score"); // unchanged
        result.ColumnNames.Should().NotContain("Name");
    }

    // -- ReorderColumns --

    [Fact]
    public void ReorderColumns_ChangesOrder()
    {
        var result = Df().ReorderColumns("Score", "Name");

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["Score", "Name"]);
    }

    // -- DropColumns --

    [Fact]
    public void DropColumns_Multiple()
    {
        var result = Df().DropColumns("Age", "Score");

        result.ColumnCount.Should().Be(1);
        result.ColumnNames.Should().Equal(["Name"]);
    }

    // -- Shape --

    [Fact]
    public void Shape_ReturnsTuple()
    {
        var (rows, cols) = Df().Shape;
        rows.Should().Be(5);
        cols.Should().Be(3);
    }

    // -- Between --

    [Fact]
    public void Between_ReturnsMaskForRange()
    {
        var col = new Column<int>("X", [10, 20, 30, 40, 50]);
        var mask = col.Between(20, 40);
        mask.Should().Equal([false, true, true, true, false]);
    }

    [Fact]
    public void Between_WithNulls()
    {
        var col = Column<int>.FromNullable("X", [10, null, 30]);
        var mask = col.Between(5, 25);
        mask.Should().Equal([true, false, false]);
    }

    // -- IsIn --

    [Fact]
    public void IsIn_NumericColumn()
    {
        var col = new Column<int>("X", [1, 2, 3, 4, 5]);
        var mask = col.IsIn(2, 4);
        mask.Should().Equal([false, true, false, true, false]);
    }

    [Fact]
    public void IsIn_StringColumn()
    {
        var col = new StringColumn("Name", ["Alice", "Bob", "Charlie"]);
        var mask = col.IsIn("Alice", "Charlie");
        mask.Should().Equal([true, false, true]);
    }

    // -- Combined filter with Between + And --

    [Fact]
    public void Between_CombinedWithAnd()
    {
        var df = Df();
        var mask = df.GetColumn<int>("Age").Between(25, 30)
            .And(df.GetColumn<double>("Score").Gt(85));

        var result = df.Filter(mask);
        result.RowCount.Should().Be(2); // Alice (25, 90) and Alice (25, 90) dupes
    }
}
