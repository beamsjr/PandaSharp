using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit;

public class DataFrameDescribeAllTests
{
    // -- DescribeAll --

    [Fact]
    public void DescribeAll_IncludesStringAndNumericStats()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice", "Charlie"]),
            new Column<int>("Age", [25, 30, 25, 35]),
            new Column<double>("Score", [90.0, 85.0, 90.0, 70.0])
        );

        var desc = df.DescribeAll();

        desc.ColumnNames.Should().Contain("stat");
        desc.ColumnNames.Should().Contain("Name");
        desc.ColumnNames.Should().Contain("Age");
        desc.ColumnNames.Should().Contain("Score");

        // Name should have unique/top/freq
        var nameCol = desc.GetStringColumn("Name");
        nameCol[0].Should().Be("4"); // count
        nameCol[1].Should().Be("3"); // unique (Alice, Bob, Charlie)
        nameCol[2].Should().Be("Alice"); // top
        nameCol[3].Should().Be("2"); // freq

        // Age should have mean/std/min/max
        var ageCol = desc.GetStringColumn("Age");
        ageCol[0].Should().Be("4"); // count
        ageCol[4].Should().NotBeNull(); // mean
    }

    // -- NumericOnly --

    [Fact]
    public void NumericOnly_FiltersToNumericColumns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25]),
            new Column<double>("Score", [90.0]),
            new Column<bool>("Active", [true])
        );

        var numeric = df.NumericOnly();
        numeric.ColumnCount.Should().Be(2);
        numeric.ColumnNames.Should().Equal(["Age", "Score"]);
    }

    // -- Itercolumns --

    [Fact]
    public void Itercolumns_YieldsNameAndColumn()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2]),
            new StringColumn("B", ["x", "y"])
        );

        var cols = df.Itercolumns().ToList();
        cols.Should().HaveCount(2);
        cols[0].Name.Should().Be("A");
        cols[0].Column.Length.Should().Be(2);
        cols[1].Name.Should().Be("B");
    }

    // -- ConcatStr expression --

    [Fact]
    public void ConcatStr_CombinesStrings()
    {
        var df = new DataFrame(
            new StringColumn("First", ["Alice", "Bob"]),
            new StringColumn("Last", ["Smith", "Jones"])
        );

        var result = ConcatStr(Col("First"), Lit(" "), Col("Last")).Evaluate(df);

        ((StringColumn)result)[0].Should().Be("Alice Smith");
        ((StringColumn)result)[1].Should().Be("Bob Jones");
    }

    [Fact]
    public void ConcatStr_WithNumeric()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30])
        );

        var result = ConcatStr(Col("Name"), Lit(" is "), Col("Age")).Evaluate(df);

        ((StringColumn)result)[0].Should().Be("Alice is 25");
    }

    [Fact]
    public void ConcatStr_NullPropagation()
    {
        var df = new DataFrame(
            new StringColumn("A", ["hello", null]),
            new StringColumn("B", [" world", " there"])
        );

        var result = ConcatStr(Col("A"), Col("B")).Evaluate(df);

        ((StringColumn)result)[0].Should().Be("hello world");
        result.IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void ConcatStr_WithColumn_InDataFrame()
    {
        var df = new DataFrame(
            new StringColumn("First", ["Alice", "Bob"]),
            new StringColumn("Last", ["Smith", "Jones"])
        );

        var result = df.WithColumn(ConcatStr(Col("First"), Lit(" "), Col("Last")), "FullName");

        result.GetStringColumn("FullName")[0].Should().Be("Alice Smith");
    }
}
