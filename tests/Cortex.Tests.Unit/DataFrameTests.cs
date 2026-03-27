using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameTests
{
    private static DataFrame CreateSampleDf() => new(
        new Column<int>("Age", [25, 30, 35, 28, 42]),
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
    );

    [Fact]
    public void Constructor_SetsProperties()
    {
        var df = CreateSampleDf();

        df.RowCount.Should().Be(5);
        df.ColumnCount.Should().Be(3);
        df.ColumnNames.Should().Equal(["Age", "Name", "Salary"]);
    }

    [Fact]
    public void Constructor_ThrowsOnDuplicateColumnNames()
    {
        var act = () => new DataFrame(
            new Column<int>("Age", [1]),
            new Column<int>("Age", [2])
        );

        act.Should().Throw<ArgumentException>().WithMessage("*Duplicate*");
    }

    [Fact]
    public void Constructor_ThrowsOnMismatchedLengths()
    {
        var act = () => new DataFrame(
            new Column<int>("A", [1, 2]),
            new Column<int>("B", [1, 2, 3])
        );

        act.Should().Throw<ArgumentException>().WithMessage("*rows*");
    }

    [Fact]
    public void ColumnAccess_ByName()
    {
        var df = CreateSampleDf();

        df["Age"].Should().BeOfType<Column<int>>();
        df["Name"].Should().BeOfType<StringColumn>();
        df["Salary"].Should().BeOfType<Column<double>>();
    }

    [Fact]
    public void ColumnAccess_ThrowsOnMissingColumn()
    {
        var df = CreateSampleDf();
        var act = () => df["Missing"];

        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void GetColumn_ReturnsTypedColumn()
    {
        var df = CreateSampleDf();
        var ages = df.GetColumn<int>("Age");

        ages[0].Should().Be(25);
        ages[4].Should().Be(42);
    }

    [Fact]
    public void GetStringColumn_ReturnsStringColumn()
    {
        var df = CreateSampleDf();
        var names = df.GetStringColumn("Name");

        names[0].Should().Be("Alice");
    }

    [Fact]
    public void RowAccess_ReturnsCorrectValues()
    {
        var df = CreateSampleDf();
        var row = df[0];

        row["Age"].Should().Be(25);
        row["Name"].Should().Be("Alice");
        row["Salary"].Should().Be(50_000.0);
    }

    [Fact]
    public void RowAccess_ThrowsOnOutOfRange()
    {
        var df = CreateSampleDf();
        var act = () => df[10];

        act.Should().Throw<IndexOutOfRangeException>();
    }

    [Fact]
    public void Head_ReturnsFirstNRows()
    {
        var df = CreateSampleDf();
        var head = df.Head(3);

        head.RowCount.Should().Be(3);
        head.GetColumn<int>("Age")[0].Should().Be(25);
        head.GetColumn<int>("Age")[2].Should().Be(35);
    }

    [Fact]
    public void Head_DefaultIsFive()
    {
        var df = CreateSampleDf();
        var head = df.Head();

        head.RowCount.Should().Be(5);
    }

    [Fact]
    public void Tail_ReturnsLastNRows()
    {
        var df = CreateSampleDf();
        var tail = df.Tail(2);

        tail.RowCount.Should().Be(2);
        tail.GetStringColumn("Name")[0].Should().Be("Diana");
        tail.GetStringColumn("Name")[1].Should().Be("Eve");
    }

    [Fact]
    public void Select_ReturnsSubsetOfColumns()
    {
        var df = CreateSampleDf();
        var selected = df.Select("Name", "Salary");

        selected.ColumnCount.Should().Be(2);
        selected.ColumnNames.Should().Equal(["Name", "Salary"]);
        selected.RowCount.Should().Be(5);
    }

    [Fact]
    public void Filter_WithBoolMask()
    {
        var df = CreateSampleDf();
        var mask = df.GetColumn<int>("Age").Gt(30);
        var filtered = df.Filter(mask);

        filtered.RowCount.Should().Be(2);
        filtered.GetStringColumn("Name")[0].Should().Be("Charlie");
        filtered.GetStringColumn("Name")[1].Should().Be("Eve");
    }

    [Fact]
    public void Filter_WithLambda()
    {
        var df = CreateSampleDf();
        var filtered = df.Filter(row => (int)row["Age"]! > 30);

        filtered.RowCount.Should().Be(2);
    }

    [Fact]
    public void Sort_Ascending()
    {
        var df = CreateSampleDf();
        var sorted = df.Sort("Age");

        sorted.GetColumn<int>("Age")[0].Should().Be(25);
        sorted.GetColumn<int>("Age")[4].Should().Be(42);
    }

    [Fact]
    public void Sort_Descending()
    {
        var df = CreateSampleDf();
        var sorted = df.Sort("Age", ascending: false);

        sorted.GetColumn<int>("Age")[0].Should().Be(42);
        sorted.GetColumn<int>("Age")[4].Should().Be(25);
    }

    [Fact]
    public void AddColumn_ReturnsNewDataFrame()
    {
        var df = CreateSampleDf();
        var newCol = new Column<bool>("Active", [true, false, true, true, false]);
        var result = df.AddColumn(newCol);

        result.ColumnCount.Should().Be(4);
        result.ColumnNames.Should().Contain("Active");
        df.ColumnCount.Should().Be(3); // original unchanged
    }

    [Fact]
    public void DropColumn_ReturnsNewDataFrame()
    {
        var df = CreateSampleDf();
        var result = df.DropColumn("Salary");

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().NotContain("Salary");
    }

    [Fact]
    public void RenameColumn_ReturnsNewDataFrame()
    {
        var df = CreateSampleDf();
        var result = df.RenameColumn("Age", "YearsOld");

        result.ColumnNames.Should().Contain("YearsOld");
        result.ColumnNames.Should().NotContain("Age");
    }

    [Fact]
    public void ToString_ProducesFormattedTable()
    {
        var df = CreateSampleDf();
        var output = df.ToString();

        output.Should().Contain("Age");
        output.Should().Contain("Name");
        output.Should().Contain("Alice");
        output.Should().Contain("5 rows x 3 columns");
    }

    [Fact]
    public void Enumerable_IteratesRows()
    {
        var df = CreateSampleDf();
        var names = new List<string?>();

        foreach (var row in df)
            names.Add(row.GetString("Name"));

        names.Should().Equal(["Alice", "Bob", "Charlie", "Diana", "Eve"]);
    }
}
