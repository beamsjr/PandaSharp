using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameQueryTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana"]),
        new Column<int>("Age", [25, 30, 35, 28]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000])
    );

    [Fact]
    public void Query_GreaterThan()
    {
        var result = Df().Query("Age > 30");
        result.RowCount.Should().Be(1);
        result.GetStringColumn("Name")[0].Should().Be("Charlie");
    }

    [Fact]
    public void Query_GreaterThanOrEqual()
    {
        var result = Df().Query("Age >= 30");
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Query_LessThan()
    {
        var result = Df().Query("Age < 28");
        result.RowCount.Should().Be(1);
        result.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void Query_Equal()
    {
        var result = Df().Query("Age == 30");
        result.RowCount.Should().Be(1);
        result.GetStringColumn("Name")[0].Should().Be("Bob");
    }

    [Fact]
    public void Query_NotEqual()
    {
        var result = Df().Query("Age != 30");
        result.RowCount.Should().Be(3);
    }

    [Fact]
    public void Query_Double()
    {
        var result = Df().Query("Salary > 60000");
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Query_String()
    {
        var result = Df().Query("Name == 'Alice'");
        result.RowCount.Should().Be(1);
    }

    [Fact]
    public void Query_InvalidFormat_Throws()
    {
        var act = () => Df().Query("invalid query");
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Iterrows_YieldsIndexAndRow()
    {
        var df = Df();
        var rows = df.Iterrows().ToList();

        rows.Should().HaveCount(4);
        rows[0].Index.Should().Be(0);
        rows[0].Row.GetString("Name").Should().Be("Alice");
        rows[3].Index.Should().Be(3);
    }

    [Fact]
    public void Itertuples_YieldsObjectArrays()
    {
        var df = Df();
        var tuples = df.Itertuples().ToList();

        tuples.Should().HaveCount(4);
        tuples[0].Should().HaveCount(3); // 3 columns
        tuples[0][0].Should().Be("Alice");
        tuples[0][1].Should().Be(25);
    }
}
