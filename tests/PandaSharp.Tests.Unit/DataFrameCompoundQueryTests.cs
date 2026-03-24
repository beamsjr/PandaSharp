using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameCompoundQueryTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        new Column<int>("Age", [25, 30, 35, 28, 42]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
    );

    [Fact]
    public void Query_And()
    {
        var result = Df().Query("Age > 28 and Salary < 80000");
        result.RowCount.Should().Be(2); // Bob (30, 62k) and Charlie (35, 75k)
    }

    [Fact]
    public void Query_Or()
    {
        var result = Df().Query("Age < 26 or Age > 40");
        result.RowCount.Should().Be(2); // Alice (25) and Eve (42)
    }

    [Fact]
    public void Query_AND_Uppercase()
    {
        var result = Df().Query("Age > 30 AND Salary > 80000");
        result.RowCount.Should().Be(1); // Eve
    }

    [Fact]
    public void Query_DoubleAmpersand()
    {
        var result = Df().Query("Age >= 30 && Age <= 35");
        result.RowCount.Should().Be(2); // Bob and Charlie
    }

    [Fact]
    public void Query_DoublePipe()
    {
        var result = Df().Query("Name == 'Alice' || Name == 'Eve'");
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void T_Shorthand_Works()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2]),
            new Column<int>("B", [3, 4])
        );

        var transposed = df.T;
        transposed.RowCount.Should().Be(2);
        transposed.ColumnCount.Should().Be(3); // column + 2 rows
    }
}
