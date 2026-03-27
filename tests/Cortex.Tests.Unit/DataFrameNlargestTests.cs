using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameNlargestTests
{
    private static DataFrame Df() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        new Column<double>("Salary", [50_000, 90_000, 75_000, 60_000, 85_000])
    );

    [Fact]
    public void Nlargest_ReturnsTopN()
    {
        var result = Df().Nlargest(3, "Salary");
        result.RowCount.Should().Be(3);
        result.GetColumn<double>("Salary")[0].Should().Be(90_000);
        result.GetColumn<double>("Salary")[1].Should().Be(85_000);
        result.GetColumn<double>("Salary")[2].Should().Be(75_000);
    }

    [Fact]
    public void Nsmallest_ReturnsBottomN()
    {
        var result = Df().Nsmallest(2, "Salary");
        result.RowCount.Should().Be(2);
        result.GetColumn<double>("Salary")[0].Should().Be(50_000);
        result.GetColumn<double>("Salary")[1].Should().Be(60_000);
    }

    [Fact]
    public void Nlargest_PreservesOtherColumns()
    {
        var result = Df().Nlargest(1, "Salary");
        result.GetStringColumn("Name")[0].Should().Be("Bob");
    }

    [Fact]
    public void Nlargest_MoreThanRows_ReturnsAll()
    {
        var result = Df().Nlargest(100, "Salary");
        result.RowCount.Should().Be(5);
    }
}
