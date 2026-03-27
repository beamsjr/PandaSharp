using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit;

public class DataFrameAssignTests
{
    [Fact]
    public void Assign_AddsNewColumn()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2, 3]));
        var result = df.Assign("B", new Column<int>("B", [10, 20, 30]));

        result.ColumnCount.Should().Be(2);
        result.GetColumn<int>("B")[0].Should().Be(10);
    }

    [Fact]
    public void Assign_ReplacesExistingColumn()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var newB = new Column<int>("B", [10, 20, 30]);
        var result = df.Assign("B", newB);

        result.ColumnCount.Should().Be(2);
        result["B"].DataType.Should().Be(typeof(int));
        result.GetColumn<int>("B")[0].Should().Be(10);
    }

    [Fact]
    public void Assign_Chain()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));

        var result = df
            .Assign("Y", new Column<int>("Y", [10, 20, 30]))
            .Assign("Z", new Column<double>("Z", [1.1, 2.2, 3.3]));

        result.ColumnCount.Should().Be(3);
    }

    [Fact]
    public void Assign_RenamesColumnIfNeeded()
    {
        var df = new DataFrame(new Column<int>("A", [1, 2]));
        var col = new Column<int>("WrongName", [10, 20]);
        var result = df.Assign("B", col);

        result.ColumnNames.Should().Contain("B");
        result.ColumnNames.Should().NotContain("WrongName");
    }

    [Fact]
    public void Cast_IntToDouble()
    {
        var df = new DataFrame(new Column<int>("Age", [25, 30, 35]));
        var result = Col("Age").Cast<double>().Evaluate(df);

        result.DataType.Should().Be(typeof(double));
        ((Column<double>)result)[0].Should().Be(25.0);
    }

    [Fact]
    public void Cast_DoubleToInt()
    {
        var df = new DataFrame(new Column<double>("Score", [95.7, 87.3]));
        var result = Col("Score").Cast<int>().Evaluate(df);

        ((Column<int>)result)[0].Should().Be(96); // Convert rounds
    }

    [Fact]
    public void Cast_WithNull()
    {
        var df = new DataFrame(Column<int>.FromNullable("X", [1, null, 3]));
        var result = Col("X").Cast<double>().Evaluate(df);

        result.IsNull(1).Should().BeTrue();
        ((Column<double>)result)[0].Should().Be(1.0);
    }
}
