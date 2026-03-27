using FluentAssertions;
using Cortex.Column;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.Expressions;

public class AggExprTests
{
    private static DataFrame Df() => new(
        new Column<double>("Value", [10.0, 20.0, 30.0, 40.0]),
        new StringColumn("Name", ["A", "B", "C", "D"])
    );

    [Fact]
    public void Sum_BroadcastsToAllRows()
    {
        var result = (Column<double>)Col("Value").Sum().Evaluate(Df());
        result.Length.Should().Be(4);
        result[0].Should().Be(100.0);
        result[3].Should().Be(100.0); // all same
    }

    [Fact]
    public void Mean_BroadcastsToAllRows()
    {
        var result = (Column<double>)Col("Value").Mean().Evaluate(Df());
        result[0].Should().Be(25.0);
    }

    [Fact]
    public void Min_BroadcastsToAllRows()
    {
        var result = (Column<double>)Col("Value").Min().Evaluate(Df());
        result[0].Should().Be(10.0);
    }

    [Fact]
    public void Max_BroadcastsToAllRows()
    {
        var result = (Column<double>)Col("Value").Max().Evaluate(Df());
        result[0].Should().Be(40.0);
    }

    [Fact]
    public void Count_BroadcastsToAllRows()
    {
        var result = (Column<double>)Col("Value").Count().Evaluate(Df());
        result[0].Should().Be(4.0);
    }

    [Fact]
    public void AggExpr_UsefulForNormalization()
    {
        var df = Df();
        // Normalize: (x - mean) / std
        var normalized = df.WithColumn(
            (Col("Value") - Col("Value").Mean()) / Col("Value").StdExpr(),
            "Normalized"
        );

        normalized.ColumnNames.Should().Contain("Normalized");
        // Mean of normalized should be ~0
        var normCol = normalized.GetColumn<double>("Normalized");
        var meanNorm = normCol.Mean()!.Value;
        meanNorm.Should().BeApproximately(0, 0.001);
    }

    [Fact]
    public void AggExpr_PctOfTotal()
    {
        var df = Df();
        // Percentage of total
        var result = df.WithColumn(
            Col("Value") / Col("Value").Sum() * Lit(100.0),
            "PctOfTotal"
        );

        result.GetColumn<double>("PctOfTotal")[0].Should().BeApproximately(10.0, 0.01);
        result.GetColumn<double>("PctOfTotal")[3].Should().BeApproximately(40.0, 0.01);
    }

    [Fact]
    public void MedianExpr_Works()
    {
        var result = (Column<double>)Col("Value").MedianExpr().Evaluate(Df());
        result[0].Should().Be(25.0); // median of [10,20,30,40]
    }
}
