using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class ExtendedAggregationTests
{
    [Fact]
    public void Median_OddCount()
    {
        var col = new Column<double>("x", [3.0, 1.0, 2.0]);
        col.Median().Should().Be(2.0);
    }

    [Fact]
    public void Median_EvenCount()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0]);
        col.Median().Should().Be(2.5);
    }

    [Fact]
    public void Median_WithNulls_SkipsNulls()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, 3.0]);
        col.Median().Should().Be(2.0);
    }

    [Fact]
    public void Std_ReturnsStandardDeviation()
    {
        // Sample std (ddof=1) of [2,4,4,4,5,5,7,9]: mean=5, var=32/7≈4.571, std≈2.138
        var col = new Column<double>("x", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        col.Std().Should().BeApproximately(2.138, 0.01);
    }

    [Fact]
    public void Var_ReturnsVariance()
    {
        // Sample variance (ddof=1): sum((x-mean)^2) / (n-1) = 32/7 ≈ 4.571
        var col = new Column<double>("x", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        col.Var().Should().BeApproximately(4.571, 0.01);
    }

    [Fact]
    public void Quantile_ReturnsCorrectPercentile()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0, 5.0]);
        col.Quantile(0.25).Should().Be(2.0);
        col.Quantile(0.75).Should().Be(4.0);
    }

    [Fact]
    public void Quantile_OutOfRange_Throws()
    {
        var col = new Column<double>("x", [1.0]);
        var act = () => col.Quantile(1.5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Mode_ReturnsMostFrequent()
    {
        var col = new Column<int>("x", [1, 2, 2, 3, 3, 3]);
        col.Mode().Should().Be(3);
    }

    [Fact]
    public void Skew_ReturnsSkewness()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0, 4.0, 5.0]);
        col.Skew().Should().BeApproximately(0.0, 0.01); // symmetric distribution
    }

    [Fact]
    public void Kurtosis_ReturnsTooFewValues_Null()
    {
        var col = new Column<double>("x", [1.0, 2.0, 3.0]);
        col.Kurtosis().Should().BeNull();
    }

    [Fact]
    public void Sem_ReturnsStandardErrorOfMean()
    {
        var col = new Column<double>("x", [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]);
        var std = col.Std()!.Value;
        col.Sem().Should().BeApproximately(std / Math.Sqrt(8), 0.001);
    }

    [Fact]
    public void CumSum_ReturnsCumulativeSum()
    {
        var col = new Column<int>("x", [1, 2, 3, 4]);
        var cum = col.CumSum();
        cum[0].Should().Be(1);
        cum[1].Should().Be(3);
        cum[2].Should().Be(6);
        cum[3].Should().Be(10);
    }

    [Fact]
    public void CumSum_WithNulls_SkipsNulls()
    {
        var col = Column<int>.FromNullable("x", [1, null, 3]);
        var cum = col.CumSum();
        cum[0].Should().Be(1);
        cum[1].Should().BeNull();
        cum[2].Should().Be(4);
    }

    [Fact]
    public void CumProd_ReturnsCumulativeProduct()
    {
        var col = new Column<int>("x", [1, 2, 3, 4]);
        var cum = col.CumProd();
        cum[0].Should().Be(1);
        cum[1].Should().Be(2);
        cum[2].Should().Be(6);
        cum[3].Should().Be(24);
    }

    [Fact]
    public void CumMin_ReturnsCumulativeMinimum()
    {
        var col = new Column<int>("x", [3, 1, 4, 1, 5]);
        var cum = col.CumMin();
        cum[0].Should().Be(3);
        cum[1].Should().Be(1);
        cum[2].Should().Be(1);
        cum[3].Should().Be(1);
        cum[4].Should().Be(1);
    }

    [Fact]
    public void CumMax_ReturnsCumulativeMaximum()
    {
        var col = new Column<int>("x", [1, 3, 2, 5, 4]);
        var cum = col.CumMax();
        cum[0].Should().Be(1);
        cum[1].Should().Be(3);
        cum[2].Should().Be(3);
        cum[3].Should().Be(5);
        cum[4].Should().Be(5);
    }

    [Fact]
    public void PctChange_ReturnsPercentChange()
    {
        var col = new Column<double>("x", [100.0, 110.0, 99.0]);
        var pct = col.PctChange();
        pct[0].Should().BeNull();
        pct[1].Should().BeApproximately(0.1, 0.001);
        pct[2].Should().BeApproximately(-0.1, 0.001);
    }

    [Fact]
    public void Diff_ReturnsDifference()
    {
        var col = new Column<double>("x", [10.0, 15.0, 12.0]);
        var diff = col.Diff();
        diff[0].Should().BeNull();
        diff[1].Should().Be(5.0);
        diff[2].Should().Be(-3.0);
    }
}
