using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.ML.Metrics;

namespace PandaSharp.ML.Tests;

public class MetricsTests
{
    [Fact]
    public void Classification_PerfectPrediction()
    {
        var yTrue = new Column<bool>("true", [true, true, false, false]);
        var yPred = new Column<bool>("pred", [true, true, false, false]);

        var m = MetricsCalculator.Classification(yTrue, yPred);
        m.Accuracy.Should().Be(1.0);
        m.Precision.Should().Be(1.0);
        m.Recall.Should().Be(1.0);
        m.F1.Should().Be(1.0);
    }

    [Fact]
    public void Classification_WithErrors()
    {
        var yTrue = new Column<bool>("true", [true, true, false, false, true]);
        var yPred = new Column<bool>("pred", [true, false, false, true, true]);

        var m = MetricsCalculator.Classification(yTrue, yPred);
        m.Accuracy.Should().Be(0.6);
        m.TruePositive.Should().Be(2);
        m.FalsePositive.Should().Be(1);
        m.FalseNegative.Should().Be(1);
        m.TrueNegative.Should().Be(1);
    }

    [Fact]
    public void Classification_FromIntColumns()
    {
        var yTrue = new Column<int>("true", [1, 1, 0, 0]);
        var yPred = new Column<int>("pred", [1, 0, 0, 0]);

        var m = MetricsCalculator.Classification(yTrue, yPred);
        m.TruePositive.Should().Be(1);
        m.FalseNegative.Should().Be(1);
    }

    [Fact]
    public void Regression_PerfectPrediction()
    {
        var yTrue = new Column<double>("true", [1, 2, 3, 4, 5]);
        var yPred = new Column<double>("pred", [1, 2, 3, 4, 5]);

        var m = MetricsCalculator.Regression(yTrue, yPred);
        m.MSE.Should().Be(0);
        m.R2.Should().Be(1);
    }

    [Fact]
    public void Regression_WithErrors()
    {
        var yTrue = new Column<double>("true", [3.0, -0.5, 2.0, 7.0]);
        var yPred = new Column<double>("pred", [2.5, 0.0, 2.0, 8.0]);

        var m = MetricsCalculator.Regression(yTrue, yPred);
        m.MSE.Should().BeApproximately(0.375, 0.001);
        m.RMSE.Should().BeApproximately(0.6124, 0.001);
        m.MAE.Should().BeApproximately(0.5, 0.001);
        m.R2.Should().BeGreaterThan(0.9);
    }

    [Fact]
    public void Regression_ToString()
    {
        var yTrue = new Column<double>("true", [1, 2, 3]);
        var yPred = new Column<double>("pred", [1.1, 2.1, 3.1]);

        var m = MetricsCalculator.Regression(yTrue, yPred);
        m.ToString().Should().Contain("MSE");
        m.ToString().Should().Contain("R²");
    }
}
