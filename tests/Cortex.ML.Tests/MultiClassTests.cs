using FluentAssertions;
using Cortex.Column;
using Cortex.ML.Metrics;

namespace Cortex.ML.Tests;

public class MultiClassTests
{
    [Fact]
    public void MultiClass_PerfectPrediction()
    {
        var yTrue = new Column<int>("T", [0, 1, 2, 0, 1, 2]);
        var yPred = new Column<int>("P", [0, 1, 2, 0, 1, 2]);

        var m = MetricsCalculator.MultiClass(yTrue, yPred);
        m.Accuracy.Should().Be(1.0);
        m.MacroF1.Should().Be(1.0);
        m.WeightedF1.Should().Be(1.0);
    }

    [Fact]
    public void MultiClass_WithErrors()
    {
        var yTrue = new Column<int>("T", [0, 0, 1, 1, 2, 2]);
        var yPred = new Column<int>("P", [0, 1, 1, 2, 2, 0]);

        var m = MetricsCalculator.MultiClass(yTrue, yPred);
        m.Accuracy.Should().Be(0.5); // 3 correct out of 6
        m.Classes.Should().Equal([0, 1, 2]);
        m.MacroF1.Should().BeGreaterThan(0);
    }

    [Fact]
    public void MultiClass_ConfusionMatrix()
    {
        var yTrue = new Column<int>("T", [0, 0, 1, 1]);
        var yPred = new Column<int>("P", [0, 1, 1, 1]);

        var m = MetricsCalculator.MultiClass(yTrue, yPred);
        m.ConfusionMatrix[0, 0].Should().Be(1); // class 0 correct
        m.ConfusionMatrix[0, 1].Should().Be(1); // class 0 predicted as 1
        m.ConfusionMatrix[1, 1].Should().Be(2); // class 1 correct
    }

    [Fact]
    public void MultiClass_PerClassMetrics()
    {
        var yTrue = new Column<int>("T", [0, 0, 0, 1, 1, 1, 2, 2, 2]);
        var yPred = new Column<int>("P", [0, 0, 0, 1, 1, 2, 2, 2, 0]);

        var m = MetricsCalculator.MultiClass(yTrue, yPred);

        // Class 0: all 3 predicted correctly, but also predicted for class 2 → precision < 1
        m.Recall[0].Should().Be(1.0); // all class 0 items found
        m.Support[0].Should().Be(3);
    }

    [Fact]
    public void MultiClass_ToString_Formatted()
    {
        var yTrue = new Column<int>("T", [0, 1, 2]);
        var yPred = new Column<int>("P", [0, 1, 2]);

        var m = MetricsCalculator.MultiClass(yTrue, yPred);
        var str = m.ToString();
        str.Should().Contain("Accuracy");
        str.Should().Contain("Macro F1");
        str.Should().Contain("Precision");
    }
}
