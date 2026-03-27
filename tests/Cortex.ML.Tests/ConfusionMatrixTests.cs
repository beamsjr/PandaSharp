using Xunit;
using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Metrics;

namespace Cortex.ML.Tests;

public class ConfusionMatrixTests
{
    [Fact]
    public void BinaryConfusionMatrix_ToDataFrame_HasCorrectLabelsAndValues()
    {
        // Arrange: known binary classification
        // yTrue: [1, 1, 0, 0, 1, 0]
        // yPred: [1, 0, 0, 1, 1, 0]
        // TP=2, FP=1, TN=2, FN=1
        // CM = [[TN, FP], [FN, TP]] = [[2, 1], [1, 2]]
        var yTrue = new Column<int>("true", [1, 1, 0, 0, 1, 0]);
        var yPred = new Column<int>("pred", [1, 0, 0, 1, 1, 0]);

        var result = MetricsCalculator.Classification(yTrue, yPred);

        // Act
        var df = result.ToDataFrame();

        // Assert
        df.ColumnNames.Should().BeEquivalentTo(["0", "1"]);
        df.RowCount.Should().Be(2);

        // Row 0 (actual=0): TN=2, FP=1
        df["0"].GetObject(0).Should().Be(2); // TN
        df["1"].GetObject(0).Should().Be(1); // FP

        // Row 1 (actual=1): FN=1, TP=2
        df["0"].GetObject(1).Should().Be(1); // FN
        df["1"].GetObject(1).Should().Be(2); // TP
    }

    [Fact]
    public void MultiClassConfusionMatrix_ToDataFrame_HasCorrectLabelsAndValues()
    {
        // Arrange: 4-class classification
        // Classes: 0, 1, 2, 3
        var yTrue = new Column<int>("true", [0, 0, 1, 1, 2, 2, 3, 3, 0, 1, 2, 3]);
        var yPred = new Column<int>("pred", [0, 1, 1, 2, 2, 0, 3, 2, 0, 1, 3, 3]);

        var result = MetricsCalculator.MultiClass(yTrue, yPred);

        // Act
        var df = result.ToDataFrame();

        // Assert: 4 columns labeled "0", "1", "2", "3"
        df.ColumnNames.Should().BeEquivalentTo(["0", "1", "2", "3"]);
        df.RowCount.Should().Be(4);

        // Verify the confusion matrix values match
        for (int row = 0; row < 4; row++)
        {
            for (int col = 0; col < 4; col++)
            {
                df[col.ToString()].GetObject(row).Should().Be(result.ConfusionMatrix[row, col],
                    $"CM[{row},{col}] should match DataFrame value");
            }
        }

        // Spot-check: class 0 actual row: predicted as 0 twice, as 1 once, as 2/3 zero
        df["0"].GetObject(0).Should().Be(2); // correctly predicted as 0
        df["1"].GetObject(0).Should().Be(1); // misclassified as 1
    }
}
