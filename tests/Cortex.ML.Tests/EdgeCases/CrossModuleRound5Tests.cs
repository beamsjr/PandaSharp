using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests.EdgeCases;

/// <summary>
/// Round 5 bug hunting: cross-module data flow issues in ML module.
/// </summary>
public class CrossModuleRound5Tests
{
    /// <summary>
    /// Bug: DataFrameDataLoader does not validate that feature columns are numeric.
    /// When a string column is included in the feature set, TypeHelpers.GetDouble
    /// falls through to Convert.ToDouble(string) which throws FormatException
    /// with an unhelpful error message. The error should be caught early with a
    /// clear message indicating the column is not numeric.
    /// </summary>
    [Fact]
    public void DataLoader_StringFeatureColumn_ShouldGiveClearError()
    {
        var df = new DataFrame(
            new StringColumn("Name", new string?[] { "Alice", "Bob", "Charlie" }),
            new Column<double>("Score", new double[] { 1.0, 2.0, 3.0 }),
            new Column<double>("Label", new double[] { 0, 1, 0 })
        );

        // Including "Name" (a string column) as a feature should fail gracefully
        var act = () =>
        {
            var loader = df.ToDataLoader(new[] { "Name", "Score" }, "Label",
                batchSize: 10, shuffle: false);
            // Force enumeration to trigger the conversion
            _ = loader.ToList();
        };

        // Should throw a clear error about non-numeric column, not a FormatException
        act.Should().Throw<Exception>()
            .Which.Message.Should().Contain("Name",
                "error should identify the problematic column");
    }

    /// <summary>
    /// Bug: DataFrameDataLoader with string label column crashes with
    /// unhelpful FormatException instead of a clear error.
    /// </summary>
    [Fact]
    public void DataLoader_StringLabelColumn_ShouldGiveClearError()
    {
        var df = new DataFrame(
            new Column<double>("F1", new double[] { 1.0, 2.0, 3.0 }),
            new StringColumn("Label", new string?[] { "cat", "dog", "cat" })
        );

        var act = () =>
        {
            var loader = df.ToDataLoader(new[] { "F1" }, "Label",
                batchSize: 10, shuffle: false);
            _ = loader.ToList();
        };

        act.Should().Throw<Exception>()
            .Which.Message.Should().Contain("Label",
                "error should identify the problematic label column");
    }

    /// <summary>
    /// Verify DataLoader correctly converts Column&lt;int&gt; features to double tensors.
    /// This tests cross-type data flow: int columns should work transparently.
    /// </summary>
    [Fact]
    public void DataLoader_IntFeatureColumns_ConvertToDoubleTensors()
    {
        var df = new DataFrame(
            new Column<int>("F1", new int[] { 1, 2, 3, 4, 5 }),
            new Column<int>("F2", new int[] { 10, 20, 30, 40, 50 }),
            new Column<double>("Label", new double[] { 0, 1, 0, 1, 0 })
        );

        var loader = df.ToDataLoader(new[] { "F1", "F2" }, "Label",
            batchSize: 5, shuffle: false);
        var batches = loader.ToList();

        batches.Should().HaveCount(1);
        var (features, labels) = batches[0];
        features.Shape.Should().BeEquivalentTo(new[] { 5, 2 });
        // First row: F1=1, F2=10
        features.Span[0].Should().Be(1.0);
        features.Span[1].Should().Be(10.0);
    }
}
