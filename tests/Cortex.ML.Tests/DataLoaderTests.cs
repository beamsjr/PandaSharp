using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests;

public class DataLoaderTests
{
    private static DataFrame SampleDf() => new(
        new Column<double>("F1", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        new Column<double>("F2", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]),
        new Column<double>("Label", [0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    );

    [Fact]
    public void DataLoader_YieldsCorrectBatchCount()
    {
        var loader = SampleDf().ToDataLoader(["F1", "F2"], "Label", batchSize: 3, shuffle: false);
        loader.BatchCount.Should().Be(4); // 10/3 = 3.33 → 4 batches
    }

    [Fact]
    public void DataLoader_BatchShapes()
    {
        var loader = SampleDf().ToDataLoader(["F1", "F2"], "Label", batchSize: 4, shuffle: false);
        var batches = loader.ToList();

        batches.Should().HaveCount(3); // 4+4+2

        batches[0].Features.Shape.Should().Equal([4, 2]); // 4 rows × 2 features
        batches[0].Labels.Shape.Should().Equal([4]);       // 4 labels
        batches[2].Features.Shape[0].Should().Be(2);       // last batch has 2 rows
    }

    [Fact]
    public void DataLoader_NoShuffle_Ordered()
    {
        var loader = SampleDf().ToDataLoader(["F1"], "Label", batchSize: 5, shuffle: false);
        var batches = loader.ToList();

        batches[0].Features[0, 0].Should().Be(1); // first row
        batches[0].Features[4, 0].Should().Be(5); // fifth row
    }

    [Fact]
    public void DataLoader_Shuffle_Deterministic()
    {
        var loader1 = SampleDf().ToDataLoader(["F1"], "Label", batchSize: 10, shuffle: true, seed: 42);
        var loader2 = SampleDf().ToDataLoader(["F1"], "Label", batchSize: 10, shuffle: true, seed: 42);

        var b1 = loader1.First().Features;
        var b2 = loader2.First().Features;

        for (int i = 0; i < 10; i++)
            b1[i, 0].Should().Be(b2[i, 0]);
    }

    [Fact]
    public void DataLoader_AllDataCovered()
    {
        var loader = SampleDf().ToDataLoader(["F1", "F2"], "Label", batchSize: 3, shuffle: false);
        int totalRows = 0;
        foreach (var (features, labels) in loader)
        {
            totalRows += features.Shape[0];
            features.Shape[1].Should().Be(2);
        }
        totalRows.Should().Be(10);
    }

    [Fact]
    public void DataLoader_SingleBatch()
    {
        var loader = SampleDf().ToDataLoader(["F1", "F2"], "Label", batchSize: 100, shuffle: false);
        var batches = loader.ToList();
        batches.Should().HaveCount(1);
        batches[0].Features.Shape.Should().Equal([10, 2]);
    }

    [Fact]
    public void DataLoader_EndToEnd_TrainingLoop()
    {
        var df = SampleDf();
        var loader = df.ToDataLoader(["F1", "F2"], "Label", batchSize: 5, shuffle: true, seed: 42);

        // Simulate a training loop
        double totalLoss = 0;
        int batchCount = 0;
        foreach (var (features, labels) in loader)
        {
            // "Forward pass": simple dot product prediction
            var weights = Tensor<double>.Ones(2);
            for (int r = 0; r < features.Shape[0]; r++)
            {
                double pred = features[r, 0] * 0.1 + features[r, 1] * 0.01;
                double loss = (pred - labels[r]) * (pred - labels[r]);
                totalLoss += loss;
            }
            batchCount++;
        }

        batchCount.Should().Be(2); // 10/5
        totalLoss.Should().BeGreaterThan(0);
    }
}
