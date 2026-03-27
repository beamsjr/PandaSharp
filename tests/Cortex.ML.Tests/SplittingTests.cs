using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Splitting;

namespace Cortex.ML.Tests;

public class SplittingTests
{
    [Fact]
    public void TrainTestSplit_CorrectSizes()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var (train, test) = df.TrainTestSplit(testFraction: 0.2, seed: 42);

        train.RowCount.Should().Be(80);
        test.RowCount.Should().Be(20);
    }

    [Fact]
    public void TrainTestSplit_NoOverlap()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var (train, test) = df.TrainTestSplit(testFraction: 0.2, seed: 42);

        var trainSet = new HashSet<int?>();
        for (int i = 0; i < train.RowCount; i++) trainSet.Add(train.GetColumn<int>("X")[i]);
        for (int i = 0; i < test.RowCount; i++)
            trainSet.Should().NotContain(test.GetColumn<int>("X")[i]);
    }

    [Fact]
    public void TrainTestSplit_Deterministic()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var (train1, _) = df.TrainTestSplit(0.2, seed: 42);
        var (train2, _) = df.TrainTestSplit(0.2, seed: 42);

        train1.GetColumn<int>("X")[0].Should().Be(train2.GetColumn<int>("X")[0]);
    }

    [Fact]
    public void TrainTestSplit_Stratified()
    {
        var labels = Enumerable.Range(0, 100).Select(i => i < 80 ? "A" : "B").ToArray();
        var df = new DataFrame(
            new Column<int>("X", Enumerable.Range(0, 100).ToArray()),
            new StringColumn("Label", labels)
        );

        var (train, test) = df.TrainTestSplit(0.2, stratifyBy: "Label", seed: 42);

        // Both splits should have both classes
        var trainLabels = new HashSet<string?>();
        for (int i = 0; i < train.RowCount; i++) trainLabels.Add(train.GetStringColumn("Label")[i]);
        trainLabels.Should().Contain("A");
        trainLabels.Should().Contain("B");
    }

    [Fact]
    public void TrainValTestSplit_ThreeWay()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var (train, val, test) = df.TrainValTestSplit(0.15, 0.15, seed: 42);

        (train.RowCount + val.RowCount + test.RowCount).Should().Be(100);
    }

    [Fact]
    public void KFold_ProducesKFolds()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 100).ToArray()));
        var folds = df.KFold(k: 5, seed: 42).ToList();

        folds.Should().HaveCount(5);
        foreach (var (fold, train, val) in folds)
            (train.RowCount + val.RowCount).Should().Be(100);
    }
}
