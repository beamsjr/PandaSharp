using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests;

public class MoreTransformerTests
{
    // -- TargetEncoder --

    [Fact]
    public void TargetEncoder_EncodesWithSmoothing()
    {
        var df = new DataFrame(
            new StringColumn("City", ["NYC", "NYC", "LA", "LA", "LA"]),
            new Column<double>("Price", [100, 120, 50, 60, 70])
        );

        var encoder = new TargetEncoder(target: "Price", smoothing: 1, columns: "City");
        var result = encoder.FitTransform(df);

        result["City"].DataType.Should().Be(typeof(double));
        // NYC mean = 110, LA mean = 60, global mean = 80
        // With smoothing=1: NYC = (2*110 + 1*80)/(2+1) = 100, LA = (3*60 + 1*80)/(3+1) = 65
        result.GetColumn<double>("City")[0].Should().BeApproximately(100, 1);
    }

    [Fact]
    public void TargetEncoder_UnseenCategory_ReturnsGlobalMean()
    {
        var train = new DataFrame(
            new StringColumn("Cat", ["A", "B"]),
            new Column<double>("Y", [10, 20])
        );
        var test = new DataFrame(
            new StringColumn("Cat", ["A", "C"]),  // C not in training
            new Column<double>("Y", [0, 0])
        );

        var encoder = new TargetEncoder(target: "Y", smoothing: 0, columns: "Cat");
        encoder.Fit(train);
        var result = encoder.Transform(test);

        result.GetColumn<double>("Cat")[1].Should().Be(15); // global mean
    }

    // -- Discretizer --

    [Fact]
    public void Discretizer_UniformBins()
    {
        var df = new DataFrame(new Column<double>("X", [0, 25, 50, 75, 100]));
        var result = new Discretizer(nBins: 4, BinStrategy.Uniform, "X").FitTransform(df);

        result["X"].DataType.Should().Be(typeof(int));
        result.GetColumn<int>("X")[0].Should().Be(0); // bin 0
        result.GetColumn<int>("X")[4].Should().Be(3); // bin 3
    }

    [Fact]
    public void Discretizer_QuantileBins()
    {
        var df = new DataFrame(new Column<double>("X", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]));
        var result = new Discretizer(nBins: 4, BinStrategy.Quantile, "X").FitTransform(df);

        // Quantile bins should distribute roughly equally
        var counts = new int[4];
        for (int i = 0; i < result.RowCount; i++)
            counts[result.GetColumn<int>("X")[i]!.Value]++;
        // Each bin should have ~2-3 elements
        counts.All(c => c >= 1).Should().BeTrue();
    }

    [Fact]
    public void Discretizer_FitThenTransform()
    {
        var train = new DataFrame(new Column<double>("X", [0, 50, 100]));
        var test = new DataFrame(new Column<double>("X", [25, 75]));

        var disc = new Discretizer(nBins: 2, BinStrategy.Uniform, "X");
        disc.Fit(train);
        var result = disc.Transform(test);

        result.GetColumn<int>("X")[0].Should().Be(0); // 25 in [0, 50)
        result.GetColumn<int>("X")[1].Should().Be(1); // 75 in [50, 100]
    }

    // -- Tensor MatMul --

    [Fact]
    public void Tensor_MatMul_2x3_times_3x2()
    {
        var a = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var b = new Tensor<double>([7, 8, 9, 10, 11, 12], 3, 2);
        var c = a.MatMul(b);

        c.Shape.Should().Equal([2, 2]);
        c[0, 0].Should().Be(58);  // 1*7 + 2*9 + 3*11
        c[0, 1].Should().Be(64);  // 1*8 + 2*10 + 3*12
        c[1, 0].Should().Be(139); // 4*7 + 5*9 + 6*11
        c[1, 1].Should().Be(154); // 4*8 + 5*10 + 6*12
    }

    [Fact]
    public void Tensor_MatMul_Identity()
    {
        var a = new Tensor<double>([1, 2, 3, 4], 2, 2);
        var identity = new Tensor<double>([1, 0, 0, 1], 2, 2);
        var result = a.MatMul(identity);

        result[0, 0].Should().Be(1);
        result[0, 1].Should().Be(2);
        result[1, 0].Should().Be(3);
        result[1, 1].Should().Be(4);
    }

    [Fact]
    public void Tensor_MatMul_ShapeMismatch_Throws()
    {
        var a = new Tensor<double>([1, 2, 3, 4], 2, 2);
        var b = new Tensor<double>([1, 2, 3, 4, 5, 6], 3, 2);

        var act = () => a.MatMul(b);
        act.Should().Throw<ArgumentException>().WithMessage("*Shape mismatch*");
    }

    [Fact]
    public void Tensor_Dot_1D()
    {
        var a = new Tensor<double>([1, 2, 3], 3);
        var b = new Tensor<double>([4, 5, 6], 3);

        a.Dot(b).Should().Be(32); // 1*4 + 2*5 + 3*6
    }

    // -- End-to-end ML pipeline --

    [Fact]
    public void EndToEnd_Pipeline_Split_Tensor()
    {
        var df = new DataFrame(
            new Column<double>("Age", [25, 30, 35, 40, 45, 50, 55, 60, 65, 70]),
            new Column<double>("Income", [30, 40, 50, 60, 70, 80, 90, 100, 110, 120]),
            new StringColumn("Category", ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"])
        );

        // 1. Build pipeline
        var pipeline = new Cortex.ML.Pipeline.FeaturePipeline(
            new StandardScaler("Age", "Income"),
            new OneHotEncoder("Category")
        );

        // 2. Split
        var (train, test) = Cortex.ML.Splitting.DataSplitting.TrainTestSplit(df, 0.3, seed: 42);

        // 3. Process
        var trainProcessed = pipeline.FitTransform(train);
        var testProcessed = pipeline.Transform(test);

        // 4. Convert to tensor
        var featureCols = trainProcessed.ColumnNames.ToArray();
        var trainTensor = trainProcessed.ToTensor(featureCols);
        var testTensor = testProcessed.ToTensor(featureCols);

        trainTensor.Rank.Should().Be(2);
        trainTensor.Shape[1].Should().Be(featureCols.Length);
        testTensor.Shape[1].Should().Be(featureCols.Length);
    }
}
