using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Tensors;
using PandaSharp.ML.Transformers;
using PandaSharp.ML.Pipeline;
using PandaSharp.ML.Splitting;
using PandaSharp.ML.Metrics;

namespace PandaSharp.ML.Tests;

public class MLHardeningTests
{
    // ===== Tensor edge cases =====

    [Fact]
    public void Tensor_ZeroLength()
    {
        var t = Tensor<double>.Zeros(0);
        t.Length.Should().Be(0);
        t.Sum().Should().Be(0);
    }

    [Fact]
    public void Tensor_1x1()
    {
        var t = new Tensor<double>([42.0], 1, 1);
        t[0, 0].Should().Be(42);
        t.Transpose()[0, 0].Should().Be(42);
    }

    [Fact]
    public void Tensor_MatMul_1x1()
    {
        var a = new Tensor<double>([3.0], 1, 1);
        var b = new Tensor<double>([4.0], 1, 1);
        a.MatMul(b)[0, 0].Should().Be(12);
    }

    [Fact]
    public void Tensor_SubtractSelf_IsZero()
    {
        var t = new Tensor<double>([1, 2, 3], 3);
        var result = t - t;
        result.Sum().Should().Be(0);
    }

    // ===== Scaler edge cases =====

    [Fact]
    public void StandardScaler_ConstantColumn_NoNaN()
    {
        var df = new DataFrame(new Column<double>("X", [5, 5, 5, 5]));
        var result = new StandardScaler("X").FitTransform(df);
        // std=0 → scaler uses 1.0 as fallback → all values become 0
        result.GetColumn<double>("X")[0].Should().Be(0);
    }

    [Fact]
    public void MinMaxScaler_ConstantColumn_NoNaN()
    {
        var df = new DataFrame(new Column<double>("X", [5, 5, 5]));
        var result = new MinMaxScaler("X").FitTransform(df);
        // range=0 → fallback → should not produce NaN
        double.IsNaN(result.GetColumn<double>("X")[0]!.Value).Should().BeFalse();
    }

    [Fact]
    public void StandardScaler_AllNulls()
    {
        var df = new DataFrame(Column<double>.FromNullable("X", [null, null, null]));
        var result = new StandardScaler("X").FitTransform(df);
        // All NaN since all inputs are null
        result.GetColumn<double>("X")[0].Should().NotBe(0); // NaN
    }

    [Fact]
    public void OneHotEncoder_SingleCategory()
    {
        var df = new DataFrame(new StringColumn("C", ["A", "A", "A"]));
        var result = new OneHotEncoder("C").FitTransform(df);
        result.ColumnNames.Should().Contain("C_A");
        result.GetColumn<int>("C_A")[0].Should().Be(1);
    }

    [Fact]
    public void LabelEncoder_EmptyString()
    {
        var df = new DataFrame(new StringColumn("C", ["", "a", ""]));
        var result = new LabelEncoder("C").FitTransform(df);
        // Empty string is a valid category
        result.GetColumn<int>("C")[0].Should().Be(result.GetColumn<int>("C")[2]);
    }

    // ===== Pipeline edge cases =====

    [Fact]
    public void Pipeline_EmptyPipeline()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var pipeline = new FeaturePipeline();
        var result = pipeline.FitTransform(df);
        result.ContentEquals(df).Should().BeTrue();
    }

    [Fact]
    public void Pipeline_SingleStep()
    {
        var df = new DataFrame(new Column<double>("X", [0, 50, 100]));
        var pipeline = new FeaturePipeline(new MinMaxScaler("X"));
        var result = pipeline.FitTransform(df);
        result.GetColumn<double>("X")[2].Should().Be(1.0);
    }

    // ===== Splitting edge cases =====

    [Fact]
    public void TrainTestSplit_SmallDataset()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3, 4, 5]));
        var (train, test) = df.TrainTestSplit(0.4, seed: 42);
        (train.RowCount + test.RowCount).Should().Be(5);
    }

    [Fact]
    public void KFold_K1()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 10).ToArray()));
        var folds = df.KFold(k: 1, seed: 42).ToList();
        folds.Should().HaveCount(1);
        folds[0].Val.RowCount.Should().Be(10); // all data in validation
    }

    [Fact]
    public void KFold_KEqualsN()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 5).ToArray()));
        var folds = df.KFold(k: 5, shuffle: false).ToList();
        folds.Should().HaveCount(5);
        foreach (var (_, _, val) in folds)
            val.RowCount.Should().Be(1); // leave-one-out
    }

    // ===== Metrics edge cases =====

    [Fact]
    public void Classification_AllSameClass()
    {
        var yTrue = new Column<bool>("T", [true, true, true]);
        var yPred = new Column<bool>("P", [true, true, true]);
        var m = MetricsCalculator.Classification(yTrue, yPred);
        m.Accuracy.Should().Be(1.0);
        m.TrueNegative.Should().Be(0);
    }

    [Fact]
    public void Regression_SinglePoint()
    {
        var yTrue = new Column<double>("T", [1.0]);
        var yPred = new Column<double>("P", [1.0]);
        var m = MetricsCalculator.Regression(yTrue, yPred);
        m.MSE.Should().Be(0);
    }

    // ===== DataLoader edge cases =====

    [Fact]
    public void DataLoader_BatchSizeLargerThanData()
    {
        var df = new DataFrame(
            new Column<double>("F", [1, 2]),
            new Column<double>("L", [0, 1])
        );
        var loader = df.ToDataLoader(["F"], "L", batchSize: 100, shuffle: false);
        var batches = loader.ToList();
        batches.Should().HaveCount(1);
        batches[0].Features.Shape[0].Should().Be(2);
    }

    [Fact]
    public void DataLoader_BatchSize1()
    {
        var df = new DataFrame(
            new Column<double>("F", [1, 2, 3]),
            new Column<double>("L", [0, 1, 0])
        );
        var loader = df.ToDataLoader(["F"], "L", batchSize: 1, shuffle: false);
        var batches = loader.ToList();
        batches.Should().HaveCount(3);
        batches[0].Features.Shape[0].Should().Be(1);
    }

    // ===== PolynomialFeatures edge cases =====

    [Fact]
    public void PolynomialFeatures_SingleColumn()
    {
        var df = new DataFrame(new Column<double>("X", [2.0, 3.0]));
        var result = new PolynomialFeatures(degree: 2, "X").FitTransform(df);
        result.ColumnNames.Should().Contain("X^2");
        result.ColumnNames.Should().NotContain("X*X"); // no self-interaction
    }

    // ===== TextVectorizer edge cases =====

    [Fact]
    public void TextVectorizer_EmptyDocument()
    {
        var df = new DataFrame(new StringColumn("T", ["hello world", "", "foo"]));
        var result = new TextVectorizer("T", maxFeatures: 5).FitTransform(df);
        result.RowCount.Should().Be(3);
    }

    [Fact]
    public void TextVectorizer_SingleWord()
    {
        var df = new DataFrame(new StringColumn("T", ["hello", "hello", "hello"]));
        var result = new TextVectorizer("T", VectorizerMode.Count, maxFeatures: 5).FitTransform(df);
        result.ColumnNames.Should().Contain("T_hello");
    }

    // ===== Discretizer edge cases =====

    [Fact]
    public void Discretizer_AllSameValue()
    {
        var df = new DataFrame(new Column<double>("X", [5, 5, 5, 5]));
        var result = new Discretizer(nBins: 3, columns: "X").FitTransform(df);
        // All same value → all should be in the same bin
        var bin0 = result.GetColumn<int>("X")[0];
        for (int i = 1; i < 4; i++)
            result.GetColumn<int>("X")[i].Should().Be(bin0);
    }
}
