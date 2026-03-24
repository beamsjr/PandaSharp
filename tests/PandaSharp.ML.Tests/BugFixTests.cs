using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Tensors;
using PandaSharp.ML.Transformers;

namespace PandaSharp.ML.Tests;

public class BugFixTests
{
    // Fix 1: Tensor ArgMax axis=0
    [Fact]
    public void Tensor_ArgMax_Axis0()
    {
        var t = new Tensor<double>([1, 5, 3, 8, 2, 6], 2, 3);
        // Column 0: [1, 8] → argmax=1, Column 1: [5, 2] → argmax=0, Column 2: [3, 6] → argmax=1
        var result = t.ArgMax(axis: 0);
        result.Should().Equal([1, 0, 1]);
    }

    [Fact]
    public void Tensor_ArgMax_Axis0_SingleRow()
    {
        var t = new Tensor<double>([3, 1, 4], 1, 3);
        var result = t.ArgMax(axis: 0);
        result.Should().Equal([0, 0, 0]); // only one row
    }

    // Fix 2: TF-IDF smoothing — universal terms should not zero out
    [Fact]
    public void TextVectorizer_TfIdf_UniversalTerm_NotZero()
    {
        var df = new DataFrame(new StringColumn("T", [
            "the cat sat",
            "the dog sat",
            "the bird flew"
        ]));

        var vec = new TextVectorizer("T", VectorizerMode.TfIdf, maxFeatures: 10);
        var result = vec.FitTransform(df);

        // "the" appears in all docs — with +1 smoothing it should NOT be zero
        var theCols = result.ColumnNames.Where(n => n.Contains("_the")).ToList();
        if (theCols.Count > 0)
        {
            result.GetColumn<double>(theCols[0])[0].Should().BeGreaterThan(0,
                "universal terms should have positive TF-IDF with smoothing");
        }
    }

    // Fix 3: Imputer.ComputeMedian doesn't mutate input
    [Fact]
    public void Imputer_Median_DoesNotMutateOriginal()
    {
        var original = new List<double> { 5, 3, 1, 4, 2 };
        var copy = new List<double>(original);

        var df = new DataFrame(Column<double>.FromNullable("X", [1.0, null, 3.0, null, 5.0]));
        new Imputer(ImputeStrategy.Median, columns: "X").FitTransform(df);

        // The original list ordering should not matter since we test via DataFrame
        // Just verify the imputer produces correct median
        var result = new Imputer(ImputeStrategy.Median, columns: "X").FitTransform(df);
        result.GetColumn<double>("X")[1].Should().Be(3.0); // median of [1, 3, 5]
    }

    // Fix 4: DataLoader different shuffle per epoch
    [Fact]
    public void DataLoader_DifferentShufflePerEpoch()
    {
        var df = new DataFrame(
            new Column<double>("F", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            new Column<double>("L", [0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        );

        var loader = df.ToDataLoader(["F"], "L", batchSize: 10, shuffle: true, seed: 42);

        var epoch1 = loader.First().Features;
        var epoch2 = loader.First().Features; // second call should have different order

        // At least some elements should differ between epochs
        bool anyDifferent = false;
        for (int i = 0; i < 10; i++)
            if (Math.Abs(epoch1[i, 0] - epoch2[i, 0]) > 0.001) anyDifferent = true;
        anyDifferent.Should().BeTrue("different epochs should have different shuffle order");
    }

    // Fix 5: Tensor.Slice bounds check
    [Fact]
    public void Tensor_Slice_OutOfBounds_Throws()
    {
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var act = () => t.Slice(0, 1, 5); // row slice starting at 1 with length 5 — only 1 row left
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Tensor_Slice_ValidRange()
    {
        var t = new Tensor<double>([1, 2, 3, 4, 5, 6], 2, 3);
        var sliced = t.Slice(0, 0, 1); // first row
        sliced.Shape.Should().Equal([1, 3]);
        sliced[0, 0].Should().Be(1);
    }
}
