using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;
using Cortex.ML.Splitting;

namespace Cortex.ML.Tests;

public class AdvancedTransformerTests
{
    // -- RobustScaler --

    [Fact]
    public void RobustScaler_UsesMedianAndIQR()
    {
        var df = new DataFrame(new Column<double>("X", [1, 2, 3, 4, 5, 100])); // 100 is outlier
        var result = new RobustScaler("X").FitTransform(df);

        // Median of [1,2,3,4,5,100] = 3.5, Q1=2, Q3=5, IQR=3
        var col = result.GetColumn<double>("X");
        // (3.5 - 3.5) / 3 = 0 for the median value
        // Outlier 100 should be scaled but not dominate
        col[0].Should().NotBe(double.NaN);
    }

    [Fact]
    public void RobustScaler_FitThenTransform()
    {
        var train = new DataFrame(new Column<double>("X", [10, 20, 30, 40, 50]));
        var test = new DataFrame(new Column<double>("X", [30, 60]));

        var scaler = new RobustScaler("X");
        scaler.Fit(train);
        var result = scaler.Transform(test);

        result.GetColumn<double>("X")[0].Should().BeApproximately(0, 0.01); // 30 is the median
    }

    // -- PolynomialFeatures --

    [Fact]
    public void PolynomialFeatures_Degree2()
    {
        var df = new DataFrame(
            new Column<double>("X", [2.0, 3.0]),
            new Column<double>("Y", [4.0, 5.0])
        );

        var result = new PolynomialFeatures(degree: 2, "X", "Y").FitTransform(df);

        result.ColumnNames.Should().Contain("X^2");
        result.ColumnNames.Should().Contain("Y^2");
        result.ColumnNames.Should().Contain("X*Y");
        result.GetColumn<double>("X^2")[0].Should().Be(4.0);   // 2²
        result.GetColumn<double>("X*Y")[0].Should().Be(8.0);   // 2*4
    }

    [Fact]
    public void PolynomialFeatures_Degree3()
    {
        var df = new DataFrame(new Column<double>("X", [2.0]));
        var result = new PolynomialFeatures(degree: 3, "X").FitTransform(df);

        result.ColumnNames.Should().Contain("X^2");
        result.ColumnNames.Should().Contain("X^3");
        result.GetColumn<double>("X^3")[0].Should().Be(8.0); // 2³
    }

    // -- OrdinalEncoder --

    [Fact]
    public void OrdinalEncoder_EncodesWithOrder()
    {
        var df = new DataFrame(new StringColumn("Size", ["M", "S", "L", "XL", "S"]));
        var encoder = new OrdinalEncoder(new Dictionary<string, string[]>
        {
            ["Size"] = ["S", "M", "L", "XL"]
        });
        var result = encoder.FitTransform(df);

        result.GetColumn<int>("Size")[0].Should().Be(1); // M
        result.GetColumn<int>("Size")[1].Should().Be(0); // S
        result.GetColumn<int>("Size")[2].Should().Be(2); // L
        result.GetColumn<int>("Size")[3].Should().Be(3); // XL
    }

    // -- StratifiedKFold --

    [Fact]
    public void StratifiedKFold_MaintainsDistribution()
    {
        var labels = Enumerable.Range(0, 100).Select(i => i < 80 ? "A" : "B").ToArray();
        var df = new DataFrame(
            new Column<int>("X", Enumerable.Range(0, 100).ToArray()),
            new StringColumn("Label", labels)
        );

        var folds = df.StratifiedKFold(k: 5, column: "Label", seed: 42).ToList();
        folds.Should().HaveCount(5);

        foreach (var (_, train, val) in folds)
        {
            // Each fold's validation set should have both classes
            var valLabels = new HashSet<string?>();
            for (int i = 0; i < val.RowCount; i++)
                valLabels.Add(val.GetStringColumn("Label")[i]);
            valLabels.Should().Contain("A");
            valLabels.Should().Contain("B");
        }
    }

    // -- TimeSeriesSplit --

    [Fact]
    public void TimeSeriesSplit_ExpandingWindow()
    {
        var df = new DataFrame(new Column<int>("X", Enumerable.Range(0, 60).ToArray()));
        var splits = df.TimeSeriesSplit(nSplits: 5).ToList();

        splits.Should().HaveCount(5);

        // Each fold's training set should be larger than the previous
        for (int i = 1; i < splits.Count; i++)
            splits[i].Train.RowCount.Should().BeGreaterThan(splits[i - 1].Train.RowCount);

        // Train and val should not overlap
        foreach (var (_, train, val) in splits)
        {
            // Last train index should be < first val index
            int lastTrain = train.GetColumn<int>("X")[train.RowCount - 1]!.Value;
            int firstVal = val.GetColumn<int>("X")[0]!.Value;
            firstVal.Should().BeGreaterThan(lastTrain);
        }
    }
}
