using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;
using PandaSharp.ML.Pipeline;

namespace PandaSharp.ML.Tests;

public class TransformerTests
{
    // -- StandardScaler --

    [Fact]
    public void StandardScaler_NormalizesColumns()
    {
        var df = new DataFrame(new Column<double>("X", [10, 20, 30, 40, 50]));
        var scaler = new StandardScaler("X");
        var result = scaler.FitTransform(df);

        var col = result.GetColumn<double>("X");
        // Mean should be ~0
        double mean = 0;
        for (int i = 0; i < col.Length; i++) mean += col[i]!.Value;
        (mean / col.Length).Should().BeApproximately(0, 0.001);
    }

    [Fact]
    public void StandardScaler_Transform_UsesFittedParams()
    {
        var train = new DataFrame(new Column<double>("X", [0, 10, 20]));
        var test = new DataFrame(new Column<double>("X", [10, 30]));

        var scaler = new StandardScaler("X");
        scaler.Fit(train);
        var result = scaler.Transform(test);

        // Test data scaled using train's mean/std
        result.GetColumn<double>("X")[0].Should().BeApproximately(0, 0.01); // 10 is the mean of [0,10,20]
    }

    // -- MinMaxScaler --

    [Fact]
    public void MinMaxScaler_ScalesTo01()
    {
        var df = new DataFrame(new Column<double>("X", [0, 50, 100]));
        var result = new MinMaxScaler("X").FitTransform(df);

        result.GetColumn<double>("X")[0].Should().Be(0);
        result.GetColumn<double>("X")[1].Should().Be(0.5);
        result.GetColumn<double>("X")[2].Should().Be(1);
    }

    // -- LabelEncoder --

    [Fact]
    public void LabelEncoder_EncodesCategories()
    {
        var df = new DataFrame(new StringColumn("Color", ["Red", "Blue", "Red", "Green"]));
        var encoder = new LabelEncoder("Color");
        var result = encoder.FitTransform(df);

        result["Color"].DataType.Should().Be(typeof(int));
        var mapping = encoder.GetMapping("Color")!;
        mapping.Should().HaveCount(3);
    }

    [Fact]
    public void LabelEncoder_TransformUnseenCategory_ReturnsNull()
    {
        var train = new DataFrame(new StringColumn("Color", ["Red", "Blue"]));
        var test = new DataFrame(new StringColumn("Color", ["Red", "Yellow"]));

        var encoder = new LabelEncoder("Color");
        encoder.Fit(train);
        var result = encoder.Transform(test);

        result["Color"].IsNull(1).Should().BeTrue(); // "Yellow" not seen in training
    }

    // -- OneHotEncoder --

    [Fact]
    public void OneHotEncoder_CreatesColumns()
    {
        var df = new DataFrame(
            new StringColumn("Color", ["Red", "Blue", "Green"]),
            new Column<int>("Value", [1, 2, 3])
        );

        var result = new OneHotEncoder("Color").FitTransform(df);

        result.ColumnNames.Should().NotContain("Color");
        result.ColumnNames.Should().Contain("Color_Red");
        result.ColumnNames.Should().Contain("Color_Blue");
        result.ColumnNames.Should().Contain("Color_Green");
        result.GetColumn<int>("Color_Red")[0].Should().Be(1);
        result.GetColumn<int>("Color_Red")[1].Should().Be(0);
    }

    // -- Imputer --

    [Fact]
    public void Imputer_Mean_FillsMissing()
    {
        var df = new DataFrame(Column<double>.FromNullable("X", [10.0, null, 30.0]));
        var result = new Imputer(ImputeStrategy.Mean, columns: "X").FitTransform(df);

        result.GetColumn<double>("X")[1].Should().Be(20.0); // mean of 10 and 30
    }

    [Fact]
    public void Imputer_Median_FillsMissing()
    {
        var df = new DataFrame(Column<double>.FromNullable("X", [10.0, null, 30.0, 40.0]));
        var result = new Imputer(ImputeStrategy.Median, columns: "X").FitTransform(df);

        result.GetColumn<double>("X")[1].Should().Be(30.0); // median of [10, 30, 40]
    }

    // -- Pipeline --

    [Fact]
    public void Pipeline_ComposesTransformers()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("Age", [25.0, null, 35.0]),
            new StringColumn("Color", ["Red", "Blue", "Red"])
        );

        var pipeline = new FeaturePipeline(
            new Imputer(ImputeStrategy.Mean, columns: "Age"),
            new StandardScaler("Age"),
            new OneHotEncoder("Color")
        );

        var result = pipeline.FitTransform(df);

        result["Age"].NullCount.Should().Be(0);
        result.ColumnNames.Should().Contain("Color_Red");
        result.ColumnNames.Should().NotContain("Color");
    }

    [Fact]
    public void Pipeline_FitThenTransform()
    {
        var train = new DataFrame(new Column<double>("X", [0, 10, 20]));
        var test = new DataFrame(new Column<double>("X", [5, 15]));

        var pipeline = new FeaturePipeline(new StandardScaler("X"));
        pipeline.Fit(train);
        var result = pipeline.Transform(test);

        result.RowCount.Should().Be(2);
    }
}
