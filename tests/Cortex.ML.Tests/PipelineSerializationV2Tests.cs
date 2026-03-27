using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;
using Cortex.ML.Pipeline;

namespace Cortex.ML.Tests;

public class PipelineSerializationV2Tests
{
    [Fact]
    public void Serialize_IncludesLearnedParams_StandardScaler()
    {
        var df = new DataFrame(new Column<double>("X", [10, 20, 30]));
        var pipeline = new FeaturePipeline(new StandardScaler("X"));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();

        json.Should().Contain("\"version\": 2");
        json.Should().Contain("StandardScaler");
        json.Should().Contain("Mean");
        json.Should().Contain("Std");
        json.Should().Contain("20"); // mean of [10,20,30]
    }

    [Fact]
    public void Serialize_IncludesLearnedParams_MinMaxScaler()
    {
        var df = new DataFrame(new Column<double>("X", [0, 50, 100]));
        var pipeline = new FeaturePipeline(new MinMaxScaler("X"));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("Min");
        json.Should().Contain("Max");
    }

    [Fact]
    public void Serialize_IncludesLearnedParams_LabelEncoder()
    {
        var df = new DataFrame(new StringColumn("C", ["Red", "Blue", "Green"]));
        var pipeline = new FeaturePipeline(new LabelEncoder("C"));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("Red");
        json.Should().Contain("Blue");
        json.Should().Contain("Green");
    }

    [Fact]
    public void Serialize_IncludesLearnedParams_OneHotEncoder()
    {
        var df = new DataFrame(new StringColumn("C", ["A", "B"]));
        var pipeline = new FeaturePipeline(new OneHotEncoder("C"));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("A"); // vocabulary
    }

    [Fact]
    public void Serialize_IncludesLearnedParams_Imputer()
    {
        var df = new DataFrame(Column<double>.FromNullable("X", [10.0, null, 30.0]));
        var pipeline = new FeaturePipeline(new Imputer(ImputeStrategy.Mean, columns: "X"));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("20"); // mean fill value
    }

    [Fact]
    public void Serialize_MultiStepPipeline()
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
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("Imputer");
        json.Should().Contain("StandardScaler");
        json.Should().Contain("OneHotEncoder");
        json.Should().Contain("params"); // all steps have params
    }

    [Fact]
    public void Serialize_UnfittedPipeline_ParamsNull()
    {
        var pipeline = new FeaturePipeline(new StandardScaler("X"));
        var json = pipeline.SerializeToJson();
        json.Should().Contain("null"); // unfitted → params is null
    }

    [Fact]
    public void Serialize_TextVectorizer()
    {
        var df = new DataFrame(new StringColumn("T", ["hello world", "foo bar"]));
        var pipeline = new FeaturePipeline(new TextVectorizer("T", maxFeatures: 5));
        pipeline.FitTransform(df);

        var json = pipeline.SerializeToJson();
        json.Should().Contain("vocabulary");
        json.Should().Contain("idfWeights");
    }
}
