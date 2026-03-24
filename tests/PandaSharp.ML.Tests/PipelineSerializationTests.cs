using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;
using PandaSharp.ML.Pipeline;

namespace PandaSharp.ML.Tests;

public class PipelineSerializationTests
{
    [Fact]
    public void Serialize_ProducesJson()
    {
        var pipeline = new FeaturePipeline(
            new Imputer(ImputeStrategy.Mean),
            new StandardScaler("Age"),
            new OneHotEncoder("Color")
        );

        var json = pipeline.SerializeToJson();

        json.Should().Contain("version");
        json.Should().Contain("Imputer");
        json.Should().Contain("StandardScaler");
        json.Should().Contain("OneHotEncoder");
    }

    [Fact]
    public void Serialize_ToBytes()
    {
        var pipeline = new FeaturePipeline(new StandardScaler("X"));
        var bytes = pipeline.Serialize();
        bytes.Length.Should().BeGreaterThan(0);

        // Verify it's valid JSON
        var json = System.Text.Encoding.UTF8.GetString(bytes);
        json.Should().Contain("StandardScaler");
    }

    [Fact]
    public void Serialize_PreservesStepOrder()
    {
        var pipeline = new FeaturePipeline(
            new MinMaxScaler("A"),
            new LabelEncoder("B"),
            new RobustScaler("C")
        );

        var json = pipeline.SerializeToJson();

        int posMinMax = json.IndexOf("MinMaxScaler");
        int posLabel = json.IndexOf("LabelEncoder");
        int posRobust = json.IndexOf("RobustScaler");

        posMinMax.Should().BeLessThan(posLabel);
        posLabel.Should().BeLessThan(posRobust);
    }
}
