using FluentAssertions;
using Cortex.ML.Tensors;
using Cortex.SafeTensors;
using Xunit;

namespace Cortex.SafeTensors.Tests.EdgeCases;

public class SafeTensorsEdgeCaseTests2 : IDisposable
{
    private readonly List<string> _tempFiles = new();

    private string GetTempFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"st_edge2_{Guid.NewGuid():N}.safetensors");
        _tempFiles.Add(path);
        return path;
    }

    public void Dispose()
    {
        foreach (var f in _tempFiles)
        {
            try { File.Delete(f); } catch { }
        }
    }

    [Fact]
    public void Metadata_RoundTrip_Preserved()
    {
        // Write tensors with metadata, verify metadata is preserved on read
        var writer = new SafeTensorWriter();
        writer.Add("weights", new Tensor<float>(new float[] { 1.0f, 2.0f }, 2));
        writer.AddMetadata("format", "pt");
        writer.AddMetadata("framework", "pandasharp");
        writer.AddMetadata("version", "1.0.0");

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var metadata = reader.GetMetadata();

        metadata.Should().ContainKey("format");
        metadata["format"].Should().Be("pt");
        metadata.Should().ContainKey("framework");
        metadata["framework"].Should().Be("pandasharp");
        metadata.Should().ContainKey("version");
        metadata["version"].Should().Be("1.0.0");
    }

    [Fact]
    public void Tensor_With_Unicode_Name_RoundTrip()
    {
        var name = "weights_\u00e9\u00e8\u00ea_\u4e2d\u6587_\ud83d\ude00";
        var writer = new SafeTensorWriter();
        writer.Add(name, new Tensor<float>(new float[] { 3.14f }, 1));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().Contain(name);
        var result = reader.GetTensor<float>(name);
        result.Span[0].Should().Be(3.14f);
    }

    [Fact]
    public void Tensor_With_Spaces_And_Slashes_In_Name_RoundTrip()
    {
        var name = "layer 1/weights/kernel";
        var writer = new SafeTensorWriter();
        writer.Add(name, new Tensor<float>(new float[] { 1.0f, 2.0f }, 2));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().Contain(name);
        var result = reader.GetTensor<float>(name);
        result.Span[0].Should().Be(1.0f);
        result.Span[1].Should().Be(2.0f);
    }

    [Fact]
    public void GetTensor_F32_Read_As_Double_Should_Convert()
    {
        // Write as F32, read as F64 (double) - should do type conversion
        var writer = new SafeTensorWriter();
        writer.Add("data", new Tensor<float>(new float[] { 1.5f, 2.5f, 3.5f }, 3));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<double>("data");

        result.Span[0].Should().BeApproximately(1.5, 1e-5);
        result.Span[1].Should().BeApproximately(2.5, 1e-5);
        result.Span[2].Should().BeApproximately(3.5, 1e-5);
    }

    [Fact]
    public void Metadata_With_Empty_Value_RoundTrip()
    {
        var writer = new SafeTensorWriter();
        writer.Add("t", new Tensor<float>(new float[] { 1.0f }, 1));
        writer.AddMetadata("key_with_empty_value", "");

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var metadata = reader.GetMetadata();
        metadata.Should().ContainKey("key_with_empty_value");
        metadata["key_with_empty_value"].Should().Be("");
    }

    [Fact]
    public void File_RoundTrip_Metadata_Preserved()
    {
        // Test file-based (memory-mapped) round trip of metadata
        var path = GetTempFile();
        var writer = new SafeTensorWriter();
        writer.Add("w", new Tensor<float>(new float[] { 1.0f }, 1));
        writer.AddMetadata("model_type", "transformer");
        writer.AddMetadata("hidden_size", "768");
        writer.Save(path);

        using var reader = SafeTensorReader.Open(path);
        var metadata = reader.GetMetadata();
        metadata["model_type"].Should().Be("transformer");
        metadata["hidden_size"].Should().Be("768");
    }
}
