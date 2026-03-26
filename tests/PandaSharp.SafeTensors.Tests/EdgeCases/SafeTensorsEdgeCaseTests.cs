using FluentAssertions;
using PandaSharp.ML.Tensors;
using PandaSharp.SafeTensors;
using Xunit;

namespace PandaSharp.SafeTensors.Tests.EdgeCases;

public class SafeTensorsEdgeCaseTests : IDisposable
{
    private readonly List<string> _tempFiles = new();

    private string GetTempFile()
    {
        var path = Path.Combine(Path.GetTempPath(), $"st_edge_{Guid.NewGuid():N}.safetensors");
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

    // === SafeTensorWriter edge cases ===

    [Fact]
    public void Writer_ZeroTensors_ShouldProduceValidFile()
    {
        // Writing 0 tensors should produce a valid file with just header
        var path = GetTempFile();
        var writer = new SafeTensorWriter();
        writer.Save(path);

        // Should be readable
        using var reader = SafeTensorReader.Open(path);
        reader.GetTensorNames().Should().BeEmpty();
    }

    [Fact]
    public void Writer_ZeroTensors_StreamRoundTrip()
    {
        // Writing 0 tensors to a stream should produce a valid result
        using var ms = new MemoryStream();
        var writer = new SafeTensorWriter();
        writer.Save(ms);

        ms.Position = 0;
        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().BeEmpty();
    }

    [Fact]
    public void Writer_DuplicateTensorNames_ShouldThrow()
    {
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(new float[] { 1.0f, 2.0f }, 2);
        writer.Add("test", tensor);

        var act = () => writer.Add("test", tensor);
        act.Should().Throw<ArgumentException>().WithMessage("*already been added*");
    }

    [Fact]
    public void Writer_VeryLargeTensorName_ShouldWork()
    {
        var name = new string('A', 1000);
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(new float[] { 1.0f }, 1);
        writer.Add(name, tensor);

        using var ms = new MemoryStream();
        writer.Save(ms);

        ms.Position = 0;
        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().Contain(name);
        var result = reader.GetTensor<float>(name);
        result.Span[0].Should().Be(1.0f);
    }

    [Fact]
    public void Writer_EmptyTensor_ZeroElements_ShouldWork()
    {
        // A tensor with shape [0] has 0 elements
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(Array.Empty<float>(), 0);
        writer.Add("empty", tensor);

        using var ms = new MemoryStream();
        writer.Save(ms);

        ms.Position = 0;
        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().Contain("empty");
        var result = reader.GetTensor<float>("empty");
        result.Length.Should().Be(0);
    }

    // === SafeTensorReader edge cases ===

    [Fact]
    public void Reader_GetTensor_NonExistentName_ShouldThrow()
    {
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(new float[] { 1.0f }, 1);
        writer.Add("exists", tensor);

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var act = () => reader.GetTensor<float>("nonexistent");
        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void Reader_GetTensor_WrongType_F32ReadAsInt_ShouldConvert()
    {
        // Write as F32, read as int - should do numeric conversion
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(new float[] { 1.0f, 2.0f, 3.0f }, 3);
        writer.Add("floats", tensor);

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        // This should go through the ConvertData path and convert float -> int
        var result = reader.GetTensor<int>("floats");
        result.Span[0].Should().Be(1);
        result.Span[1].Should().Be(2);
        result.Span[2].Should().Be(3);
    }

    [Fact]
    public void Reader_TruncatedFile_ShouldThrow()
    {
        // Create a valid file then truncate it
        var writer = new SafeTensorWriter();
        var tensor = new Tensor<float>(new float[] { 1.0f, 2.0f, 3.0f, 4.0f }, 4);
        writer.Add("data", tensor);

        using var ms = new MemoryStream();
        writer.Save(ms);
        var bytes = ms.ToArray();

        // Truncate: only keep half the data
        var truncated = bytes[..(bytes.Length - 8)];
        using var tms = new MemoryStream(truncated);

        // Reading header works, but accessing truncated tensor data should throw
        using var reader = SafeTensorReader.Open(tms);
        var act = () => reader.GetTensor<float>("data");
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void Reader_CorruptHeaderSize_ShouldThrow()
    {
        // Write a massive header size that exceeds stream length
        using var ms = new MemoryStream();
        var headerSize = BitConverter.GetBytes((ulong)999999999);
        ms.Write(headerSize);
        ms.Write(new byte[] { 0, 0 }); // tiny stream
        ms.Position = 0;

        var act = () => SafeTensorReader.Open(ms);
        act.Should().Throw<Exception>(); // InvalidDataException or EndOfStreamException
    }

    // === Round-trip correctness ===

    [Fact]
    public void RoundTrip_F32_ExactValues()
    {
        var values = new float[] { 0f, 1f, -1f, float.MaxValue, float.MinValue, float.Epsilon, MathF.PI };
        var writer = new SafeTensorWriter();
        writer.Add("f32", new Tensor<float>(values, values.Length));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<float>("f32");
        for (int i = 0; i < values.Length; i++)
            result.Span[i].Should().Be(values[i]);
    }

    [Fact]
    public void RoundTrip_F64_ExactValues()
    {
        var values = new double[] { 0, 1, -1, double.MaxValue, double.MinValue, double.Epsilon, Math.PI };
        var writer = new SafeTensorWriter();
        writer.Add("f64", new Tensor<double>(values, values.Length));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<double>("f64");
        for (int i = 0; i < values.Length; i++)
            result.Span[i].Should().Be(values[i]);
    }

    [Fact]
    public void RoundTrip_I32_ExactValues()
    {
        var values = new int[] { 0, 1, -1, int.MaxValue, int.MinValue, 42 };
        var writer = new SafeTensorWriter();
        writer.Add("i32", new Tensor<int>(values, values.Length));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<int>("i32");
        for (int i = 0; i < values.Length; i++)
            result.Span[i].Should().Be(values[i]);
    }

    [Fact]
    public void RoundTrip_I64_ExactValues()
    {
        var values = new long[] { 0, 1, -1, long.MaxValue, long.MinValue, 42 };
        var writer = new SafeTensorWriter();
        writer.Add("i64", new Tensor<long>(values, values.Length));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<long>("i64");
        for (int i = 0; i < values.Length; i++)
            result.Span[i].Should().Be(values[i]);
    }

    [Fact]
    public void RoundTrip_MultipleTensors_AllTypesInOneFile()
    {
        var writer = new SafeTensorWriter();
        writer.Add("f32", new Tensor<float>(new float[] { 1.5f, 2.5f }, 2));
        writer.Add("i32", new Tensor<int>(new int[] { 10, 20 }, 2));
        writer.Add("f64", new Tensor<double>(new double[] { 3.14 }, 1));
        writer.Add("i64", new Tensor<long>(new long[] { long.MaxValue }, 1));
        writer.AddMetadata("format", "test");

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        reader.GetTensorNames().Should().HaveCount(4);
        reader.GetMetadata()["format"].Should().Be("test");

        reader.GetTensor<float>("f32").Span[0].Should().Be(1.5f);
        reader.GetTensor<int>("i32").Span[1].Should().Be(20);
        reader.GetTensor<double>("f64").Span[0].Should().Be(3.14);
        reader.GetTensor<long>("i64").Span[0].Should().Be(long.MaxValue);
    }

    [Fact]
    public void RoundTrip_2DShape_Preserved()
    {
        var data = new float[] { 1, 2, 3, 4, 5, 6 };
        var writer = new SafeTensorWriter();
        writer.Add("matrix", new Tensor<float>(data, 2, 3));

        using var ms = new MemoryStream();
        writer.Save(ms);
        ms.Position = 0;

        using var reader = SafeTensorReader.Open(ms);
        var result = reader.GetTensor<float>("matrix");
        result.Shape.Should().BeEquivalentTo(new[] { 2, 3 });
        result.Span[0].Should().Be(1f);
        result.Span[5].Should().Be(6f);
    }
}
