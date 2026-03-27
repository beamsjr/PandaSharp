using Xunit;
using FluentAssertions;
using Cortex.ML.Tensors;
using Cortex.SafeTensors;

namespace Cortex.SafeTensors.Tests;

public class SafeTensorsRoundTripTests
{
    [Fact]
    public void RoundTrip_float_tensor()
    {
        var path = Path.GetTempFileName();
        try
        {
            var original = new Tensor<float>([1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f], 2, 3);

            new SafeTensorWriter()
                .Add("weights", original)
                .Save(path);

            using var reader = SafeTensorReader.Open(path);
            var loaded = reader.GetTensor<float>("weights");

            loaded.Shape.Should().BeEquivalentTo(new[] { 2, 3 });
            loaded.ToArray().Should().BeEquivalentTo(original.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RoundTrip_multiple_tensors()
    {
        var path = Path.GetTempFileName();
        try
        {
            var t1 = new Tensor<float>([1.0f, 2.0f], 2);
            var t2 = new Tensor<float>([3.0f, 4.0f, 5.0f], 3);
            var t3 = new Tensor<float>([6.0f, 7.0f, 8.0f, 9.0f], 2, 2);

            new SafeTensorWriter()
                .Add("a", t1)
                .Add("b", t2)
                .Add("c", t3)
                .Save(path);

            using var reader = SafeTensorReader.Open(path);

            var loadedA = reader.GetTensor<float>("a");
            loadedA.Shape.Should().BeEquivalentTo(new[] { 2 });
            loadedA.ToArray().Should().BeEquivalentTo(t1.ToArray());

            var loadedB = reader.GetTensor<float>("b");
            loadedB.Shape.Should().BeEquivalentTo(new[] { 3 });
            loadedB.ToArray().Should().BeEquivalentTo(t2.ToArray());

            var loadedC = reader.GetTensor<float>("c");
            loadedC.Shape.Should().BeEquivalentTo(new[] { 2, 2 });
            loadedC.ToArray().Should().BeEquivalentTo(t3.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RoundTrip_metadata()
    {
        var path = Path.GetTempFileName();
        try
        {
            var tensor = new Tensor<float>([1.0f], 1);

            new SafeTensorWriter()
                .Add("x", tensor)
                .AddMetadata("format", "pandasharp")
                .AddMetadata("version", "1.0")
                .Save(path);

            using var reader = SafeTensorReader.Open(path);
            var metadata = reader.GetMetadata();

            metadata.Should().ContainKey("format").WhoseValue.Should().Be("pandasharp");
            metadata.Should().ContainKey("version").WhoseValue.Should().Be("1.0");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void GetTensorNames_returns_all_names()
    {
        var path = Path.GetTempFileName();
        try
        {
            new SafeTensorWriter()
                .Add("alpha", new Tensor<float>([1.0f], 1))
                .Add("beta", new Tensor<float>([2.0f], 1))
                .Add("gamma", new Tensor<float>([3.0f], 1))
                .Save(path);

            using var reader = SafeTensorReader.Open(path);
            var names = reader.GetTensorNames();

            names.Should().BeEquivalentTo(["alpha", "beta", "gamma"]);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void RoundTrip_double_tensor()
    {
        var path = Path.GetTempFileName();
        try
        {
            var original = new Tensor<double>([1.1, 2.2, 3.3, 4.4], 2, 2);

            new SafeTensorWriter()
                .Add("doubles", original)
                .Save(path);

            using var reader = SafeTensorReader.Open(path);
            var loaded = reader.GetTensor<double>("doubles");

            loaded.Shape.Should().BeEquivalentTo(new[] { 2, 2 });
            loaded.ToArray().Should().BeEquivalentTo(original.ToArray());
        }
        finally
        {
            File.Delete(path);
        }
    }
}

public class SafeTensorsNullGuardTests
{
    [Fact]
    public void Open_null_path_throws_ArgumentNullException()
    {
        var act = () => SafeTensorReader.Open((string)null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void GetTensor_null_name_throws_ArgumentNullException()
    {
        var path = Path.GetTempFileName();
        try
        {
            new SafeTensorWriter()
                .Add("x", new Tensor<float>([1.0f], 1))
                .Save(path);

            using var reader = SafeTensorReader.Open(path);

            var act = () => reader.GetTensor<float>(null!);

            act.Should().Throw<ArgumentNullException>();
        }
        finally
        {
            File.Delete(path);
        }
    }
}
