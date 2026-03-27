using FluentAssertions;
using Cortex.ML.Torch;
using TorchSharp;
using Xunit;

namespace Cortex.ML.Torch.Tests;

public class NeuralNetModelsTests
{
    [Fact]
    public void CreateMLP_HasCorrectLayerCount()
    {
        var model = NeuralNetModels.CreateMLP(
            inputDim: 4,
            hiddenDims: [16, 8],
            outputDim: 1);

        // 2 hidden layers + 1 output = 3 linear layers
        // Each hidden has linear + relu = 2 modules, output has 1 = total 5
        var parameters = model.named_parameters().ToList();
        // 3 linear layers = 3 weight + 3 bias = 6 parameters
        parameters.Should().HaveCount(6);
    }

    [Fact]
    public void CreateMLP_CorrectInputOutputDimensions()
    {
        var model = NeuralNetModels.CreateMLP(
            inputDim: 10,
            hiddenDims: [32, 16],
            outputDim: 3);

        using var input = torch.randn(5, 10);
        using var output = model.call(input);

        output.shape.Should().BeEquivalentTo(new long[] { 5, 3 });
    }

    [Fact]
    public void CreateMLP_WithDropout_DoesNotThrow()
    {
        var model = NeuralNetModels.CreateMLP(
            inputDim: 4,
            hiddenDims: [8],
            outputDim: 1,
            dropout: 0.5);

        using var input = torch.randn(2, 4);
        using var output = model.call(input);

        output.shape.Should().BeEquivalentTo(new long[] { 2, 1 });
    }

    [Fact]
    public void CreateLinear_CorrectDimensions()
    {
        var model = NeuralNetModels.CreateLinear(inputDim: 5, outputDim: 1);

        using var input = torch.randn(3, 5);
        using var output = model.call(input);

        output.shape.Should().BeEquivalentTo(new long[] { 3, 1 });
    }
}
