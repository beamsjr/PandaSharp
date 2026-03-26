using TorchSharp;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace PandaSharp.ML.Torch;

/// <summary>
/// Pre-built neural network architectures for common tasks.
/// </summary>
public static class NeuralNetModels
{
    /// <summary>
    /// Creates a simple feedforward multi-layer perceptron (MLP).
    /// Architecture: Linear → [ReLU → Dropout →] ... → Linear(outputDim).
    /// </summary>
    /// <param name="inputDim">Number of input features.</param>
    /// <param name="hiddenDims">Sizes of hidden layers.</param>
    /// <param name="outputDim">Number of output neurons.</param>
    /// <param name="relu">Whether to apply ReLU activation between layers.</param>
    /// <param name="dropout">Dropout probability (0 = no dropout).</param>
    public static Module<Tensor, Tensor> CreateMLP(
        int inputDim, int[] hiddenDims, int outputDim,
        bool relu = true, double dropout = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(inputDim);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputDim);

        var layers = new List<(string name, Module<Tensor, Tensor> module)>();
        int prevDim = inputDim;

        for (int i = 0; i < hiddenDims.Length; i++)
        {
            int dim = hiddenDims[i];
            layers.Add(($"linear_{i}", Linear(prevDim, dim)));

            if (relu)
                layers.Add(($"relu_{i}", ReLU()));

            if (dropout > 0)
                layers.Add(($"dropout_{i}", Dropout(dropout)));

            prevDim = dim;
        }

        layers.Add(("output", Linear(prevDim, outputDim)));

        return Sequential(layers);
    }

    /// <summary>
    /// Creates a single linear layer (linear regression model).
    /// </summary>
    /// <param name="inputDim">Number of input features.</param>
    /// <param name="outputDim">Number of output dimensions.</param>
    public static Module<Tensor, Tensor> CreateLinear(int inputDim, int outputDim)
    {
        return Sequential(
            ("linear", Linear(inputDim, outputDim))
        );
    }
}
