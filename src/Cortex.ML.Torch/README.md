# Cortex.ML.Torch

TorchSharp GPU training bridge for Cortex with neural network model support.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+, `Cortex`, and `Cortex.ML`. GPU training requires a CUDA-capable device.

## Features

- **DataFrame to Tensor** zero-copy conversion for efficient data loading
- **GPU training** — CUDA-accelerated model training via TorchSharp
- **Built-in neural networks** — MLP, CNN, and custom module support
- **DataLoader integration** — batched, shuffled iteration over DataFrames
- **SafeTensors support** — save and load model weights in HuggingFace format
- **Training utilities** — learning rate schedulers, early stopping, checkpointing

## Installation

```bash
dotnet add package Cortex.ML.Torch
```

## Quick Start

```csharp
using Cortex;
using Cortex.ML.Torch;

var df = DataFrame.ReadCsv("train.csv");
var dataset = df.ToTorchDataset(labelColumn: "target");
var loader = new DataLoader(dataset, batchSize: 64, shuffle: true);

var model = new MLP(inputDim: 10, hiddenDim: 64, outputDim: 1);
var trainer = new Trainer(model, lr: 1e-3, epochs: 20, device: Device.CUDA);
trainer.Fit(loader);
```

## Custom Neural Networks

```csharp
var nn = NeuralNetModels.CreateMLP(inputDim: 20, hiddenDims: [64, 32], outputDim: 1);
var result = TorchTrainer.Train(nn, trainDf, features, "target",
    torch.nn.MSELoss(), new TrainingConfig { Epochs = 50, Device = "auto" });
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML** | Classical ML models (required) |
| **Cortex.ML.Onnx** | Export and run ONNX models |
| **Cortex.SafeTensors** | HuggingFace SafeTensors format I/O |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
