# PandaSharp.ML.Torch

TorchSharp GPU training bridge for PandaSharp with neural network model support.

## Features

- **DataFrame to Tensor** zero-copy conversion for efficient data loading
- **GPU training** — CUDA-accelerated model training via TorchSharp
- **Built-in neural networks** — MLP, CNN, and custom module support
- **DataLoader integration** — batched, shuffled iteration over DataFrames
- **SafeTensors support** — save and load model weights in HuggingFace format

## Installation

```bash
dotnet add package PandaSharp.ML.Torch
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.ML.Torch;

var df = DataFrame.ReadCsv("train.csv");
var dataset = df.ToTorchDataset(labelColumn: "target");
var loader = new DataLoader(dataset, batchSize: 64, shuffle: true);

var model = new MLP(inputDim: 10, hiddenDim: 64, outputDim: 1);
var trainer = new Trainer(model, lr: 1e-3, epochs: 20, device: Device.CUDA);
trainer.Fit(loader);
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
