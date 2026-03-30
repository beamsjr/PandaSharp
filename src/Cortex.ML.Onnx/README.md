# Cortex.ML.Onnx

ONNX Runtime model inference bridge for Cortex DataFrames.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Load and run ONNX models** directly on Cortex DataFrames
- **Automatic input/output mapping** between DataFrame columns and model tensors
- **Hardware acceleration** — CPU, CUDA, and DirectML execution providers
- **Batch inference** for high-throughput prediction workloads

## Installation

```bash
dotnet add package Cortex.ML.Onnx
```

## Quick Start

```csharp
using Cortex;
using Cortex.ML.Onnx;

var df = DataFrame.ReadCsv("features.csv");

var session = new OnnxModel("model.onnx");
DataFrame predictions = session.Predict(df);

predictions.Head(5).Print();
```

## GPU Inference

```csharp
// Use CUDA execution provider for GPU inference
var session = new OnnxModel("model.onnx", executionProvider: "CUDA");
DataFrame predictions = session.Predict(largeBatch);
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML** | Classical ML models and training |
| **Cortex.ML.Torch** | TorchSharp GPU training |
| **Cortex.Vision** | Vision models with ONNX inference |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
