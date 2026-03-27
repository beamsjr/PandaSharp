# Cortex.ML.Onnx

ONNX Runtime model inference bridge for Cortex DataFrames.

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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
