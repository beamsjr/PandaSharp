# Cortex.SafeTensors

HuggingFace SafeTensors format reader/writer for Cortex with memory-mapped tensor loading.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+.

## Features

- **Read SafeTensors files** — load HuggingFace model weights and embeddings
- **Write SafeTensors files** — export tensors in the interoperable SafeTensors format
- **Memory-mapped I/O** — mmap-backed loading for zero-copy, low-memory access
- **Zero-copy tensor views** — access tensor data without deserialization overhead
- **Multi-file support** — load sharded model weights across multiple files

## Installation

```bash
dotnet add package Cortex.SafeTensors
```

## Quick Start

```csharp
using Cortex.ML;
using Cortex.SafeTensors;

// Load model weights
var tensors = SafeTensorsFile.Load("model.safetensors");
var weights = tensors["encoder.weight"];  // zero-copy tensor view

// Save tensors
var output = new SafeTensorsBuilder();
output.Add("embeddings", embeddingTensor);
output.Save("embeddings.safetensors");
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex.ML** | ML models and tensor operations |
| **Cortex.ML.Torch** | TorchSharp training with SafeTensors model saving |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
