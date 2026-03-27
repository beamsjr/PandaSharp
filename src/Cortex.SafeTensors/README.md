# Cortex.SafeTensors

HuggingFace SafeTensors format reader/writer for Cortex with memory-mapped tensor loading.

## Features

- **Read SafeTensors files** — load HuggingFace model weights and embeddings
- **Write SafeTensors files** — export tensors in the interoperable SafeTensors format
- **Memory-mapped I/O** — mmap-backed loading for zero-copy, low-memory access
- **Zero-copy tensor views** — access tensor data without deserialization overhead

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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
