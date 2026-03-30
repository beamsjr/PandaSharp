# Cortex.GPU

ILGPU-based GPU acceleration for Cortex with CUDA, OpenCL, and CPU fallback.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package. GPU support requires CUDA or OpenCL drivers; falls back to CPU automatically.

## Features

- **GPU-accelerated operations** — element-wise math, aggregations, and matrix ops on the GPU
- **Multi-backend** — CUDA, OpenCL, and CPU fallback via ILGPU
- **Automatic device selection** — picks the best available accelerator
- **DataFrame integration** — drop-in GPU acceleration for Cortex operations
- **Custom kernels** — write your own ILGPU kernels over DataFrame columns

## Installation

```bash
dotnet add package Cortex.GPU
```

## Quick Start

```csharp
using Cortex;
using Cortex.GPU;

var df = DataFrame.ReadCsv("large_dataset.csv");

// Enable GPU acceleration
using var gpu = new GpuContext();
var result = gpu.Accelerate(df)
    .Select("col1 * col2 + col3")
    .Sum("result");

Console.WriteLine($"Sum: {result}");
```

## GPU Matrix Operations

```csharp
// Automatic device selection (CUDA > OpenCL > CPU)
var corr = df.GpuCorr();
var dist = X.GpuPairwiseDistances(Y);
var product = A.GpuMatMul(B);
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML.Torch** | TorchSharp GPU training for neural networks |
| **Cortex.ML** | Classical ML with BLAS acceleration |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
