# PandaSharp.GPU

ILGPU-based GPU acceleration for PandaSharp with CUDA, OpenCL, and CPU fallback.

## Features

- **GPU-accelerated operations** — element-wise math, aggregations, and matrix ops on the GPU
- **Multi-backend** — CUDA, OpenCL, and CPU fallback via ILGPU
- **Automatic device selection** — picks the best available accelerator
- **DataFrame integration** — drop-in GPU acceleration for PandaSharp operations
- **Custom kernels** — write your own ILGPU kernels over DataFrame columns

## Installation

```bash
dotnet add package PandaSharp.GPU
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.GPU;

var df = DataFrame.ReadCsv("large_dataset.csv");

// Enable GPU acceleration
using var gpu = new GpuContext();
var result = gpu.Accelerate(df)
    .Select("col1 * col2 + col3")
    .Sum("result");

Console.WriteLine($"Sum: {result}");
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
