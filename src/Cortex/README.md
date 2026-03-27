# Cortex

A high-performance, pandas-like DataFrame library for .NET backed by Apache Arrow with SIMD acceleration.

## Features

- **Arrow-backed columnar storage** for cache-friendly, zero-copy data access
- **SIMD-accelerated** operations for blazing-fast numeric computation
- **Rich I/O** — CSV, Parquet, Excel, JSON, and HTML out of the box
- **Expressive API** — filtering, grouping, joining, pivoting, and reshaping
- **Native C accelerators** for quantile and aggregation hot paths

## Installation

```bash
dotnet add package Cortex
```

## Quick Start

```csharp
using Cortex;

var df = DataFrame.ReadCsv("sales.csv");

var summary = df.GroupBy("region")
    .Agg(x => x.Sum("revenue"), x => x.Mean("quantity"))
    .SortBy("revenue", ascending: false);

summary.Head(10).Print();
```

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
