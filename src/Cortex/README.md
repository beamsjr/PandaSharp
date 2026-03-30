# Cortex

A high-performance, pandas-like DataFrame library for .NET backed by Apache Arrow with SIMD acceleration.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+

## Features

- **Arrow-backed columnar storage** for cache-friendly, zero-copy data access
- **SIMD-accelerated** operations for blazing-fast numeric computation
- **Rich I/O** — CSV, Parquet, Excel, JSON, Avro, ORC, and HTML out of the box
- **Expressive API** — filtering, grouping, joining, pivoting, and reshaping
- **Lazy evaluation** — deferred query execution with filter/sort/select optimization
- **Native C accelerators** for quantile and aggregation hot paths
- **Dictionary-encoded strings** for O(K) categorical operations

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

## Filtering and Boolean Indexing

```csharp
// Boolean indexing
var senior = df[df["Age"].Gt(28)];

// Query expressions
var filtered = df.Query("Salary > 55000");

// Arithmetic operators on columns
var bonus = df.GetColumn<double>("Salary") * 0.1;
```

## Lazy Evaluation

```csharp
var result = df.Lazy()
    .Filter(Col("Age") > Lit(25))
    .Sort("Salary", ascending: false)
    .Select("Name", "Salary")
    .Head(10)
    .Collect();
```

## I/O

```csharp
// Auto-detect format from extension
var df = DataFrameIO.Load("data.parquet");  // .csv, .json, .arrow, .xlsx, .avro, .orc
df.Save("output.csv.gz");                    // gzip-compressed
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex.ML** | Machine learning models and transformers |
| **Cortex.Viz** | Plotly chart generation and dashboards |
| **Cortex.GPU** | GPU-accelerated DataFrame operations |
| **Cortex.IO.Database** | SQL database I/O with push-down |
| **Cortex.TimeSeries** | ARIMA, SARIMA, and Holt-Winters forecasting |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
