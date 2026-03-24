# PandaSharp

A high-performance, pandas-like DataFrame library for .NET — the first complete data platform native to C#.

**1,246 tests | 15 libraries | Arrow-backed | SIMD-accelerated | .NET 10**

## The Ecosystem

| Package | Description |
|---------|-------------|
| **PandaSharp** | Core DataFrame — Arrow storage, SIMD arithmetic, lazy eval, I/O, profiling |
| **PandaSharp.ML** | Tensors, transformers, cross-validation, metrics |
| **PandaSharp.ML.MLNet** | ML.NET IDataView bridge |
| **PandaSharp.ML.Torch** | TorchSharp tensor bridge |
| **PandaSharp.ML.Onnx** | ONNX Runtime inference |
| **PandaSharp.IO.Database** | SQL push-down, lazy DB scans, connection pooling |
| **PandaSharp.Viz** | Interactive Plotly.js charts + PNG/SVG export |
| **PandaSharp.Streaming** | Real-time event processing with windowed aggregation |
| **PandaSharp.Streaming.Kafka** | Kafka consumer/producer integration |
| **PandaSharp.Streaming.Redis** | Redis Streams source/sink |
| **PandaSharp.Geo** | Geospatial — R-tree index, polygon geometry, GeoParquet, reprojection |
| **PandaSharp.Cloud** | S3, Azure Blob, GCS storage adapters |
| **PandaSharp.Flight** | Arrow Flight RPC for distributed transport |

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Column;

var df = DataFrame.FromDictionary(new() {
    ["Name"] = new string?[] { "Alice", "Bob", "Charlie", "Diana" },
    ["Age"] = new int[] { 25, 30, 35, 28 },
    ["Salary"] = new double[] { 50_000, 62_000, 75_000, 58_000 }
});

Console.WriteLine(df);
// ┌───┬─────────┬─────┬────────┐
// │   │ Name    │ Age │ Salary │
// ├───┼─────────┼─────┼────────┤
// │ 0 │ Alice   │  25 │  50000 │
// │ 1 │ Bob     │  30 │  62000 │
// │ 2 │ Charlie │  35 │  75000 │
// │ 3 │ Diana   │  28 │  58000 │
// └───┴─────────┴─────┴────────┘
```

## Core Operations

```csharp
// Filter, Sort, Select
var result = df.Eval("Salary > 55000");           // string expression filter
var sorted = df.Sort("Salary", ascending: false);  // typed sort
var top3 = df.Head(3);

// GroupBy with typed accumulators
var byDept = df.GroupBy("Department").Sum();

// Joins (typed int fast path)
var joined = orders.Join(customers, "CustomerId");

// Lazy evaluation with optimizer
var lazy = df.Lazy()
    .Filter(Col("Age") > Lit(25))
    .Sort("Salary", ascending: false)
    .Select("Name", "Salary")
    .Head(10)
    .Collect();  // compiles optimized plan, executes once
```

## I/O — Read & Write Anything

```csharp
// Auto-detect format from extension
var df = DataFrameIO.Load("data.parquet");    // or .csv, .csv.gz, .json, .jsonl, .arrow, .xlsx
df.Save("output.csv.gz");                      // gzip-compressed CSV

// Database with SQL push-down
var scanner = new DatabaseScanner(conn, "orders", new PostgresDialect());
var result = scanner.Lazy()
    .Filter(Col("amount") > Lit(1000))
    .Sort("amount", ascending: false)
    .Head(100)
    .Collect();  // executes as single SQL query on the database
```

## Visualization

```csharp
using PandaSharp.Viz;

// Interactive Plotly.js charts
df.Viz().Bar("Month", "Revenue").Title("Sales").ToHtml("chart.html");
df.Viz().Scatter("X", "Y").Title("Correlation").ToHtml("scatter.html");
df.Viz().Histogram("Price").ToHtml("dist.html");
```

## Machine Learning

```csharp
using PandaSharp.ML;

// Train/test split
var (train, test) = DataSplitting.TrainTestSplit(df, testFraction: 0.2);

// Feature pipeline
var pipeline = new FeaturePipeline(new Imputer(ImputeStrategy.Mean), new StandardScaler());
var scaled = pipeline.FitTransform(train.Select("Feature1", "Feature2"));

// Tensors with SIMD
var tensor = df.ToTensor<double>("Col1", "Col2", "Col3");
var result = tensor.MatMul(weights);
```

## Streaming

```csharp
using PandaSharp.Streaming;

StreamFrame.From(new WebSocketSource("ws://localhost:8080/events"))
    .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
    .Agg("price", AggType.Mean, "avg_price")
    .Agg("volume", AggType.Sum, "total_volume")
    .OnEmit(df => Console.WriteLine(df))
    .Start();
```

## Geospatial

```csharp
using PandaSharp.Geo;

var geo = df.ToGeoColumn("lat", "lon");
var nearby = geo.WithinDistance(new GeoPoint(40.7128, -74.0060), radiusKm: 50);
var nearest = RTree.Build(geo).Nearest(target);  // 63x faster than brute-force

// Coordinate reprojection
var utm = geo.Reproject(Crs.Wgs84, Crs.Utm(18));
```

## Performance

| Operation | PandaSharp | Notes |
|-----------|-----------|-------|
| Sum (100K doubles) | 31 us | SIMD, **zero allocation** |
| Filter (100K rows) | 202 us | Boolean mask, branchless |
| Sort (100K rows) | 9.2 ms | Typed struct comparers |
| Parquet read (100K) | 1.7 ms | Zero-boxing typed arrays |
| Join (100K × 1K) | 196 us | Typed int hash join |
| R-tree nearest (10K) | 1.5 us | 63x vs brute-force |
| SQL generation | 377 ns | Full query plan |
| Profile (100K × 3) | 39 ms | Stats + correlation + dedup |

## Samples

```bash
# Core DataFrame operations
dotnet run --project samples/PandaSharp.Samples

# Interactive charts (opens in browser)
dotnet run --project samples/PandaSharp.Samples.Viz

# Database with SQLite
dotnet run --project samples/PandaSharp.Samples.Database

# Database with PostgreSQL
dotnet run --project samples/PandaSharp.Samples.Postgres -- "Host=localhost;Username=postgres;Password=xxx;Database=postgres"

# Machine learning pipeline
dotnet run --project samples/PandaSharp.Samples.ML
```

## Building

```bash
dotnet build
dotnet test
dotnet run --project tests/PandaSharp.Tests.Benchmarks -c Release
```

## License

MIT
