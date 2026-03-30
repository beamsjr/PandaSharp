# Cortex

A high-performance data science platform for .NET — DataFrame, ML, GPU, Vision, NLP, Time Series, and more.

**2,344 tests | 22 packages | Arrow-backed | SIMD + BLAS + GPU accelerated | .NET 10**

[![Build](https://github.com/beamsjr/PandaSharp/actions/workflows/build-test.yml/badge.svg)](https://github.com/beamsjr/PandaSharp/actions/workflows/build-test.yml)

## Install

```bash
dotnet add package Cortex           # Core DataFrame
dotnet add package Cortex.ML        # Machine Learning
dotnet add package Cortex.GPU       # GPU Acceleration
dotnet add package Cortex.Vision    # Image/Video Processing
dotnet add package Cortex.Text      # NLP Pipeline
dotnet add package Cortex.TimeSeries # Forecasting
dotnet add package Cortex.Notebooks  # Interactive Notebook App
```

## The Ecosystem

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame — Arrow storage, SIMD arithmetic, lazy eval, CSV/Parquet/JSON/Avro/ORC I/O |
| **Cortex.ML** | 25+ models (Linear, Trees, KNN, KMeans, PCA, t-SNE), transformers, metrics, BLAS/LAPACK |
| **Cortex.ML.Torch** | TorchSharp GPU training, neural networks, SafeTensors loading |
| **Cortex.ML.MLNet** | ML.NET IDataView bridge |
| **Cortex.ML.Onnx** | ONNX Runtime model inference |
| **Cortex.GPU** | ILGPU-based GPU acceleration — CUDA, OpenCL, CPU fallback |
| **Cortex.Vision** | Image transforms, ONNX classifiers/embedders, video decode (FFmpeg + Apple AVFoundation) |
| **Cortex.Text** | Tokenizers (BPE, WordPiece), stemming, embeddings, Levenshtein, Jaro-Winkler |
| **Cortex.TimeSeries** | ARIMA, SARIMA, Holt-Winters, AutoARIMA, seasonal decomposition, ACF/PACF |
| **Cortex.SafeTensors** | HuggingFace SafeTensors format — memory-mapped reader/writer |
| **Cortex.Viz** | Plotly.js charts, StoryBoard narrative reports, PNG/SVG export |
| **Cortex.Plot** | ScottPlot charting integration |
| **Cortex.Interactive** | Jupyter/Polyglot notebooks, `DataFrame.Explore()` web UI |
| **Cortex.Geo** | R-tree spatial index, Haversine distance, coordinate transforms, GeoParquet |
| **Cortex.IO.Database** | SQL push-down, lazy DB scans (Postgres, SQL Server, SQLite, MySQL) |
| **Cortex.Streaming** | Real-time event processing with windowed aggregation |
| **Cortex.Streaming.Kafka** | Apache Kafka connector |
| **Cortex.Streaming.Redis** | Redis Streams connector |
| **Cortex.Cloud** | S3, Azure Blob, GCS storage with retry + circuit breaker |
| **Cortex.Flight** | Apache Arrow Flight RPC |
| **Cortex.Notebooks** | Blazor notebook web app with C# scripting, Plotly viz, and AI chat |
| **Cortex.MCP** | Model Context Protocol (MCP) server for AI-driven notebook interaction |

## Quick Start

```csharp
using Cortex;
using Cortex.Column;

var df = DataFrame.FromDictionary(new() {
    ["Name"]   = new[] { "Alice", "Bob", "Charlie", "Diana" },
    ["Age"]    = new[] { 25, 30, 35, 28 },
    ["Salary"] = new[] { 50_000.0, 62_000, 75_000, 58_000 }
});

// Boolean indexing
var senior = df[df["Age"].Gt(28)];

// Arithmetic operators
var bonus = df.GetColumn<double>("Salary") * 0.1;

// GroupBy with multi-column aggregation
var stats = df.GroupBy("Department")
    .Agg(("Salary", AggFunc.Mean), ("Salary", AggFunc.Max));

// Method chaining
var result = df.Pipe(Normalize).Pipe(Encode).Pipe(Split);

Console.WriteLine(df);
```

## Performance vs Python

Benchmarked on 14.7M rows (6,179 stocks), 50K ML samples. [Full results](benchmarks/).

| Suite | vs Python | Highlights |
|-------|-----------|------------|
| **DataFrame** (20 categories) | **2.4x faster** | GroupBy 78x, Correlation 28x, String ops 11x |
| **TimeSeries** (8 categories) | **131x faster** | Holt-Winters 699x, AutoARIMA 258x, ARIMA 33x |
| **Text/NLP** (6 categories) | **7.5x faster** | Stemming 12x, Bigrams 2.4x |
| **ML Models** (21 categories) | 0.9x | GBT 1.7x faster, LinearReg 3x (BLAS) |

## Core Operations

```csharp
// Filter, Sort, Describe
var filtered = df.Query("Salary > 55000");
var sorted = df.Sort("Salary", ascending: false);
var stats = df.Describe();  // count, mean, std, min, quartiles, max

// Joins
var joined = orders.Join(customers, "CustomerId");

// Lazy evaluation
var result = df.Lazy()
    .Filter(Col("Age") > Lit(25))
    .Sort("Salary", ascending: false)
    .Select("Name", "Salary")
    .Head(10)
    .Collect();
```

## I/O

```csharp
// Auto-detect format
var df = DataFrameIO.Load("data.parquet");  // .csv, .json, .arrow, .xlsx, .avro, .orc
df.Save("output.csv.gz");                    // gzip-compressed

// Database with SQL push-down
var scanner = new DatabaseScanner(conn, "orders", new PostgresDialect());
var result = scanner.Lazy()
    .Filter(Col("amount") > Lit(1000))
    .Head(100)
    .Collect();  // single SQL query
```

## Machine Learning

```csharp
using Cortex.ML.Models;
using Cortex.ML.Tensors;

var X = df.ToTensor<double>("Feature1", "Feature2", "Feature3");
var y = df.ToTensor<double>("Target");

// Train (uses BLAS/LAPACK when available)
var model = new RandomForestRegressor(nEstimators: 100, maxDepth: 10);
model.Fit(X, y);

var predictions = model.Predict(X_test);
var r2 = model.Score(X_test, y_test);

// GPU training with TorchSharp
var nn = NeuralNetModels.CreateMLP(inputDim: 20, hiddenDims: [64, 32], outputDim: 1);
var result = TorchTrainer.Train(nn, trainDf, features, "target",
    torch.nn.MSELoss(), new TrainingConfig { Epochs = 50, Device = "auto" });
```

## Time Series

```csharp
using Cortex.TimeSeries.Models;

var model = new ARIMA(p: 2, d: 1, q: 1);
model.Fit(historicalData);
var forecast = model.Forecast(steps: 30);

// Automatic model selection
var auto = new AutoARIMA(maxP: 5, maxD: 2, maxQ: 5);
auto.Fit(data);
```

## Vision

```csharp
using Cortex.Vision;

var images = ImageIO.LoadBatch(paths, resizeWidth: 224, resizeHeight: 224);
var pipeline = new ImagePipeline()
    .Add(new Resize(224, 224))
    .Add(new Normalize(ImageNet.Mean, ImageNet.Std));
var processed = pipeline.Transform(images);
```

## GPU Acceleration

```csharp
using Cortex.GPU;

// Automatic device selection (CUDA > OpenCL > CPU)
var corr = df.GpuCorr();
var dist = X.GpuPairwiseDistances(Y);
var product = A.GpuMatMul(B);
```

## Notebooks

Cortex.Notebooks is a self-hosted Blazor web app for interactive C# data analysis — like Jupyter, but native .NET.

```bash
dotnet run --project src/Cortex.Notebooks
```

- Code cells with C# scripting (Roslyn) and full IntelliSense via Monaco editor
- Inline Plotly chart rendering from Cortex.Viz
- Markdown cells for documentation
- AI chat panel for natural-language data exploration
- SQLite-backed notebook persistence
- MCP server (`Cortex.MCP`) for AI tools (Claude, etc.) to read/write/run notebook cells

## Architecture

- **Arrow-backed columnar storage** with zero-copy slicing
- **SIMD acceleration** via `Vector<T>` for bulk numeric operations
- **Native C accelerators** (libpandasharp) for rolling windows, aggregations, string ops
- **Apple Accelerate BLAS/LAPACK** for matrix operations (cblas_dgemm, dgesv, dgesvd)
- **C++ std::nth_element** for O(n) quantile computation
- **KD-tree** for O(K log N) nearest neighbor search
- **ILGPU** for GPU-accelerated operations
- **Dictionary-encoded strings** for O(K) categorical operations

## Building

```bash
dotnet build
dotnet test
```

## License

MIT
