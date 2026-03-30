# Cortex.ML.MLNet

ML.NET integration bridge for Cortex, enabling seamless conversion between DataFrames and IDataView.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+, `Cortex`, and `Microsoft.ML`.

## Features

- **DataFrame to IDataView** conversion with automatic schema mapping
- **IDataView to DataFrame** for post-processing ML.NET results in Cortex
- **Schema inference** — automatically maps .NET types to ML.NET column types
- **Pipeline interop** — use Cortex preprocessing with ML.NET trainers

## Installation

```bash
dotnet add package Cortex.ML.MLNet
```

## Quick Start

```csharp
using Microsoft.ML;
using Cortex;
using Cortex.ML.MLNet;

var mlContext = new MLContext();
var df = DataFrame.ReadCsv("housing.csv");

IDataView dataView = df.ToDataView(mlContext);
var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
    .Append(mlContext.Regression.Trainers.Sdca());

var model = pipeline.Fit(dataView);
```

## Round-Trip Conversion

```csharp
// Cortex → ML.NET → Cortex
IDataView dataView = df.ToDataView(mlContext);
var result = model.Transform(dataView);
var resultDf = result.ToDataFrame();  // back to Cortex DataFrame
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.ML** | Cortex-native ML models (alternative to ML.NET) |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
