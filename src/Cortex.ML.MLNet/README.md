# Cortex.ML.MLNet

ML.NET integration bridge for Cortex, enabling seamless conversion between DataFrames and IDataView.

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

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
