# PandaSharp.ML.MLNet

ML.NET integration bridge for PandaSharp, enabling seamless conversion between DataFrames and IDataView.

## Features

- **DataFrame to IDataView** conversion with automatic schema mapping
- **IDataView to DataFrame** for post-processing ML.NET results in PandaSharp
- **Schema inference** — automatically maps .NET types to ML.NET column types
- **Pipeline interop** — use PandaSharp preprocessing with ML.NET trainers

## Installation

```bash
dotnet add package PandaSharp.ML.MLNet
```

## Quick Start

```csharp
using Microsoft.ML;
using PandaSharp;
using PandaSharp.ML.MLNet;

var mlContext = new MLContext();
var df = DataFrame.ReadCsv("housing.csv");

IDataView dataView = df.ToDataView(mlContext);
var pipeline = mlContext.Transforms.NormalizeMinMax("Features")
    .Append(mlContext.Regression.Trainers.Sdca());

var model = pipeline.Fit(dataView);
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
