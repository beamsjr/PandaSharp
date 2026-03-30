# Cortex.Plot

ScottPlot charting integration for Cortex DataFrames.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **ScottPlot integration** — render publication-quality charts from DataFrames
- **Common chart types** — scatter, line, bar, histogram, box plot, and heatmap
- **Export to PNG, SVG, or BMP** for reports and presentations
- **Customizable styling** — colors, fonts, axes, legends, and annotations

## Installation

```bash
dotnet add package Cortex.Plot
```

## Quick Start

```csharp
using Cortex;
using Cortex.Plot;

var df = DataFrame.ReadCsv("timeseries.csv");

var plot = df.ScottPlot.Line(x: "date", y: "value");
plot.Title("Sensor Readings Over Time");
plot.SavePng("chart.png", width: 800, height: 400);
```

## Multiple Series

```csharp
var plot = df.ScottPlot.Scatter(x: "x", y: "actual");
plot.Add.Scatter(df["x"], df["predicted"]);
plot.Legend.IsVisible = true;
plot.SavePng("comparison.png");
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Viz** | Plotly-based interactive charts (alternative) |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
