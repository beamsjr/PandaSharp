# PandaSharp.Plot

ScottPlot charting integration for PandaSharp DataFrames.

## Features

- **ScottPlot integration** — render publication-quality charts from DataFrames
- **Common chart types** — scatter, line, bar, histogram, box plot, and heatmap
- **Export to PNG, SVG, or BMP** for reports and presentations
- **Customizable styling** — colors, fonts, axes, legends, and annotations

## Installation

```bash
dotnet add package PandaSharp.Plot
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Plot;

var df = DataFrame.ReadCsv("timeseries.csv");

var plot = df.ScottPlot.Line(x: "date", y: "value");
plot.Title("Sensor Readings Over Time");
plot.SavePng("chart.png", width: 800, height: 400);
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
