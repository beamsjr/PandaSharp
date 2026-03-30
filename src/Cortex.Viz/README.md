# Cortex.Viz

Plotly-powered HTML chart generation and StoryBoard report builder for Cortex.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Interactive Plotly charts** — bar, line, scatter, heatmap, histogram, and more
- **StoryBoard reports** — combine multiple charts into a single HTML dashboard
- **Auto-detect chart types** based on column data types
- **Export to HTML, PNG, or SVG** for embedding in web apps or documents
- **D3.js rendering** for advanced custom visualizations
- **Fluent API** for composing and customizing visualizations

## Installation

```bash
dotnet add package Cortex.Viz
```

## Quick Start

```csharp
using Cortex;
using Cortex.Viz;

var df = DataFrame.ReadCsv("sales.csv");

df.Plot.Bar(x: "month", y: "revenue", color: "region")
  .WithTitle("Monthly Revenue by Region")
  .SaveHtml("report.html");
```

## StoryBoard Dashboards

```csharp
var board = new StoryBoard("Q4 Report");
board.Add(df.Plot.Line(x: "date", y: "revenue"));
board.Add(df.Plot.Histogram("quantity"));
board.Add(df.Describe());  // summary statistics table
board.SaveHtml("dashboard.html");
```

## Export Formats

```csharp
var chart = df.Plot.Scatter(x: "x", y: "y");
chart.SaveHtml("chart.html");   // interactive HTML
chart.SavePng("chart.png");     // static image
chart.SaveSvg("chart.svg");     // vector format
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Plot** | ScottPlot charting (alternative to Plotly) |
| **Cortex.Interactive** | Jupyter notebook inline rendering |
| **Cortex.Notebooks** | Blazor notebook app with inline Plotly |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
