# Cortex.Viz

Plotly-powered HTML chart generation and StoryBoard report builder for Cortex.

## Features

- **Interactive Plotly charts** — bar, line, scatter, heatmap, histogram, and more
- **StoryBoard reports** — combine multiple charts into a single HTML dashboard
- **Auto-detect chart types** based on column data types
- **Export to HTML, PNG, or SVG** for embedding in web apps or documents
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

var board = new StoryBoard("Q4 Report");
board.Add(df.Plot.Line(x: "date", y: "revenue"));
board.Add(df.Plot.Histogram("quantity"));
board.SaveHtml("dashboard.html");
```

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
