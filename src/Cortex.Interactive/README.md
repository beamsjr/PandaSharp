# Cortex.Interactive

Jupyter and Polyglot notebook support for Cortex with DataFrame.Explore() web UI.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Jupyter/Polyglot integration** — rich HTML rendering of DataFrames in notebooks
- **DataFrame.Explore()** — interactive web UI for sorting, filtering, and inspecting data
- **Auto-formatting** — smart display of numbers, dates, and nested types
- **Chart rendering** — inline visualization support in notebook cells

## Installation

```bash
dotnet add package Cortex.Interactive
```

## Quick Start

```csharp
// In a .NET Interactive notebook cell:
#r "nuget: Cortex.Interactive"

using Cortex;
using Cortex.Interactive;

var df = DataFrame.ReadCsv("data.csv");
df.Head(20)           // renders as a rich HTML table
df.Explore()          // opens an interactive data explorer
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Viz** | Plotly charts for inline notebook rendering |
| **Cortex.Notebooks** | Standalone Blazor notebook app (alternative to Jupyter) |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
