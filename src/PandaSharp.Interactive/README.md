# PandaSharp.Interactive

Jupyter and Polyglot notebook support for PandaSharp with DataFrame.Explore() web UI.

## Features

- **Jupyter/Polyglot integration** — rich HTML rendering of DataFrames in notebooks
- **DataFrame.Explore()** — interactive web UI for sorting, filtering, and inspecting data
- **Auto-formatting** — smart display of numbers, dates, and nested types
- **Chart rendering** — inline visualization support in notebook cells

## Installation

```bash
dotnet add package PandaSharp.Interactive
```

## Quick Start

```csharp
// In a .NET Interactive notebook cell:
#r "nuget: PandaSharp.Interactive"

using PandaSharp;
using PandaSharp.Interactive;

var df = DataFrame.ReadCsv("data.csv");
df.Head(20)           // renders as a rich HTML table
df.Explore()          // opens an interactive data explorer
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
