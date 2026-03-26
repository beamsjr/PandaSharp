# PandaSharp.Streaming

Real-time DataFrame streaming engine with windowed aggregations for PandaSharp.

## Features

- **Micro-batch streaming** — process real-time data as a stream of DataFrames
- **Windowed aggregations** — tumbling, sliding, and session windows
- **Stateful processing** — maintain running state across micro-batches
- **Pluggable sources and sinks** — extensible connector architecture
- **Backpressure handling** for reliable stream processing

## Installation

```bash
dotnet add package PandaSharp.Streaming
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.Streaming;

var stream = new DataFrameStream()
    .From(new CsvSource("events/*.csv", pollingInterval: TimeSpan.FromSeconds(5)))
    .Window(TumblingWindow.Of(TimeSpan.FromMinutes(1)))
    .GroupBy("sensor_id")
    .Agg(x => x.Mean("temperature"))
    .To(new ConsoleSink());

await stream.StartAsync();
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
