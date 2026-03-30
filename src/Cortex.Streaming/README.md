# Cortex.Streaming

Real-time DataFrame streaming engine with windowed aggregations for Cortex.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Micro-batch streaming** — process real-time data as a stream of DataFrames
- **Windowed aggregations** — tumbling, sliding, and session windows
- **Stateful processing** — maintain running state across micro-batches
- **Pluggable sources and sinks** — extensible connector architecture
- **Backpressure handling** for reliable stream processing

## Installation

```bash
dotnet add package Cortex.Streaming
```

## Quick Start

```csharp
using Cortex;
using Cortex.Streaming;

var stream = new DataFrameStream()
    .From(new CsvSource("events/*.csv", pollingInterval: TimeSpan.FromSeconds(5)))
    .Window(TumblingWindow.Of(TimeSpan.FromMinutes(1)))
    .GroupBy("sensor_id")
    .Agg(x => x.Mean("temperature"))
    .To(new ConsoleSink());

await stream.StartAsync();
```

## Window Types

```csharp
// Tumbling windows (non-overlapping)
.Window(TumblingWindow.Of(TimeSpan.FromMinutes(1)))

// Sliding windows (overlapping)
.Window(SlidingWindow.Of(size: TimeSpan.FromMinutes(5), slide: TimeSpan.FromMinutes(1)))

// Session windows (gap-based)
.Window(SessionWindow.WithGap(TimeSpan.FromMinutes(10)))
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Streaming.Kafka** | Apache Kafka source and sink |
| **Cortex.Streaming.Redis** | Redis Streams source and sink |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
