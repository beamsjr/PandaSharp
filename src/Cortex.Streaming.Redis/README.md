# Cortex.Streaming.Redis

Redis Streams connector for Cortex Streaming.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+, `Cortex`, and `Cortex.Streaming`.

## Features

- **Redis Streams source** — consume stream entries as DataFrames
- **Redis Streams sink** — write DataFrames to Redis Streams
- **Consumer group support** for distributed processing
- **Automatic acknowledgment** and pending entry recovery

## Installation

```bash
dotnet add package Cortex.Streaming.Redis
```

## Quick Start

```csharp
using Cortex.Streaming;
using Cortex.Streaming.Redis;

var stream = new DataFrameStream()
    .From(new RedisSource("localhost:6379", streamKey: "sensor-data", group: "processors"))
    .Window(TumblingWindow.Of(TimeSpan.FromSeconds(10)))
    .GroupBy("device_id")
    .Agg(x => x.Mean("value"))
    .To(new RedisSink("localhost:6379", streamKey: "aggregated"));

await stream.StartAsync();
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Streaming** | Streaming engine (required) |
| **Cortex.Streaming.Kafka** | Apache Kafka connector |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
