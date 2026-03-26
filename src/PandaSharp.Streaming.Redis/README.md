# PandaSharp.Streaming.Redis

Redis Streams connector for PandaSharp Streaming.

## Features

- **Redis Streams source** — consume stream entries as DataFrames
- **Redis Streams sink** — write DataFrames to Redis Streams
- **Consumer group support** for distributed processing
- **Automatic acknowledgment** and pending entry recovery

## Installation

```bash
dotnet add package PandaSharp.Streaming.Redis
```

## Quick Start

```csharp
using PandaSharp.Streaming;
using PandaSharp.Streaming.Redis;

var stream = new DataFrameStream()
    .From(new RedisSource("localhost:6379", streamKey: "sensor-data", group: "processors"))
    .Window(TumblingWindow.Of(TimeSpan.FromSeconds(10)))
    .GroupBy("device_id")
    .Agg(x => x.Mean("value"))
    .To(new RedisSink("localhost:6379", streamKey: "aggregated"));

await stream.StartAsync();
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
