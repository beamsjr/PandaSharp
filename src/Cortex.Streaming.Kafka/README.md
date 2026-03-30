# Cortex.Streaming.Kafka

Apache Kafka connector for Cortex Streaming — consumer source and producer sink.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+, `Cortex`, and `Cortex.Streaming`.

## Features

- **Kafka consumer source** — ingest topics as a stream of DataFrames
- **Kafka producer sink** — write DataFrames back to Kafka topics
- **Consumer group support** for scalable parallel consumption
- **Schema-aware deserialization** — JSON, Avro, and raw byte payloads
- **Offset management** — automatic or manual commit strategies

## Installation

```bash
dotnet add package Cortex.Streaming.Kafka
```

## Quick Start

```csharp
using Cortex.Streaming;
using Cortex.Streaming.Kafka;

var stream = new DataFrameStream()
    .From(new KafkaSource("localhost:9092", topic: "events", groupId: "my-app"))
    .Window(TumblingWindow.Of(TimeSpan.FromSeconds(30)))
    .Agg(x => x.Count())
    .To(new KafkaSink("localhost:9092", topic: "event-counts"));

await stream.StartAsync();
```

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Streaming** | Streaming engine (required) |
| **Cortex.Streaming.Redis** | Redis Streams connector |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
