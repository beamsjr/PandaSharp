using System.Text.Json;
using System.Threading.Channels;
using Confluent.Kafka;

namespace Cortex.Streaming.Kafka;

/// <summary>
/// Kafka consumer source for Cortex Streaming.
/// Connects to a Kafka topic and emits StreamEvents from JSON messages.
///
/// Usage:
///   var source = new KafkaSource(new KafkaSourceConfig
///   {
///       BootstrapServers = "localhost:9092",
///       Topic = "events",
///       GroupId = "my-consumer-group"
///   });
///   StreamFrame.From(source)
///       .Window(new TumblingWindow(TimeSpan.FromMinutes(1)))
///       .Agg("value", AggType.Sum, "total")
///       .OnEmit(df => Console.WriteLine(df))
///       .Start();
/// </summary>
public class KafkaSource : IStreamSource
{
    private readonly KafkaSourceConfig _config;
    private IConsumer<string, string>? _consumer;

    public KafkaSource(KafkaSourceConfig config) => _config = config;

    public async Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken)
    {
        var consumerConfig = new ConsumerConfig
        {
            BootstrapServers = _config.BootstrapServers,
            GroupId = _config.GroupId,
            AutoOffsetReset = _config.AutoOffsetReset,
            EnableAutoCommit = _config.EnableAutoCommit,
            EnableAutoOffsetStore = false // manual offset tracking for exactly-once
        };

        _consumer = new ConsumerBuilder<string, string>(consumerConfig).Build();
        _consumer.Subscribe(_config.Topic);

        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                ConsumeResult<string, string>? result;
                try
                {
                    result = _consumer.Consume(TimeSpan.FromMilliseconds(100));
                }
                catch (ConsumeException)
                {
                    continue;
                }

                if (result is null) continue;

                var evt = ParseMessage(result);
                if (evt is not null)
                {
                    await writer.WriteAsync(evt, cancellationToken);

                    // Store offset after successful processing
                    if (!_config.EnableAutoCommit)
                        _consumer.StoreOffset(result);
                }
            }
        }
        catch (OperationCanceledException) { }
        finally
        {
            if (!_config.EnableAutoCommit)
                _consumer.Commit();
            writer.TryComplete();
        }
    }

    private StreamEvent? ParseMessage(ConsumeResult<string, string> result)
    {
        try
        {
            using var doc = JsonDocument.Parse(result.Message.Value);
            var root = doc.RootElement;

            // Extract timestamp from message or Kafka metadata
            DateTimeOffset timestamp;
            if (_config.TimestampField is not null &&
                root.TryGetProperty(_config.TimestampField, out var tsProp) &&
                tsProp.ValueKind == JsonValueKind.String)
            {
                timestamp = DateTimeOffset.Parse(tsProp.GetString()!);
            }
            else
            {
                timestamp = result.Message.Timestamp.UtcDateTime;
            }

            var data = new Dictionary<string, object?>();
            // Add Kafka metadata
            if (_config.IncludeMetadata)
            {
                data["_kafka_topic"] = result.Topic;
                data["_kafka_partition"] = result.Partition.Value;
                data["_kafka_offset"] = result.Offset.Value;
                data["_kafka_key"] = result.Message.Key;
            }

            // Extract JSON fields
            foreach (var prop in root.EnumerateObject())
            {
                if (prop.Name == _config.TimestampField) continue;
                data[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.Number => prop.Value.TryGetInt64(out long l) && prop.Value.GetDouble() == l
                        ? (object)l : prop.Value.GetDouble(),
                    JsonValueKind.String => prop.Value.GetString(),
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.Null => null,
                    _ => prop.Value.ToString()
                };
            }

            return new StreamEvent(timestamp, data);
        }
        catch
        {
            return null;
        }
    }

    public ValueTask DisposeAsync()
    {
        _consumer?.Close();
        _consumer?.Dispose();
        return ValueTask.CompletedTask;
    }
}

public class KafkaSourceConfig
{
    public string BootstrapServers { get; set; } = "localhost:9092";
    public string Topic { get; set; } = "";
    public string GroupId { get; set; } = "pandasharp-consumer";
    public AutoOffsetReset AutoOffsetReset { get; set; } = AutoOffsetReset.Latest;
    public bool EnableAutoCommit { get; set; } = false;
    public string? TimestampField { get; set; } = "timestamp";
    public bool IncludeMetadata { get; set; } = false;
}
