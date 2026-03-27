using System.Text.Json;
using Confluent.Kafka;
using Cortex;

namespace Cortex.Streaming.Kafka;

/// <summary>
/// Kafka producer sink for Cortex Streaming.
/// Publishes DataFrame rows as JSON messages to a Kafka topic.
///
/// Usage:
///   var sink = new KafkaSink("localhost:9092", "output-topic");
///   StreamFrame.From(source)
///       .Window(new TumblingWindow(TimeSpan.FromMinutes(1)))
///       .Agg("value", AggType.Sum, "total")
///       .OnEmit(df => sink.Publish(df))
///       .Start();
///   sink.Flush();
/// </summary>
public class KafkaSink : IDisposable
{
    private readonly IProducer<string, string> _producer;
    private readonly string _topic;
    private readonly string? _keyColumn;

    public KafkaSink(string bootstrapServers, string topic, string? keyColumn = null)
    {
        _topic = topic;
        _keyColumn = keyColumn;
        var config = new ProducerConfig { BootstrapServers = bootstrapServers };
        _producer = new ProducerBuilder<string, string>(config).Build();
    }

    /// <summary>
    /// Publish all rows of a DataFrame as JSON messages to the Kafka topic.
    /// Each row becomes one message.
    /// </summary>
    public void Publish(DataFrame df)
    {
        for (int r = 0; r < df.RowCount; r++)
        {
            var json = RowToJson(df, r);
            string? key = _keyColumn is not null && df.ColumnNames.Contains(_keyColumn)
                ? df[_keyColumn].GetObject(r)?.ToString()
                : null;

            _producer.Produce(_topic, new Message<string, string>
            {
                Key = key ?? "",
                Value = json
            });
        }
    }

    /// <summary>Publish asynchronously with delivery confirmation.</summary>
    public async Task PublishAsync(DataFrame df, CancellationToken cancellationToken = default)
    {
        for (int r = 0; r < df.RowCount; r++)
        {
            var json = RowToJson(df, r);
            string? key = _keyColumn is not null && df.ColumnNames.Contains(_keyColumn)
                ? df[_keyColumn].GetObject(r)?.ToString()
                : null;

            await _producer.ProduceAsync(_topic, new Message<string, string>
            {
                Key = key ?? "",
                Value = json
            }, cancellationToken);
        }
    }

    /// <summary>Flush pending messages.</summary>
    public void Flush(TimeSpan? timeout = null)
        => _producer.Flush(timeout ?? TimeSpan.FromSeconds(10));

    private static string RowToJson(DataFrame df, int row)
    {
        var dict = new Dictionary<string, object?>();
        foreach (var name in df.ColumnNames)
            dict[name] = df[name].GetObject(row);

        return JsonSerializer.Serialize(dict);
    }

    public void Dispose()
    {
        _producer.Flush(TimeSpan.FromSeconds(5));
        _producer.Dispose();
    }
}
