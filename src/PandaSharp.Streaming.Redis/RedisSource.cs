using System.Threading.Channels;
using StackExchange.Redis;

namespace PandaSharp.Streaming.Redis;

/// <summary>
/// Redis Streams consumer source for PandaSharp Streaming.
/// Reads from a Redis Stream using XREAD or XREADGROUP.
///
/// Usage:
///   var source = new RedisSource("localhost:6379", "my-stream");
///   StreamFrame.From(source)
///       .Window(new TumblingWindow(TimeSpan.FromSeconds(10)))
///       .Agg("value", AggType.Sum, "total")
///       .OnEmit(df => Console.WriteLine(df))
///       .Start();
/// </summary>
public class RedisSource : IStreamSource
{
    private readonly string _connectionString;
    private readonly string _streamKey;
    private readonly string? _consumerGroup;
    private readonly string _consumerName;
    private readonly int _batchSize;
    private ConnectionMultiplexer? _redis;

    public RedisSource(string connectionString, string streamKey,
        string? consumerGroup = null, string? consumerName = null, int batchSize = 100)
    {
        _connectionString = connectionString;
        _streamKey = streamKey;
        _consumerGroup = consumerGroup;
        _consumerName = consumerName ?? $"pandasharp-{Environment.ProcessId}";
        _batchSize = batchSize;
    }

    public async Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken)
    {
        _redis = await ConnectionMultiplexer.ConnectAsync(_connectionString);
        var db = _redis.GetDatabase();

        // Create consumer group if needed
        if (_consumerGroup is not null)
        {
            try
            {
                await db.StreamCreateConsumerGroupAsync(_streamKey, _consumerGroup, StreamPosition.NewMessages);
            }
            catch (RedisServerException ex) when (ex.Message.Contains("BUSYGROUP"))
            {
                // Group already exists
            }
        }

        var lastId = "0-0";
        try
        {
            while (!cancellationToken.IsCancellationRequested)
            {
                StreamEntry[] entries;

                if (_consumerGroup is not null)
                {
                    entries = await db.StreamReadGroupAsync(
                        _streamKey, _consumerGroup, _consumerName,
                        count: _batchSize);
                }
                else
                {
                    entries = await db.StreamReadAsync(_streamKey, lastId, count: _batchSize);
                }

                if (entries.Length == 0)
                {
                    await Task.Delay(50, cancellationToken);
                    continue;
                }

                foreach (var entry in entries)
                {
                    var data = new Dictionary<string, object?>();
                    DateTimeOffset timestamp = DateTimeOffset.UtcNow;

                    foreach (var field in entry.Values)
                    {
                        var key = field.Name.ToString();
                        var value = field.Value.ToString();

                        if (key == "timestamp" && DateTimeOffset.TryParse(value, out var ts))
                        {
                            timestamp = ts;
                            continue;
                        }

                        // Try to parse as number
                        if (double.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out double dbl))
                            data[key] = dbl;
                        else
                            data[key] = value;
                    }

                    data["_redis_id"] = entry.Id.ToString();
                    await writer.WriteAsync(new StreamEvent(timestamp, data), cancellationToken);
                    lastId = entry.Id!;

                    // Acknowledge in consumer group
                    if (_consumerGroup is not null)
                        await db.StreamAcknowledgeAsync(_streamKey, _consumerGroup, entry.Id);
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (RedisConnectionException) { }
        finally
        {
            writer.TryComplete();
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_redis is not null)
            await _redis.DisposeAsync();
    }
}
