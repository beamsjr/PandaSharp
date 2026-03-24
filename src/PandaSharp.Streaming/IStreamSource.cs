using System.Threading.Channels;

namespace PandaSharp.Streaming;

/// <summary>
/// Interface for stream event sources. Implementations push events into the provided channel.
/// Built-in: EnumerableSource (testing), ChannelSource (in-process pub/sub).
/// External packages can implement KafkaSource, RedisSource, WebSocketSource, etc.
/// </summary>
public interface IStreamSource : IAsyncDisposable
{
    /// <summary>Start producing events into the channel.</summary>
    Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken);
}

/// <summary>
/// In-memory source that replays an enumerable of events. Useful for testing.
/// </summary>
public class EnumerableSource : IStreamSource
{
    private readonly IEnumerable<StreamEvent> _events;

    public EnumerableSource(IEnumerable<StreamEvent> events) => _events = events;

    public async Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken)
    {
        foreach (var evt in _events)
        {
            if (cancellationToken.IsCancellationRequested) break;
            await writer.WriteAsync(evt, cancellationToken);
        }
        writer.Complete();
    }

    public ValueTask DisposeAsync() => ValueTask.CompletedTask;
}

/// <summary>
/// Channel-based source for in-process pub/sub. Push events via the Writer property.
/// </summary>
public class ChannelSource : IStreamSource
{
    private readonly Channel<StreamEvent> _channel;
    public ChannelWriter<StreamEvent> Writer => _channel.Writer;

    public ChannelSource(int capacity = 1024)
    {
        _channel = Channel.CreateBounded<StreamEvent>(new BoundedChannelOptions(capacity)
        {
            FullMode = BoundedChannelFullMode.Wait
        });
    }

    public async Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken)
    {
        await foreach (var evt in _channel.Reader.ReadAllAsync(cancellationToken))
        {
            await writer.WriteAsync(evt, cancellationToken);
        }
        writer.Complete();
    }

    public ValueTask DisposeAsync()
    {
        _channel.Writer.TryComplete();
        return ValueTask.CompletedTask;
    }
}
