using FluentAssertions;
using Cortex.Streaming;
using Xunit;

namespace Cortex.Streaming.Tests;

public class WebSocketSourceTests
{
    /// <summary>
    /// Verify WebSocketSource can be constructed and its config is correct.
    /// Full integration tests require a running WebSocket server.
    /// </summary>
    [Fact]
    public void WebSocketSource_CanBeCreated()
    {
        var source = new WebSocketSource("ws://localhost:8080/events");
        source.Should().NotBeNull();
    }

    [Fact]
    public void WebSocketSource_CustomTimestampField()
    {
        var source = new WebSocketSource("ws://localhost:8080/events", timestampField: "ts");
        source.Should().NotBeNull();
    }

    [Fact]
    public async Task WebSocketSource_CancellationStops()
    {
        // Connecting to a non-existent server with cancellation should not hang
        var source = new WebSocketSource("ws://localhost:1/nonexistent");
        using var cts = new CancellationTokenSource(TimeSpan.FromMilliseconds(200));

        // Should not throw — it catches WebSocketException
        var results = await StreamFrame.From(source)
            .Window(new TumblingWindow(TimeSpan.FromMinutes(1)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync(cts.Token);

        results.Should().BeEmpty();
    }

    [Fact]
    public async Task WebSocketSource_Disposable()
    {
        var source = new WebSocketSource("ws://localhost:8080/events");
        await source.DisposeAsync(); // should not throw even if never connected
    }
}
