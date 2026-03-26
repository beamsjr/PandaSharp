using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Streaming;
using Xunit;

namespace PandaSharp.Streaming.Tests;

public class StreamingEdgeCaseRound3Tests
{
    private static DateTimeOffset T(int minutes) =>
        new DateTimeOffset(2024, 1, 1, 0, 0, 0, TimeSpan.Zero).AddMinutes(minutes);

    private static StreamEvent Evt(int minutes, double value) =>
        new StreamEvent(T(minutes), new Dictionary<string, object?> { ["value"] = value });

    // ═══ Sliding window: event exactly at window boundary ═══

    [Fact]
    public async Task SlidingWindow_EventAtBoundary_AssignedCorrectly()
    {
        // 10-minute windows sliding every 5 minutes
        // Event at t=10 should be in windows [5,15) and [10,20)
        var events = new[] { Evt(10, 100) };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new SlidingWindow(TimeSpan.FromMinutes(10), TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        // Event at t=10 should appear in exactly 2 windows
        results.Should().HaveCount(2);
        results.Sum(r => r.GetColumn<double>("total")[0]!.Value).Should().Be(200);
    }

    // ═══ Session window: single event creates a session ═══

    [Fact]
    public async Task SessionWindow_SingleEvent_CreatesOneSession()
    {
        var events = new[] { Evt(5, 42) };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new SessionWindow(TimeSpan.FromMinutes(3)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].GetColumn<double>("total")[0].Should().Be(42);
    }

    // ═══ Events with missing data column ═══

    [Fact]
    public async Task Agg_MissingColumn_TreatedAsEmpty()
    {
        // Events don't have "price" field but aggregation asks for it
        var events = new[]
        {
            new StreamEvent(T(0), new Dictionary<string, object?> { ["volume"] = 100.0 }),
            new StreamEvent(T(1), new Dictionary<string, object?> { ["volume"] = 200.0 }),
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("price", AggType.Sum, "total_price")
            .Agg("price", AggType.Count, "count")
            .CollectAsync();

        results.Should().HaveCount(1);
        // No values found for "price", so sum should be 0 and count should be 0
        results[0].GetColumn<double>("total_price")[0].Should().Be(0);
        results[0].GetColumn<double>("count")[0].Should().Be(0);
    }

    // ═══ Events with null values ═══

    [Fact]
    public async Task Agg_NullValues_Skipped()
    {
        var events = new[]
        {
            new StreamEvent(T(0), new Dictionary<string, object?> { ["value"] = 10.0 }),
            new StreamEvent(T(1), new Dictionary<string, object?> { ["value"] = null }),
            new StreamEvent(T(2), new Dictionary<string, object?> { ["value"] = 30.0 }),
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .Agg("value", AggType.Count, "count")
            .CollectAsync();

        results.Should().HaveCount(1);
        // Null should be skipped
        results[0].GetColumn<double>("total")[0].Should().Be(40); // 10 + 30
        results[0].GetColumn<double>("count")[0].Should().Be(2);  // only 2 non-null
    }

    // ═══ Multiple aggregations on same column ═══

    [Fact]
    public async Task MultipleAggs_SameColumn()
    {
        var events = new[] { Evt(0, 10), Evt(1, 20), Evt(2, 30) };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "sum")
            .Agg("value", AggType.Mean, "mean")
            .Agg("value", AggType.Min, "min")
            .Agg("value", AggType.Max, "max")
            .Agg("value", AggType.Count, "count")
            .CollectAsync();

        results.Should().HaveCount(1);
        var r = results[0];
        r.GetColumn<double>("sum")[0].Should().Be(60);
        r.GetColumn<double>("mean")[0].Should().Be(20);
        r.GetColumn<double>("min")[0].Should().Be(10);
        r.GetColumn<double>("max")[0].Should().Be(30);
        r.GetColumn<double>("count")[0].Should().Be(3);
    }

    // ═══ Cancellation ═══

    [Fact]
    public async Task StartAsync_AlreadyCancelledToken_CompletesQuickly()
    {
        var cts = new CancellationTokenSource();
        cts.Cancel();

        var events = Enumerable.Range(0, 1000).Select(i => Evt(i, i)).ToArray();

        // With an already cancelled token, it should complete quickly (may emit nothing or partial)
        var act = async () =>
        {
            await StreamFrame.From(new EnumerableSource(events))
                .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
                .Agg("value", AggType.Sum, "total")
                .StartAsync(cts.Token);
        };

        // Should either complete or throw OperationCanceledException — not hang forever
        await act.Should().CompleteWithinAsync(TimeSpan.FromSeconds(5));
    }
}
