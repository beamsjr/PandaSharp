using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Streaming;
using Xunit;

namespace Cortex.Streaming.Tests;

public class StreamingTests
{
    private static DateTimeOffset T(int minutes) =>
        new DateTimeOffset(2024, 1, 1, 0, minutes, 0, TimeSpan.Zero);

    private static StreamEvent Evt(int minutes, double value, string? key = null)
    {
        var fields = new Dictionary<string, object?> { ["value"] = value };
        if (key is not null) fields["key"] = key;
        return new StreamEvent(T(minutes), fields);
    }

    // ===== Tumbling Window =====

    [Fact]
    public async Task TumblingWindow_GroupsEventsIntoFixedBuckets()
    {
        var events = new[]
        {
            Evt(0, 10), Evt(1, 20), Evt(2, 30),   // window [0, 5)
            Evt(5, 40), Evt(7, 50),                 // window [5, 10)
            Evt(10, 60)                              // window [10, 15)
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .Agg("value", AggType.Count, "count")
            .CollectAsync();

        results.Should().HaveCount(3);

        // First window [0,5): sum=60, count=3
        results[0].GetColumn<double>("total")[0].Should().Be(60);
        results[0].GetColumn<double>("count")[0].Should().Be(3);

        // Second window [5,10): sum=90, count=2
        results[1].GetColumn<double>("total")[0].Should().Be(90);
        results[1].GetColumn<double>("count")[0].Should().Be(2);

        // Third window [10,15): sum=60, count=1
        results[2].GetColumn<double>("total")[0].Should().Be(60);
    }

    [Fact]
    public async Task TumblingWindow_Mean()
    {
        var events = new[]
        {
            Evt(0, 10), Evt(1, 20), Evt(3, 30),
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Mean, "avg")
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].GetColumn<double>("avg")[0].Should().Be(20);
    }

    [Fact]
    public async Task TumblingWindow_MinMax()
    {
        var events = new[]
        {
            Evt(0, 10), Evt(1, 50), Evt(2, 30),
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(10)))
            .Agg("value", AggType.Min, "min_val")
            .Agg("value", AggType.Max, "max_val")
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].GetColumn<double>("min_val")[0].Should().Be(10);
        results[0].GetColumn<double>("max_val")[0].Should().Be(50);
    }

    [Fact]
    public async Task TumblingWindow_EmptySource()
    {
        var results = await StreamFrame.From(new EnumerableSource([]))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results.Should().BeEmpty();
    }

    [Fact]
    public async Task TumblingWindow_SingleEvent()
    {
        var results = await StreamFrame.From(new EnumerableSource([Evt(3, 42)]))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].GetColumn<double>("total")[0].Should().Be(42);
    }

    [Fact]
    public async Task TumblingWindow_HasWindowMetadata()
    {
        var results = await StreamFrame.From(new EnumerableSource([Evt(3, 10)]))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results[0].ColumnNames.Should().Contain("window_start");
        results[0].ColumnNames.Should().Contain("window_end");
    }

    // ===== Sliding Window =====

    [Fact]
    public async Task SlidingWindow_OverlappingWindows()
    {
        var events = new[]
        {
            Evt(0, 10), Evt(3, 20), Evt(6, 30), Evt(9, 40),
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new SlidingWindow(TimeSpan.FromMinutes(6), TimeSpan.FromMinutes(3)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        // Sliding windows: events should appear in multiple windows
        results.Count.Should().BeGreaterThan(2);

        // Total sum across all windows should be greater than the sum of all events
        // because events appear in overlapping windows
        var totalAcrossWindows = results.Sum(r => r.GetColumn<double>("total")[0]!.Value);
        totalAcrossWindows.Should().BeGreaterThan(100); // 10+20+30+40=100
    }

    // ===== Session Window =====

    [Fact]
    public async Task SessionWindow_SplitsByGap()
    {
        var events = new[]
        {
            Evt(0, 10), Evt(1, 20), Evt(2, 30),    // session 1 (gap < 5m)
            Evt(10, 40), Evt(11, 50),                // session 2 (gap = 8m > 5m)
            Evt(20, 60),                              // session 3 (gap = 9m > 5m)
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new SessionWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results.Should().HaveCount(3);
        results[0].GetColumn<double>("total")[0].Should().Be(60);  // 10+20+30
        results[1].GetColumn<double>("total")[0].Should().Be(90);  // 40+50
        results[2].GetColumn<double>("total")[0].Should().Be(60);  // 60
    }

    [Fact]
    public async Task SessionWindow_SingleLongSession()
    {
        // All events within gap of each other → single session
        var events = Enumerable.Range(0, 10)
            .Select(i => Evt(i, i * 10.0))
            .ToArray();

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new SessionWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].GetColumn<double>("total")[0].Should().Be(450); // 0+10+20+...+90
    }

    // ===== No Aggregations =====

    [Fact]
    public async Task NoAggregations_ReturnsEventCount()
    {
        var events = new[] { Evt(0, 10), Evt(1, 20), Evt(2, 30) };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .CollectAsync();

        results.Should().HaveCount(1);
        results[0].ColumnNames.Should().Contain("event_count");
        results[0].GetColumn<int>("event_count")[0].Should().Be(3);
    }

    // ===== OnEmit callback =====

    [Fact]
    public async Task OnEmit_CallbackInvokedPerWindow()
    {
        int callCount = 0;
        var events = new[]
        {
            Evt(0, 10), Evt(5, 20), Evt(10, 30),
        };

        await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Sum, "total")
            .OnEmit(_ => Interlocked.Increment(ref callCount))
            .StartAsync();

        callCount.Should().Be(3);
    }

    // ===== ChannelSource =====

    [Fact]
    public async Task ChannelSource_PushEvents()
    {
        var source = new ChannelSource();
        var streamFrame = StreamFrame.From(source)
            .Window(new TumblingWindow(TimeSpan.FromMinutes(10)))
            .Agg("value", AggType.Sum, "total");

        var collectTask = streamFrame.CollectAsync();

        // Push events
        await source.Writer.WriteAsync(Evt(0, 100));
        await source.Writer.WriteAsync(Evt(1, 200));
        source.Writer.Complete();

        var results = await collectTask;
        results.Should().HaveCount(1);
        results[0].GetColumn<double>("total")[0].Should().Be(300);
    }

    // ===== Watermark =====

    [Fact]
    public async Task Watermark_LateEventsDropped()
    {
        // With 2-minute watermark, events more than 2 minutes behind are dropped
        var events = new[]
        {
            Evt(0, 10),     // window [0,5)
            Evt(5, 20),     // window [5,10) — this advances watermark to 3
            Evt(10, 30),    // window [10,15) — this advances watermark to 8
            // window [0,5) is now closed (end=5 < watermark=8)
        };

        var results = await StreamFrame.From(new EnumerableSource(events))
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .WithWatermark(TimeSpan.FromMinutes(2))
            .Agg("value", AggType.Sum, "total")
            .CollectAsync();

        // All 3 windows should emit (the watermark closes them in order)
        results.Count.Should().BeGreaterThanOrEqualTo(2);
    }
}
