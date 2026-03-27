using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Streaming;
using Xunit;

namespace Cortex.Streaming.Tests;

/// <summary>
/// Round 5 bug hunting: cross-module data flow issues in Streaming.
/// </summary>
public class CrossModuleRound5Tests
{
    private static DateTimeOffset T(int minutes) =>
        new DateTimeOffset(2024, 1, 1, 0, minutes, 0, TimeSpan.Zero);

    /// <summary>
    /// Bug: AggType.Count in streaming only counts values that are successfully
    /// converted to double via Convert.ToDouble(). For string data fields,
    /// Convert.ToDouble throws FormatException which is caught and swallowed,
    /// so the count is 0 even though events have the field.
    ///
    /// For example, if events have {"category": "A"} and you Agg("category", Count, "cnt"),
    /// cnt should be the number of events with "category" present and non-null,
    /// but instead it's 0 because "A" can't be converted to double.
    /// </summary>
    [Fact]
    public async Task Count_ShouldCountNonNullFieldPresence_NotNumericConvertibility()
    {
        var events = new[]
        {
            new StreamEvent(T(0), ("category", (object?)"A"), ("value", (object?)10.0)),
            new StreamEvent(T(1), ("category", (object?)"B"), ("value", (object?)20.0)),
            new StreamEvent(T(2), ("category", (object?)"A"), ("value", (object?)30.0)),
        };

        var source = new EnumerableSource(events);
        var results = await StreamFrame.From(source)
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("category", AggType.Count, "category_count")
            .CollectAsync();

        results.Should().HaveCount(1);
        var df = results[0];
        var countCol = df.GetColumn<double>("category_count");
        // All 3 events have "category" non-null, so count should be 3
        countCol[0].Should().Be(3.0,
            "Count should count events with non-null field, not just numeric-convertible ones");
    }

    /// <summary>
    /// Verify that Count with null values in some events is correct:
    /// should count only events where the field is present AND non-null.
    /// </summary>
    [Fact]
    public async Task Count_ShouldExcludeNullValues()
    {
        var events = new[]
        {
            new StreamEvent(T(0), ("value", (object?)10.0)),
            new StreamEvent(T(1), ("value", (object?)null)),   // null value
            new StreamEvent(T(2), ("value", (object?)30.0)),
        };

        var source = new EnumerableSource(events);
        var results = await StreamFrame.From(source)
            .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
            .Agg("value", AggType.Count, "value_count")
            .CollectAsync();

        results.Should().HaveCount(1);
        var df = results[0];
        var countCol = df.GetColumn<double>("value_count");
        // Only 2 events have non-null "value"
        countCol[0].Should().Be(2.0, "null values should not be counted");
    }
}
