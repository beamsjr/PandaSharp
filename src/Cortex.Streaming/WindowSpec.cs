namespace Cortex.Streaming;

/// <summary>
/// Specifies the windowing strategy for stream aggregation.
/// </summary>
public abstract class WindowSpec
{
    /// <summary>Assign an event to one or more window keys (start timestamps).</summary>
    public abstract IEnumerable<DateTimeOffset> AssignWindows(DateTimeOffset eventTime);

    /// <summary>Get the window end time given a window start.</summary>
    public abstract DateTimeOffset GetWindowEnd(DateTimeOffset windowStart);
}

/// <summary>
/// Fixed-size, non-overlapping time windows.
/// Example: TumblingWindow(TimeSpan.FromMinutes(5)) groups events into 5-minute buckets.
/// </summary>
public class TumblingWindow : WindowSpec
{
    public TimeSpan Size { get; }

    public TumblingWindow(TimeSpan size) => Size = size;

    public override IEnumerable<DateTimeOffset> AssignWindows(DateTimeOffset eventTime)
    {
        long ticks = eventTime.UtcTicks;
        long windowTicks = Size.Ticks;
        long windowStart = ticks - (ticks % windowTicks);
        yield return new DateTimeOffset(windowStart, TimeSpan.Zero);
    }

    public override DateTimeOffset GetWindowEnd(DateTimeOffset windowStart) => windowStart + Size;
}

/// <summary>
/// Fixed-size, overlapping windows that advance by a slide interval.
/// Example: SlidingWindow(TimeSpan.FromMinutes(10), TimeSpan.FromMinutes(2))
/// creates 10-minute windows that start every 2 minutes.
/// </summary>
public class SlidingWindow : WindowSpec
{
    public TimeSpan Size { get; }
    public TimeSpan Slide { get; }

    public SlidingWindow(TimeSpan size, TimeSpan slide)
    {
        Size = size;
        Slide = slide;
    }

    public override IEnumerable<DateTimeOffset> AssignWindows(DateTimeOffset eventTime)
    {
        long ticks = eventTime.UtcTicks;
        long slideTicks = Slide.Ticks;
        long sizeTicks = Size.Ticks;

        // Find the earliest window that contains this event
        long latestStart = ticks - (ticks % slideTicks);
        long earliestStart = latestStart - sizeTicks + slideTicks;

        for (long start = earliestStart; start <= latestStart; start += slideTicks)
        {
            if (start + sizeTicks > ticks)
                yield return new DateTimeOffset(start, TimeSpan.Zero);
        }
    }

    public override DateTimeOffset GetWindowEnd(DateTimeOffset windowStart) => windowStart + Size;
}

/// <summary>
/// Dynamic windows based on activity gaps. A new window starts when the gap
/// between consecutive events exceeds the gap duration.
/// </summary>
public class SessionWindow : WindowSpec
{
    public TimeSpan Gap { get; }

    public SessionWindow(TimeSpan gap) => Gap = gap;

    public override IEnumerable<DateTimeOffset> AssignWindows(DateTimeOffset eventTime)
    {
        // Session windows are assigned during processing, not statically.
        // The StreamProcessor handles session merging.
        yield return eventTime;
    }

    public override DateTimeOffset GetWindowEnd(DateTimeOffset windowStart) => windowStart + Gap;
}
