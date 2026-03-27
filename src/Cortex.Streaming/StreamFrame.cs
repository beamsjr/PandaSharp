using System.Threading.Channels;
using Cortex;
using Cortex.Column;

namespace Cortex.Streaming;

/// <summary>
/// Configuration for a streaming pipeline.
/// </summary>
public class StreamConfig
{
    /// <summary>Maximum events to buffer before forcing a window emit (default 10000).</summary>
    public int MaxBufferSize { get; set; } = 10_000;

    /// <summary>Channel capacity for backpressure (default 4096).</summary>
    public int ChannelCapacity { get; set; } = 4096;

    /// <summary>Allowed lateness for out-of-order events (default 0 = no late events).</summary>
    public TimeSpan AllowedLateness { get; set; } = TimeSpan.Zero;
}

/// <summary>
/// Real-time streaming DataFrame processor.
/// Accumulates events into micro-batches, applies windowed aggregations,
/// and emits result DataFrames via callbacks.
///
/// Usage:
///   var stream = StreamFrame.From(source)
///       .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
///       .Agg("value", AggType.Sum, "total")
///       .Agg("value", AggType.Count, "count")
///       .OnEmit(df => Console.WriteLine(df))
///       .Start();
/// </summary>
public class StreamFrame
{
    private readonly IStreamSource _source;
    private WindowSpec _window = new TumblingWindow(TimeSpan.FromMinutes(1));
    private readonly List<AggSpec> _aggregations = new();
    private readonly List<Action<DataFrame>> _emitCallbacks = new();
    private StreamConfig _config = new();

    private StreamFrame(IStreamSource source) => _source = source;

    /// <summary>Create a StreamFrame from a source.</summary>
    public static StreamFrame From(IStreamSource source) => new(source);

    /// <summary>Set the windowing strategy.</summary>
    public StreamFrame Window(WindowSpec window) { _window = window; return this; }

    /// <summary>Add an aggregation.</summary>
    public StreamFrame Agg(string sourceColumn, AggType aggType, string outputName)
    {
        _aggregations.Add(new AggSpec(sourceColumn, aggType, outputName));
        return this;
    }

    /// <summary>Register a callback for emitted window results.</summary>
    public StreamFrame OnEmit(Action<DataFrame> callback) { _emitCallbacks.Add(callback); return this; }

    /// <summary>Set stream configuration.</summary>
    public StreamFrame WithConfig(StreamConfig config) { _config = config; return this; }

    /// <summary>Set allowed lateness for watermarks.</summary>
    public StreamFrame WithWatermark(TimeSpan allowedLateness)
    {
        _config.AllowedLateness = allowedLateness;
        return this;
    }

    /// <summary>
    /// Start the streaming pipeline. Returns a task that completes when the source is exhausted.
    /// </summary>
    public async Task StartAsync(CancellationToken cancellationToken = default)
    {
        var channel = Channel.CreateBounded<StreamEvent>(
            new BoundedChannelOptions(_config.ChannelCapacity)
            {
                FullMode = BoundedChannelFullMode.Wait
            });

        // Start source producer
        var producerTask = Task.Run(async () =>
        {
            try { await _source.StartAsync(channel.Writer, cancellationToken); }
            catch (OperationCanceledException) { }
            finally { channel.Writer.TryComplete(); }
        }, cancellationToken);

        // Process events
        if (_window is SessionWindow sessionWindow)
            await ProcessSessionWindows(channel.Reader, sessionWindow, cancellationToken);
        else
            await ProcessFixedWindows(channel.Reader, cancellationToken);

        await producerTask;
    }

    /// <summary>Synchronous convenience wrapper.</summary>
    public void Start(CancellationToken cancellationToken = default)
        => StartAsync(cancellationToken).GetAwaiter().GetResult();

    /// <summary>
    /// Collect all emitted DataFrames into a list. Useful for testing.
    /// </summary>
    public async Task<List<DataFrame>> CollectAsync(CancellationToken cancellationToken = default)
    {
        var results = new List<DataFrame>();
        _emitCallbacks.Add(df => results.Add(df));
        await StartAsync(cancellationToken);
        return results;
    }

    public List<DataFrame> Collect(CancellationToken cancellationToken = default)
        => CollectAsync(cancellationToken).GetAwaiter().GetResult();

    private async Task ProcessFixedWindows(ChannelReader<StreamEvent> reader, CancellationToken ct)
    {
        // window_start → list of events in that window
        var windows = new Dictionary<DateTimeOffset, List<StreamEvent>>();
        var watermark = DateTimeOffset.MinValue;

        await foreach (var evt in reader.ReadAllAsync(ct))
        {
            // Update watermark
            if (evt.Timestamp > watermark)
                watermark = evt.Timestamp - _config.AllowedLateness;

            // Assign event to windows
            foreach (var windowStart in _window.AssignWindows(evt.Timestamp))
            {
                var windowEnd = _window.GetWindowEnd(windowStart);

                // Drop late events past the watermark
                if (windowEnd <= watermark && _config.AllowedLateness > TimeSpan.Zero)
                    continue;

                if (!windows.TryGetValue(windowStart, out var list))
                {
                    list = new List<StreamEvent>();
                    windows[windowStart] = list;
                }
                list.Add(evt);
            }

            // Emit closed windows (window end <= watermark)
            var closedWindows = windows.Keys
                .Where(ws => _window.GetWindowEnd(ws) <= watermark)
                .ToList();

            foreach (var ws in closedWindows)
            {
                EmitWindow(ws, _window.GetWindowEnd(ws), windows[ws]);
                windows.Remove(ws);
            }
        }

        // Emit remaining windows at stream end
        foreach (var (ws, events) in windows.OrderBy(kv => kv.Key))
            EmitWindow(ws, _window.GetWindowEnd(ws), events);
    }

    private async Task ProcessSessionWindows(ChannelReader<StreamEvent> reader, SessionWindow session, CancellationToken ct)
    {
        // Sessions keyed by start time, tracking last event time
        var sessions = new List<(DateTimeOffset Start, DateTimeOffset LastEvent, List<StreamEvent> Events)>();

        await foreach (var evt in reader.ReadAllAsync(ct))
        {
            bool merged = false;
            for (int i = 0; i < sessions.Count; i++)
            {
                var (start, last, events) = sessions[i];
                if (evt.Timestamp >= start && evt.Timestamp <= last + session.Gap)
                {
                    events.Add(evt);
                    var newLast = evt.Timestamp > last ? evt.Timestamp : last;
                    sessions[i] = (start, newLast, events);
                    merged = true;
                    break;
                }
            }

            if (!merged)
            {
                // Emit any sessions that are expired (last event + gap < current event)
                for (int i = sessions.Count - 1; i >= 0; i--)
                {
                    var (start, last, events) = sessions[i];
                    if (evt.Timestamp > last + session.Gap)
                    {
                        EmitWindow(start, last + session.Gap, events);
                        sessions.RemoveAt(i);
                    }
                }

                sessions.Add((evt.Timestamp, evt.Timestamp, new List<StreamEvent> { evt }));
            }
        }

        // Emit remaining sessions
        foreach (var (start, last, events) in sessions.OrderBy(s => s.Start))
            EmitWindow(start, last + session.Gap, events);
    }

    private void EmitWindow(DateTimeOffset windowStart, DateTimeOffset windowEnd, List<StreamEvent> events)
    {
        if (events.Count == 0) return;

        var df = BuildAggregatedDataFrame(windowStart, windowEnd, events);
        foreach (var callback in _emitCallbacks)
            callback(df);
    }

    private DataFrame BuildAggregatedDataFrame(DateTimeOffset windowStart, DateTimeOffset windowEnd, List<StreamEvent> events)
    {
        var columns = new List<IColumn>();

        // Window metadata columns
        columns.Add(new StringColumn("window_start", [windowStart.ToString("o")]));
        columns.Add(new StringColumn("window_end", [windowEnd.ToString("o")]));

        // Compute aggregations
        foreach (var agg in _aggregations)
        {
            var values = new List<double>();
            int nonNullCount = 0;
            foreach (var evt in events)
            {
                if (evt.Data.TryGetValue(agg.SourceColumn, out var val) && val is not null)
                {
                    nonNullCount++;
                    try { values.Add(Convert.ToDouble(val)); }
                    catch { }
                }
            }

            double result = agg.Type switch
            {
                AggType.Sum => values.Sum(),
                AggType.Mean => values.Count > 0 ? values.Average() : 0,
                AggType.Min => values.Count > 0 ? values.Min() : 0,
                AggType.Max => values.Count > 0 ? values.Max() : 0,
                // Count should count all events with the field present and non-null,
                // not just those that are convertible to double
                AggType.Count => nonNullCount,
                _ => 0
            };

            columns.Add(new Column<double>(agg.OutputName, [result]));
        }

        // If no aggregations, include event count
        if (_aggregations.Count == 0)
            columns.Add(new Column<int>("event_count", [events.Count]));

        return new DataFrame(columns);
    }
}

public enum AggType { Sum, Mean, Min, Max, Count }

internal record AggSpec(string SourceColumn, AggType Type, string OutputName);
