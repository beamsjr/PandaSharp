namespace Cortex.Streaming;

/// <summary>
/// A single event in the stream with a timestamp and key-value data.
/// </summary>
public class StreamEvent
{
    /// <summary>Event timestamp used for windowing.</summary>
    public DateTimeOffset Timestamp { get; }

    /// <summary>Event data as key-value pairs.</summary>
    public IReadOnlyDictionary<string, object?> Data { get; }

    public StreamEvent(DateTimeOffset timestamp, Dictionary<string, object?> data)
    {
        Timestamp = timestamp;
        Data = data;
    }

    public StreamEvent(DateTimeOffset timestamp, params (string Key, object? Value)[] fields)
    {
        Timestamp = timestamp;
        var dict = new Dictionary<string, object?>(fields.Length);
        foreach (var (k, v) in fields)
            dict[k] = v;
        Data = dict;
    }
}
