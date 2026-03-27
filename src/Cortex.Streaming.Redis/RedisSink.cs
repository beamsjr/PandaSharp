using StackExchange.Redis;
using Cortex;

namespace Cortex.Streaming.Redis;

/// <summary>
/// Redis Streams producer sink — publishes DataFrame rows as stream entries via XADD.
///
/// Usage:
///   var sink = new RedisSink("localhost:6379", "output-stream");
///   StreamFrame.From(source)
///       .OnEmit(df => sink.Publish(df))
///       .Start();
/// </summary>
public class RedisSink : IDisposable
{
    private readonly ConnectionMultiplexer _redis;
    private readonly string _streamKey;
    private readonly int? _maxLen;

    public RedisSink(string connectionString, string streamKey, int? maxLen = null)
    {
        _redis = ConnectionMultiplexer.Connect(connectionString);
        _streamKey = streamKey;
        _maxLen = maxLen;
    }

    /// <summary>Publish all rows of a DataFrame as Redis Stream entries.</summary>
    public void Publish(DataFrame df)
    {
        var db = _redis.GetDatabase();
        for (int r = 0; r < df.RowCount; r++)
        {
            var fields = new List<NameValueEntry>();
            foreach (var name in df.ColumnNames)
            {
                var val = df[name].GetObject(r)?.ToString() ?? "";
                fields.Add(new NameValueEntry(name, val));
            }

            db.StreamAdd(_streamKey, fields.ToArray(), maxLength: _maxLen);
        }
    }

    /// <summary>Publish asynchronously.</summary>
    public async Task PublishAsync(DataFrame df)
    {
        var db = _redis.GetDatabase();
        for (int r = 0; r < df.RowCount; r++)
        {
            var fields = new List<NameValueEntry>();
            foreach (var name in df.ColumnNames)
            {
                var val = df[name].GetObject(r)?.ToString() ?? "";
                fields.Add(new NameValueEntry(name, val));
            }

            await db.StreamAddAsync(_streamKey, fields.ToArray(), maxLength: _maxLen);
        }
    }

    public void Dispose() => _redis.Dispose();
}
