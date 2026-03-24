using System.Net.WebSockets;
using System.Text;
using System.Text.Json;
using System.Threading.Channels;

namespace PandaSharp.Streaming;

/// <summary>
/// Stream source that connects to a WebSocket endpoint and parses incoming
/// JSON messages as StreamEvents. Uses System.Net.WebSockets — no external dependencies.
///
/// Usage:
///   var source = new WebSocketSource("ws://localhost:8080/events");
///   StreamFrame.From(source)
///       .Window(new TumblingWindow(TimeSpan.FromSeconds(10)))
///       .Agg("price", AggType.Mean, "avg_price")
///       .OnEmit(df => Console.WriteLine(df))
///       .Start();
///
/// JSON messages should be objects with a "timestamp" field (ISO 8601) and data fields:
///   {"timestamp": "2024-01-01T00:00:00Z", "price": 42.5, "symbol": "AAPL"}
/// If no timestamp field is present, DateTimeOffset.UtcNow is used.
/// </summary>
public class WebSocketSource : IStreamSource
{
    private readonly Uri _uri;
    private readonly string _timestampField;
    private readonly int _bufferSize;
    private ClientWebSocket? _ws;

    /// <summary>
    /// Create a WebSocket source.
    /// </summary>
    /// <param name="uri">WebSocket URI (ws:// or wss://).</param>
    /// <param name="timestampField">JSON field name for event timestamp (default "timestamp").</param>
    /// <param name="bufferSize">Receive buffer size in bytes (default 4096).</param>
    public WebSocketSource(string uri, string timestampField = "timestamp", int bufferSize = 4096)
    {
        _uri = new Uri(uri);
        _timestampField = timestampField;
        _bufferSize = bufferSize;
    }

    public async Task StartAsync(ChannelWriter<StreamEvent> writer, CancellationToken cancellationToken)
    {
        _ws = new ClientWebSocket();

        try
        {
            await _ws.ConnectAsync(_uri, cancellationToken);

            var buffer = new byte[_bufferSize];
            var messageBuffer = new StringBuilder();

            while (_ws.State == WebSocketState.Open && !cancellationToken.IsCancellationRequested)
            {
                var result = await _ws.ReceiveAsync(buffer, cancellationToken);

                if (result.MessageType == WebSocketMessageType.Close)
                    break;

                if (result.MessageType == WebSocketMessageType.Text)
                {
                    messageBuffer.Append(Encoding.UTF8.GetString(buffer, 0, result.Count));

                    if (result.EndOfMessage)
                    {
                        var json = messageBuffer.ToString();
                        messageBuffer.Clear();

                        var evt = ParseEvent(json);
                        if (evt is not null)
                            await writer.WriteAsync(evt, cancellationToken);
                    }
                }
            }
        }
        catch (OperationCanceledException) { }
        catch (WebSocketException) { }
        finally
        {
            writer.TryComplete();
        }
    }

    private StreamEvent? ParseEvent(string json)
    {
        try
        {
            using var doc = JsonDocument.Parse(json);
            var root = doc.RootElement;

            if (root.ValueKind != JsonValueKind.Object)
                return null;

            // Extract timestamp
            DateTimeOffset timestamp;
            if (root.TryGetProperty(_timestampField, out var tsProp))
            {
                timestamp = tsProp.ValueKind == JsonValueKind.String
                    ? DateTimeOffset.Parse(tsProp.GetString()!)
                    : DateTimeOffset.FromUnixTimeMilliseconds(tsProp.GetInt64());
            }
            else
            {
                timestamp = DateTimeOffset.UtcNow;
            }

            // Extract all other fields as data
            var data = new Dictionary<string, object?>();
            foreach (var prop in root.EnumerateObject())
            {
                if (prop.Name == _timestampField) continue;
                data[prop.Name] = prop.Value.ValueKind switch
                {
                    JsonValueKind.Number => prop.Value.TryGetInt64(out long l) && prop.Value.GetDouble() == l
                        ? (object)l : prop.Value.GetDouble(),
                    JsonValueKind.String => prop.Value.GetString(),
                    JsonValueKind.True => true,
                    JsonValueKind.False => false,
                    JsonValueKind.Null => null,
                    _ => prop.Value.ToString()
                };
            }

            return new StreamEvent(timestamp, data);
        }
        catch
        {
            return null; // skip malformed messages
        }
    }

    public async ValueTask DisposeAsync()
    {
        if (_ws is { State: WebSocketState.Open })
        {
            try
            {
                await _ws.CloseAsync(WebSocketCloseStatus.NormalClosure, "Closing", CancellationToken.None);
            }
            catch { }
        }
        _ws?.Dispose();
    }
}
