using Apache.Arrow;
using Apache.Arrow.Flight;
using Apache.Arrow.Flight.Client;
using Apache.Arrow.Ipc;
using Grpc.Net.Client;
using Cortex;
using Cortex.IO;

namespace Cortex.Flight;

/// <summary>
/// Arrow Flight client for distributed DataFrame transport.
/// Enables zero-copy transfer of DataFrames between processes via gRPC.
///
/// Usage:
///   var client = new FlightDataClient("grpc://localhost:50051");
///   var df = await client.GetDataFrameAsync("my-dataset");
///   await client.PutDataFrameAsync("results", resultDf);
/// </summary>
public class FlightDataClient : IAsyncDisposable
{
    private readonly FlightClient _client;
    private readonly GrpcChannel _channel;

    public FlightDataClient(string endpoint)
    {
        _channel = GrpcChannel.ForAddress(endpoint);
        _client = new FlightClient(_channel);
    }

    /// <summary>
    /// Get a DataFrame from a Flight endpoint by descriptor.
    /// </summary>
    public async Task<DataFrame> GetDataFrameAsync(string descriptor, CancellationToken cancellationToken = default)
    {
        var flightDescriptor = FlightDescriptor.CreatePathDescriptor(descriptor);
        var info = await _client.GetInfo(flightDescriptor).ResponseAsync;

        var batches = new List<RecordBatch>();
        foreach (var endpoint in info.Endpoints)
        {
            var stream = _client.GetStream(endpoint.Ticket);
            while (await stream.ResponseStream.MoveNext(cancellationToken))
            {
                batches.Add(stream.ResponseStream.Current);
            }
        }

        if (batches.Count == 0)
            return new DataFrame();

        // Convert first batch to DataFrame, concatenate rest
        var result = ArrowIpcReader.FromRecordBatch(batches[0]);
        for (int i = 1; i < batches.Count; i++)
        {
            var next = ArrowIpcReader.FromRecordBatch(batches[i]);
            result = Cortex.Concat.ConcatExtensions.Concat(result, next);
        }

        return result;
    }

    /// <summary>
    /// Put a DataFrame to a Flight endpoint.
    /// </summary>
    public async Task PutDataFrameAsync(string descriptor, DataFrame df, CancellationToken cancellationToken = default)
    {
        var flightDescriptor = FlightDescriptor.CreatePathDescriptor(descriptor);

        // Build Arrow schema and record batch
        var schema = BuildSchema(df);
        var batch = BuildBatch(df, schema);

        var stream = _client.StartPut(flightDescriptor);
        await stream.RequestStream.WriteAsync(batch);
        await stream.RequestStream.CompleteAsync();
    }

    /// <summary>List available datasets on the server.</summary>
    public async Task<List<string>> ListFlightsAsync(CancellationToken cancellationToken = default)
    {
        var results = new List<string>();
        var listing = _client.ListFlights();
        while (await listing.ResponseStream.MoveNext(cancellationToken))
        {
            var info = listing.ResponseStream.Current;
            results.Add(info.Descriptor.Paths.FirstOrDefault() ?? "unknown");
        }
        return results;
    }

    private static Apache.Arrow.Schema BuildSchema(DataFrame df)
    {
        var fields = new List<Field>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var arrowType = col.DataType switch
            {
                Type t when t == typeof(int) => (Apache.Arrow.Types.IArrowType)Apache.Arrow.Types.Int32Type.Default,
                Type t when t == typeof(long) => Apache.Arrow.Types.Int64Type.Default,
                Type t when t == typeof(double) => Apache.Arrow.Types.DoubleType.Default,
                Type t when t == typeof(float) => Apache.Arrow.Types.FloatType.Default,
                Type t when t == typeof(bool) => Apache.Arrow.Types.BooleanType.Default,
                _ => Apache.Arrow.Types.StringType.Default
            };
            fields.Add(new Field(name, arrowType, col.NullCount > 0));
        }
        return new Apache.Arrow.Schema(fields, null);
    }

    private static RecordBatch BuildBatch(DataFrame df, Apache.Arrow.Schema schema)
    {
        // Reuse ArrowIpcWriter's logic via stream round-trip
        using var ms = new MemoryStream();
        ArrowIpcWriter.Write(df, ms);
        ms.Position = 0;
        using var reader = new ArrowFileReader(ms);
        return reader.ReadNextRecordBatch()!;
    }

    public async ValueTask DisposeAsync()
    {
        _channel.Dispose();
    }
}
