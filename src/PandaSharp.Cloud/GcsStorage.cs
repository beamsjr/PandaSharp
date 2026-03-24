using Google.Cloud.Storage.V1;

namespace PandaSharp.Cloud;

/// <summary>
/// Google Cloud Storage adapter.
///
/// Usage:
///   var gcs = new GcsStorage(); // uses default credentials
///   var df = await gcs.ReadAsync("gs://bucket/path/data.parquet");
/// </summary>
public class GcsStorage : ICloudStorage
{
    private readonly StorageClient _client;

    public GcsStorage(StorageClient? client = null)
    {
        _client = client ?? StorageClient.Create();
    }

    public async Task<Stream> OpenReadAsync(string path, CancellationToken cancellationToken = default)
    {
        var (bucket, obj) = ParseGcsPath(path);
        var stream = new MemoryStream();
        await _client.DownloadObjectAsync(bucket, obj, stream, cancellationToken: cancellationToken);
        stream.Position = 0;
        return stream;
    }

    public async Task WriteAsync(Stream data, string path, CancellationToken cancellationToken = default)
    {
        var (bucket, obj) = ParseGcsPath(path);
        await _client.UploadObjectAsync(bucket, obj, null, data, cancellationToken: cancellationToken);
    }

    private static (string Bucket, string Object) ParseGcsPath(string path)
    {
        if (path.StartsWith("gs://", StringComparison.OrdinalIgnoreCase))
            path = path[5..];
        var slash = path.IndexOf('/');
        if (slash < 0) throw new ArgumentException($"Invalid GCS path: {path}");
        return (path[..slash], path[(slash + 1)..]);
    }
}
