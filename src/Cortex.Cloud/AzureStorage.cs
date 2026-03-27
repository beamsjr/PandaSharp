using Azure.Storage.Blobs;

namespace Cortex.Cloud;

/// <summary>
/// Azure Blob Storage adapter.
///
/// Usage:
///   var azure = new AzureStorage("DefaultEndpointsProtocol=https;AccountName=...");
///   var df = await azure.ReadAsync("container/path/data.parquet");
/// </summary>
public class AzureStorage : ICloudStorage
{
    private readonly BlobServiceClient _client;

    public AzureStorage(string connectionString)
    {
        _client = new BlobServiceClient(connectionString);
    }

    public async Task<Stream> OpenReadAsync(string path, CancellationToken cancellationToken = default)
    {
        var (container, blob) = ParseAzurePath(path);
        var blobClient = _client.GetBlobContainerClient(container).GetBlobClient(blob);
        var response = await blobClient.DownloadAsync(cancellationToken);
        return response.Value.Content;
    }

    public async Task WriteAsync(Stream data, string path, CancellationToken cancellationToken = default)
    {
        var (container, blob) = ParseAzurePath(path);
        var containerClient = _client.GetBlobContainerClient(container);
        await containerClient.CreateIfNotExistsAsync(cancellationToken: cancellationToken);
        var blobClient = containerClient.GetBlobClient(blob);
        await blobClient.UploadAsync(data, overwrite: true, cancellationToken);
    }

    private static (string Container, string Blob) ParseAzurePath(string path)
    {
        // Strip https://account.blob.core.windows.net/ prefix if present
        if (path.Contains("blob.core.windows.net"))
        {
            var uri = new Uri(path);
            var segments = uri.AbsolutePath.TrimStart('/').Split('/', 2);
            return (segments[0], segments.Length > 1 ? segments[1] : "");
        }

        var slash = path.IndexOf('/');
        if (slash < 0) throw new ArgumentException($"Invalid Azure path: {path}");
        return (path[..slash], path[(slash + 1)..]);
    }
}
