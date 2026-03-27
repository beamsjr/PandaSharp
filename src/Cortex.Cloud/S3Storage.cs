using Amazon.S3;
using Amazon.S3.Model;

namespace Cortex.Cloud;

/// <summary>
/// Amazon S3 storage adapter.
/// Reads/writes DataFrames from S3 buckets.
///
/// Usage:
///   var s3 = new S3Storage(); // uses default AWS credentials
///   var df = await s3.ReadAsync("my-bucket", "data/file.parquet");
///   await s3.WriteAsync(df, "my-bucket", "output/result.csv");
/// </summary>
public class S3Storage : ICloudStorage
{
    private readonly IAmazonS3 _client;

    public S3Storage(IAmazonS3? client = null)
    {
        _client = client ?? new AmazonS3Client();
    }

    public S3Storage(string accessKey, string secretKey, Amazon.RegionEndpoint region)
    {
        _client = new AmazonS3Client(accessKey, secretKey, region);
    }

    public async Task<Stream> OpenReadAsync(string path, CancellationToken cancellationToken = default)
    {
        var (bucket, key) = ParseS3Path(path);
        var response = await _client.GetObjectAsync(bucket, key, cancellationToken);
        return response.ResponseStream;
    }

    public async Task WriteAsync(Stream data, string path, CancellationToken cancellationToken = default)
    {
        var (bucket, key) = ParseS3Path(path);
        var request = new PutObjectRequest
        {
            BucketName = bucket,
            Key = key,
            InputStream = data
        };
        await _client.PutObjectAsync(request, cancellationToken);
    }

    /// <summary>Read a DataFrame from S3.</summary>
    public Task<Cortex.DataFrame> ReadAsync(string bucket, string key,
        CancellationToken cancellationToken = default)
        => ((ICloudStorage)this).ReadAsync($"s3://{bucket}/{key}", cancellationToken);

    /// <summary>Write a DataFrame to S3.</summary>
    public Task WriteAsync(Cortex.DataFrame df, string bucket, string key,
        CancellationToken cancellationToken = default)
        => ((ICloudStorage)this).WriteAsync(df, $"s3://{bucket}/{key}", cancellationToken);

    private static (string Bucket, string Key) ParseS3Path(string path)
    {
        if (path.StartsWith("s3://", StringComparison.OrdinalIgnoreCase))
            path = path[5..];
        var slash = path.IndexOf('/');
        if (slash < 0) throw new ArgumentException($"Invalid S3 path: {path}");
        return (path[..slash], path[(slash + 1)..]);
    }
}
