using PandaSharp;
using PandaSharp.IO;

namespace PandaSharp.Cloud;

/// <summary>
/// Unified cloud storage interface for reading/writing DataFrames.
/// Supports S3, Azure Blob Storage, and Google Cloud Storage.
///
/// Usage:
///   // S3
///   var df = await S3Storage.ReadAsync("s3://bucket/path/data.parquet");
///   await S3Storage.WriteAsync(df, "s3://bucket/output/result.csv");
///
///   // Azure
///   var df = await AzureStorage.ReadAsync("https://account.blob.core.windows.net/container/data.parquet");
///
///   // GCS
///   var df = await GcsStorage.ReadAsync("gs://bucket/data.csv");
/// </summary>
public interface ICloudStorage
{
    Task<Stream> OpenReadAsync(string path, CancellationToken cancellationToken = default);
    Task WriteAsync(Stream data, string path, CancellationToken cancellationToken = default);
}

/// <summary>Extension methods for reading/writing DataFrames via cloud storage.</summary>
public static class CloudDataFrameExtensions
{
    /// <summary>Read a DataFrame from cloud storage. Format auto-detected from path extension.</summary>
    public static async Task<DataFrame> ReadAsync(this ICloudStorage storage, string path,
        CancellationToken cancellationToken = default)
    {
        using var stream = await storage.OpenReadAsync(path, cancellationToken);

        // Buffer to MemoryStream for seekable access
        var memStream = new MemoryStream();
        await stream.CopyToAsync(memStream, cancellationToken);
        memStream.Position = 0;

        var ext = GetExtension(path);
        return ext switch
        {
            ".csv" => CsvReader.Read(memStream),
            ".parquet" => ParquetIO.ReadParquet(memStream),
            ".arrow" or ".ipc" => ArrowIpcReader.Read(memStream),
            ".json" => JsonReader.ReadString(new StreamReader(memStream).ReadToEnd()),
            _ => throw new NotSupportedException($"Unsupported format: {ext}")
        };
    }

    /// <summary>Write a DataFrame to cloud storage. Format auto-detected from path extension.</summary>
    public static async Task WriteAsync(this ICloudStorage storage, DataFrame df, string path,
        CancellationToken cancellationToken = default)
    {
        using var memStream = new MemoryStream();
        var ext = GetExtension(path);

        switch (ext)
        {
            case ".csv": CsvWriter.Write(df, memStream); break;
            case ".parquet": await ParquetIO.WriteParquetAsync(df, memStream); break;
            case ".arrow" or ".ipc": ArrowIpcWriter.Write(df, memStream); break;
            case ".json":
                var json = DataFrameIO.ToJsonString(df);
                var writer = new StreamWriter(memStream);
                await writer.WriteAsync(json);
                await writer.FlushAsync(cancellationToken);
                break;
            default:
                throw new NotSupportedException($"Unsupported format: {ext}");
        }

        memStream.Position = 0;
        await storage.WriteAsync(memStream, path, cancellationToken);
    }

    private static string GetExtension(string path)
    {
        // Strip query strings (S3 signed URLs, etc.)
        var cleanPath = path.Split('?')[0];
        return Path.GetExtension(cleanPath).ToLowerInvariant();
    }

    /// <summary>Write a DataFrame directly to a Parquet file in cloud storage.</summary>
    public static Task WriteParquetAsync(this ICloudStorage storage, DataFrame df, string path,
        CancellationToken cancellationToken = default)
        => storage.WriteAsync(df, path.EndsWith(".parquet") ? path : path + ".parquet", cancellationToken);
}
