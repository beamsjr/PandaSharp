using System.Text.Json;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.IO;

namespace PandaSharp.Geo;

/// <summary>
/// GeoParquet reader/writer: standard Parquet files with WKB geometry columns.
/// Follows the GeoParquet specification for metadata.
///
/// Usage:
///   GeoParquet.Write(df, geoColumn, "output.parquet");
///   var (df, geo) = GeoParquet.Read("output.parquet");
/// </summary>
public static class GeoParquet
{
    private const string GeometryColumnName = "geometry";

    /// <summary>
    /// Write a DataFrame with a GeoColumn to a GeoParquet file.
    /// The geometry is stored as a WKB binary column.
    /// </summary>
    public static void Write(DataFrame df, GeoColumn geoColumn, string path, string geometryColumn = GeometryColumnName)
    {
        // Build a new DataFrame with WKB geometry added
        var wkbValues = new byte[geoColumn.Length][];
        for (int i = 0; i < geoColumn.Length; i++)
            wkbValues[i] = Wkb.EncodePoint(geoColumn[i]);

        // Store WKB as hex strings in Parquet (portable across all Parquet implementations)
        var hexValues = new string?[geoColumn.Length];
        for (int i = 0; i < geoColumn.Length; i++)
            hexValues[i] = Convert.ToHexString(wkbValues[i]);

        var geoDf = df.AddColumn(new StringColumn(geometryColumn, hexValues));

        // Write the Parquet file
        ParquetIO.WriteParquet(geoDf, path);

        // Write the geo metadata sidecar file
        WriteGeoMetadata(path, geometryColumn, geoColumn.Bounds());
    }

    /// <summary>
    /// Read a GeoParquet file, returning a DataFrame and a GeoColumn.
    /// </summary>
    public static (DataFrame DataFrame, GeoColumn Geometry) Read(string path, string? geometryColumn = null)
    {
        var df = ParquetIO.ReadParquet(path);

        // Try to find the geometry column from metadata or by name
        geometryColumn ??= DetectGeometryColumn(path, df);

        if (!df.ColumnNames.Contains(geometryColumn))
            throw new InvalidDataException($"Geometry column '{geometryColumn}' not found in Parquet file.");

        // Decode WKB hex strings back to GeoColumn
        var hexCol = df.GetStringColumn(geometryColumn);
        var lats = new double[df.RowCount];
        var lons = new double[df.RowCount];

        for (int i = 0; i < df.RowCount; i++)
        {
            var hex = hexCol[i];
            if (hex is null) continue;
            var bytes = Convert.FromHexString(hex);
            var point = Wkb.DecodePoint(bytes);
            lats[i] = point.Latitude;
            lons[i] = point.Longitude;
        }

        var geo = new GeoColumn(geometryColumn, lats, lons);

        // Return DataFrame without the geometry column (it's now in the GeoColumn)
        var dataDf = df.DropColumn(geometryColumn);

        return (dataDf, geo);
    }

    /// <summary>
    /// Read a GeoParquet file as a single DataFrame with lat/lon columns extracted.
    /// </summary>
    public static DataFrame ReadAsDataFrame(string path, string? geometryColumn = null,
        string latColumn = "latitude", string lonColumn = "longitude")
    {
        var (df, geo) = Read(path, geometryColumn);
        var latValues = geo.Latitudes.ToArray();
        var lonValues = geo.Longitudes.ToArray();
        return df
            .AddColumn(new Column<double>(latColumn, latValues))
            .AddColumn(new Column<double>(lonColumn, lonValues));
    }

    private static void WriteGeoMetadata(string parquetPath, string geometryColumn, BoundingBox bounds)
    {
        var metadataPath = parquetPath + ".geo.json";
        var metadata = new
        {
            version = "1.0.0",
            primary_column = geometryColumn,
            columns = new Dictionary<string, object>
            {
                [geometryColumn] = new
                {
                    encoding = "WKB",
                    geometry_types = new[] { "Point" },
                    bbox = new[] { bounds.MinLon, bounds.MinLat, bounds.MaxLon, bounds.MaxLat }
                }
            }
        };

        var json = JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true });
        File.WriteAllText(metadataPath, json);
    }

    private static string DetectGeometryColumn(string path, DataFrame df)
    {
        // Try sidecar metadata file
        var metadataPath = path + ".geo.json";
        if (File.Exists(metadataPath))
        {
            try
            {
                var json = File.ReadAllText(metadataPath);
                using var doc = JsonDocument.Parse(json);
                if (doc.RootElement.TryGetProperty("primary_column", out var pc))
                    return pc.GetString() ?? GeometryColumnName;
            }
            catch { }
        }

        // Try common column names
        if (df.ColumnNames.Contains("geometry")) return "geometry";
        if (df.ColumnNames.Contains("geom")) return "geom";
        if (df.ColumnNames.Contains("wkb_geometry")) return "wkb_geometry";

        return GeometryColumnName;
    }
}
