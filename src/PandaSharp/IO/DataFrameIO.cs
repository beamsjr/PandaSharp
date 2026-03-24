namespace PandaSharp.IO;

/// <summary>
/// Extension methods adding I/O capabilities to DataFrame.
/// </summary>
public static class DataFrameIO
{
    // -- CSV --
    public static DataFrame ReadCsv(string path, CsvReadOptions? options = null)
        => CsvReader.Read(path, options);

    public static DataFrame ReadCsv(Stream stream, CsvReadOptions? options = null)
        => CsvReader.Read(stream, options);

    public static IEnumerable<DataFrame> ReadCsvChunked(string path, int chunkSize, CsvReadOptions? options = null)
        => CsvReader.ReadChunked(path, chunkSize, options);

    public static void ToCsv(this DataFrame df, string path, CsvWriteOptions? options = null)
        => CsvWriter.Write(df, path, options);

    public static void ToCsv(this DataFrame df, Stream stream, CsvWriteOptions? options = null)
        => CsvWriter.Write(df, stream, options);

    // -- JSON --
    public static DataFrame ReadJson(string path) => JsonReader.Read(path);
    public static DataFrame ReadJsonString(string json) => JsonReader.ReadString(json);

    public static void ToJson(this DataFrame df, string path, JsonOrient orient = JsonOrient.Records)
        => JsonWriter.Write(df, path, orient);

    public static string ToJsonString(this DataFrame df, JsonOrient orient = JsonOrient.Records)
        => JsonWriter.WriteString(df, orient);

    // -- Arrow IPC --
    public static DataFrame ReadArrow(string path) => ArrowIpcReader.Read(path);
    public static DataFrame ReadArrow(Stream stream) => ArrowIpcReader.Read(stream);

    public static void ToArrow(this DataFrame df, string path) => ArrowIpcWriter.Write(df, path);
    public static void ToArrow(this DataFrame df, Stream stream) => ArrowIpcWriter.Write(df, stream);

    // -- Parquet --
    public static DataFrame ReadParquet(string path, ParquetReadOptions? options = null)
        => ParquetIO.ReadParquet(path, options);
    public static void ToParquet(this DataFrame df, string path)
        => ParquetIO.WriteParquet(df, path);

    // -- Excel --
    public static DataFrame ReadExcel(string path, ExcelReadOptions? options = null)
        => ExcelIO.ReadExcel(path, options);
    public static void ToExcel(this DataFrame df, string path, string sheetName = "Sheet1")
        => ExcelIO.WriteExcel(df, path, sheetName);

    // -- HTML --
    public static List<DataFrame> ReadHtml(string html, int? tableIndex = null)
        => HtmlTableReader.ReadHtml(html, tableIndex);
    public static DataFrame ReadHtmlFile(string path, int tableIndex = 0)
        => HtmlTableReader.ReadHtmlFile(path, tableIndex);

    // -- Clipboard --
    public static DataFrame FromClipboard(CsvReadOptions? options = null)
        => ClipboardIO.FromClipboard(options);
    public static void ToClipboard(this DataFrame df, char delimiter = '\t')
        => ClipboardIO.ToClipboard(df, delimiter);

    // -- Universal Save/Load --

    /// <summary>
    /// Save a DataFrame to any supported format, auto-detected by file extension.
    /// Supported: .csv, .csv.gz, .tsv, .json, .jsonl, .parquet, .arrow, .xlsx
    /// </summary>
    public static void Save(this DataFrame df, string path)
    {
        var ext = GetEffectiveExtension(path);
        switch (ext)
        {
            case ".csv" or ".csv.gz" or ".tsv":
                var csvOptions = ext == ".tsv" ? new CsvWriteOptions { Delimiter = '\t' } : null;
                CsvWriter.Write(df, path, csvOptions);
                break;
            case ".json":
                JsonWriter.Write(df, path);
                break;
            case ".jsonl":
                File.WriteAllText(path, ToJsonLines(df));
                break;
            case ".parquet":
                ParquetIO.WriteParquet(df, path);
                break;
            case ".arrow" or ".ipc":
                ArrowIpcWriter.Write(df, path);
                break;
            case ".xlsx":
                ExcelIO.WriteExcel(df, path);
                break;
            default:
                throw new NotSupportedException(
                    $"Unknown file format '{ext}'. Supported: .csv, .csv.gz, .tsv, .json, .jsonl, .parquet, .arrow, .xlsx");
        }
    }

    /// <summary>
    /// Load a DataFrame from any supported format, auto-detected by file extension.
    /// </summary>
    public static DataFrame Load(string path)
    {
        if (!File.Exists(path))
            throw new FileNotFoundException($"File not found: {path}");

        var ext = GetEffectiveExtension(path);
        return ext switch
        {
            ".csv" or ".csv.gz" or ".tsv" => CsvReader.Read(path,
                ext == ".tsv" ? new CsvReadOptions { Delimiter = '\t' } : null),
            ".json" => JsonReader.Read(path),
            ".jsonl" => JsonReader.ReadLines(path),
            ".parquet" => ParquetIO.ReadParquet(path),
            ".arrow" or ".ipc" => ArrowIpcReader.Read(path),
            ".xlsx" => ExcelIO.ReadExcel(path),
            _ => throw new NotSupportedException(
                $"Unknown file format '{ext}'. Supported: .csv, .csv.gz, .tsv, .json, .jsonl, .parquet, .arrow, .xlsx")
        };
    }

    private static string GetEffectiveExtension(string path)
    {
        if (path.EndsWith(".csv.gz", StringComparison.OrdinalIgnoreCase))
            return ".csv.gz";
        return Path.GetExtension(path).ToLowerInvariant();
    }

    private static string ToJsonLines(DataFrame df)
    {
        var sb = new System.Text.StringBuilder();
        for (int r = 0; r < df.RowCount; r++)
        {
            sb.Append('{');
            for (int c = 0; c < df.ColumnCount; c++)
            {
                if (c > 0) sb.Append(',');
                var name = df.ColumnNames[c];
                var val = df[name].GetObject(r);
                sb.Append($"\"{name}\":");
                if (val is null)
                    sb.Append("null");
                else if (val is string s)
                    sb.Append($"\"{s.Replace("\"", "\\\"")}\"");
                else if (val is bool b)
                    sb.Append(b ? "true" : "false");
                else
                    sb.Append(Convert.ToString(val, System.Globalization.CultureInfo.InvariantCulture));
            }
            sb.AppendLine("}");
        }
        return sb.ToString();
    }
}
