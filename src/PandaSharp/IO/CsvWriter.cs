using System.Globalization;
using System.IO.Compression;
using System.Text;

namespace PandaSharp.IO;

public class CsvWriteOptions
{
    public char Delimiter { get; set; } = ',';
    public bool WriteHeader { get; set; } = true;
    public Encoding Encoding { get; set; } = Encoding.UTF8;
    public string NullRepresentation { get; set; } = "";
}

public static class CsvWriter
{
    public static void Write(DataFrame df, string path, CsvWriteOptions? options = null)
    {
        using var stream = File.Create(path);

        // Auto-detect gzip by extension
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
        {
            using var gzip = new GZipStream(stream, CompressionLevel.Optimal);
            Write(df, gzip, options);
            return;
        }

        Write(df, stream, options);
    }

    public static void Write(DataFrame df, Stream stream, CsvWriteOptions? options = null)
    {
        options ??= new CsvWriteOptions();
        using var writer = new StreamWriter(stream, options.Encoding, leaveOpen: true);

        if (options.WriteHeader)
        {
            var quotedHeaders = df.ColumnNames.Select(name => QuoteHeaderIfNeeded(name, options.Delimiter)).ToArray();
            writer.WriteLine(string.Join(options.Delimiter, quotedHeaders));
        }

        for (int r = 0; r < df.RowCount; r++)
        {
            var fields = new string[df.ColumnCount];
            for (int c = 0; c < df.ColumnCount; c++)
            {
                var col = df[df.ColumnNames[c]];
                var val = col.GetObject(r);
                fields[c] = FormatValue(val, options);
            }
            writer.WriteLine(string.Join(options.Delimiter, fields));
        }
    }

    private static string QuoteHeaderIfNeeded(string name, char delimiter)
    {
        if (name.Contains(delimiter) || name.Contains('"') || name.Contains('\n') || name.Contains('\r'))
        {
            return $"\"{name.Replace("\"", "\"\"")}\"";
        }
        return name;
    }

    private static string FormatValue(object? value, CsvWriteOptions options)
    {
        if (value is null) return options.NullRepresentation;

        var str = value switch
        {
            double d => d.ToString("G", CultureInfo.InvariantCulture),
            float f => f.ToString("G", CultureInfo.InvariantCulture),
            DateTime dt => dt.ToString("yyyy-MM-dd HH:mm:ss", CultureInfo.InvariantCulture),
            _ => value.ToString() ?? options.NullRepresentation
        };

        // Normalize backslash-escaped quotes to bare quotes before RFC 4180 quoting
        if (str.Contains("\\\""))
            str = str.Replace("\\\"", "\"");

        // Quote if contains delimiter, quote, or newline/carriage return
        if (str.Contains(options.Delimiter) || str.Contains('"') || str.Contains('\n') || str.Contains('\r'))
        {
            str = $"\"{str.Replace("\"", "\"\"")}\"";
        }
        // Quote empty strings to distinguish from null (which is written as unquoted NullRepresentation)
        else if (str.Length == 0 && str == options.NullRepresentation)
        {
            str = "\"\"";
        }

        return str;
    }
}
