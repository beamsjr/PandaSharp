using System.Text;

namespace Cortex.IO;

public class CsvReadOptions
{
    public char Delimiter { get; set; } = ',';
    public bool HasHeader { get; set; } = true;
    public char QuoteChar { get; set; } = '"';
    public char? CommentChar { get; set; }
    public Encoding Encoding { get; set; } = Encoding.UTF8;
    public int SkipRows { get; set; } = 0;
    public string[] NullValues { get; set; } = ["", "NA", "N/A", "null", "NULL", "None"];
    public string? DateFormat { get; set; }
    public int SampleRows { get; set; } = 100;

    /// <summary>
    /// Override type inference for specific columns. Key = column name, Value = target type.
    /// </summary>
    public Dictionary<string, Type>? ColumnTypes { get; set; }

    /// <summary>
    /// Provide a complete schema (ordered column name → type) to skip type inference entirely.
    /// When set, ColumnTypes and SampleRows are ignored. Much faster for large files.
    /// </summary>
    public (string Name, Type Type)[]? Schema { get; set; }

    /// <summary>
    /// When true (default), throw FormatException on unclosed quoted fields.
    /// Set to false for lenient parsing of messy CSV data.
    /// </summary>
    public bool StrictQuoting { get; set; } = true;
}
