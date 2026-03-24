namespace PandaSharp.Statistics;

public record ColumnInfo(string Name, string DataType, int NonNullCount, int NullCount, long EstimatedBytes);

public class InfoResult
{
    public int RowCount { get; init; }
    public int ColumnCount { get; init; }
    public IReadOnlyList<ColumnInfo> Columns { get; init; } = [];
    public long TotalEstimatedBytes => Columns.Sum(c => c.EstimatedBytes);

    public override string ToString()
    {
        var lines = new List<string>
        {
            $"DataFrame: {RowCount} rows x {ColumnCount} columns",
            $"{"Column",-20} {"Type",-12} {"Non-Null",-10} {"Null",-8} {"Memory",8}",
            new string('-', 62)
        };

        foreach (var col in Columns)
        {
            lines.Add($"{col.Name,-20} {col.DataType,-12} {col.NonNullCount,-10} {col.NullCount,-8} {FormatBytes(col.EstimatedBytes),8}");
        }

        lines.Add(new string('-', 62));
        lines.Add($"Total memory: {FormatBytes(TotalEstimatedBytes)}");

        return string.Join(Environment.NewLine, lines);
    }

    private static string FormatBytes(long bytes)
    {
        if (bytes < 1024) return $"{bytes} B";
        if (bytes < 1024 * 1024) return $"{bytes / 1024.0:F1} KB";
        return $"{bytes / (1024.0 * 1024.0):F1} MB";
    }
}
