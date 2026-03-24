using System.Text;

namespace PandaSharp.Display;

public static class MarkdownFormatter
{
    public static string Format(DataFrame df, int maxRows = 50)
    {
        if (df.ColumnCount == 0) return "(empty DataFrame)";

        int displayRows = Math.Min(df.RowCount, maxRows);
        var sb = new StringBuilder();

        // Header
        sb.Append("| ");
        sb.Append(string.Join(" | ", df.ColumnNames));
        sb.AppendLine(" |");

        // Separator
        sb.Append("| ");
        sb.Append(string.Join(" | ", df.ColumnNames.Select(n => IsNumeric(df[n].DataType) ? "---:" : ":---")));
        sb.AppendLine(" |");

        // Data
        for (int r = 0; r < displayRows; r++)
        {
            sb.Append("| ");
            var values = new string[df.ColumnCount];
            for (int c = 0; c < df.ColumnCount; c++)
                values[c] = FormatValue(df[df.ColumnNames[c]].GetObject(r));
            sb.Append(string.Join(" | ", values));
            sb.AppendLine(" |");
        }

        if (df.RowCount > maxRows)
            sb.AppendLine($"*... {df.RowCount - maxRows} more rows*");

        return sb.ToString();
    }

    private static string FormatValue(object? value) => value switch
    {
        null => "",
        double d when d == Math.Floor(d) && Math.Abs(d) < 1e15 => d.ToString("F0"),
        double d => d.ToString("G10"),
        float f => f.ToString("G"),
        DateTime dt => dt.ToString("yyyy-MM-dd"),
        bool b => b ? "true" : "false",
        _ => value.ToString() ?? ""
    };

    private static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal);
}
