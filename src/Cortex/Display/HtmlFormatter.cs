using System.Text;
using System.Web;

namespace Cortex.Display;

public static class HtmlFormatter
{
    public static string Format(DataFrame df, int maxRows = 50)
    {
        if (df.ColumnCount == 0) return "<p>(empty DataFrame)</p>";

        int displayRows = Math.Min(df.RowCount, maxRows);
        bool truncated = df.RowCount > maxRows;

        var sb = new StringBuilder();
        sb.AppendLine("<table style=\"border-collapse:collapse;font-family:monospace;font-size:13px;\">");

        // Header
        sb.AppendLine("<thead><tr>");
        sb.Append("<th style=\"border:1px solid #ddd;padding:4px 8px;background:#f5f5f5;\"></th>");
        foreach (var name in df.ColumnNames)
        {
            sb.Append($"<th style=\"border:1px solid #ddd;padding:4px 8px;background:#f5f5f5;\">{Escape(name)}</th>");
        }
        sb.AppendLine("</tr></thead>");

        // Body
        sb.AppendLine("<tbody>");
        for (int r = 0; r < displayRows; r++)
        {
            sb.Append("<tr>");
            sb.Append($"<td style=\"border:1px solid #ddd;padding:4px 8px;background:#fafafa;font-weight:bold;\">{r}</td>");
            for (int c = 0; c < df.ColumnCount; c++)
            {
                var col = df[df.ColumnNames[c]];
                var val = FormatValue(col.GetObject(r));
                string align = IsNumeric(col.DataType) ? "right" : "left";
                string style = col.IsNull(r)
                    ? "border:1px solid #ddd;padding:4px 8px;color:#999;font-style:italic;"
                    : $"border:1px solid #ddd;padding:4px 8px;text-align:{align};";
                sb.Append($"<td style=\"{style}\">{Escape(val)}</td>");
            }
            sb.AppendLine("</tr>");
        }

        if (truncated)
        {
            sb.Append("<tr>");
            sb.Append($"<td style=\"border:1px solid #ddd;padding:4px 8px;\">...</td>");
            for (int c = 0; c < df.ColumnCount; c++)
                sb.Append("<td style=\"border:1px solid #ddd;padding:4px 8px;\">...</td>");
            sb.AppendLine("</tr>");
        }

        sb.AppendLine("</tbody></table>");
        sb.Append($"<p style=\"font-size:12px;color:#666;\">{df.RowCount} rows × {df.ColumnCount} columns</p>");

        return sb.ToString();
    }

    private static string FormatValue(object? value) => value switch
    {
        null => "null",
        double d => d.ToString("G"),
        float f => f.ToString("G"),
        DateTime dt => dt.ToString("yyyy-MM-dd HH:mm:ss"),
        _ => value.ToString() ?? "null"
    };

    private static string Escape(string text) => HttpUtility.HtmlEncode(text);

    private static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal);
}
