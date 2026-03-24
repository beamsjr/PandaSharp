using System.Text;
using Microsoft.DotNet.Interactive;
using Microsoft.DotNet.Interactive.Formatting;
using PandaSharp.Column;

namespace PandaSharp.Interactive;

/// <summary>
/// .NET Interactive / Polyglot Notebooks integration.
/// Auto-registers HTML formatters so DataFrames render as styled tables in notebook cells.
///
/// Usage in a notebook:
///   #r "nuget: PandaSharp.Interactive"
///   using PandaSharp;
///   var df = DataFrame.FromDictionary(new() { ["Name"] = new[] { "Alice", "Bob" }, ["Age"] = new[] { 25, 30 } });
///   df  // renders as styled HTML table
/// </summary>
public class DataFrameKernelExtension : IKernelExtension
{
    public Task OnLoadAsync(Kernel kernel)
    {
        RegisterFormatters();
        return Task.CompletedTask;
    }

    public static void RegisterFormatters()
    {
        Formatter.Register<DataFrame>(FormatDelegate, HtmlFormatter.MimeType);

        Formatter.Register<Statistics.InfoResult>((info, context) =>
        {
            context.Writer.Write($"<pre>{System.Web.HttpUtility.HtmlEncode(info.ToString())}</pre>");
            return true;
        }, HtmlFormatter.MimeType);
    }

    /// <summary>Maximum rows to display. Configurable.</summary>
    public static int MaxRows { get; set; } = 25;

    /// <summary>Maximum columns to display before truncating.</summary>
    public static int MaxColumns { get; set; } = 20;

    private static bool FormatDelegate(DataFrame df, FormatContext context)
    {
        context.Writer.Write(FormatDataFrame(df));
        return true;
    }

    /// <summary>
    /// Generate the styled HTML for a DataFrame. Public for direct use and testing.
    /// </summary>
    public static string FormatDataFrame(DataFrame df)
    {
        var sb = new StringBuilder();
        int displayRows = Math.Min(df.RowCount, MaxRows);
        int displayCols = Math.Min(df.ColumnCount, MaxColumns);
        bool rowsTruncated = df.RowCount > MaxRows;
        bool colsTruncated = df.ColumnCount > MaxColumns;

        sb.AppendLine("<div style=\"max-height:600px;overflow:auto;\">");
        sb.AppendLine("<table style=\"border-collapse:collapse;font-family:'Segoe UI',system-ui,sans-serif;font-size:13px;width:auto;\">");

        // Header
        sb.AppendLine("<thead>");
        sb.Append("<tr style=\"background:#f8f9fa;border-bottom:2px solid #dee2e6;\">");
        sb.Append("<th style=\"padding:6px 12px;text-align:left;color:#6c757d;font-weight:500;\"></th>");
        for (int c = 0; c < displayCols; c++)
        {
            var name = df.ColumnNames[c];
            var dtype = FormatDtype(df[name].DataType);
            sb.Append($"<th style=\"padding:6px 12px;text-align:left;font-weight:600;\">");
            sb.Append($"{Escape(name)}<br/><span style=\"font-weight:400;color:#6c757d;font-size:11px;\">{dtype}</span>");
            sb.Append("</th>");
        }
        if (colsTruncated) sb.Append("<th style=\"padding:6px 12px;color:#6c757d;\">...</th>");
        sb.AppendLine("</tr>");
        sb.AppendLine("</thead>");

        // Body
        sb.AppendLine("<tbody>");
        for (int r = 0; r < displayRows; r++)
        {
            string bg = r % 2 == 0 ? "#ffffff" : "#f8f9fa";
            sb.Append($"<tr style=\"background:{bg};border-bottom:1px solid #eee;\">");
            sb.Append($"<td style=\"padding:4px 12px;color:#6c757d;font-weight:500;\">{r}</td>");
            for (int c = 0; c < displayCols; c++)
            {
                var col = df[df.ColumnNames[c]];
                bool isNull = col.IsNull(r);
                string val = isNull ? "null" : FormatValue(col.GetObject(r)!, col.DataType);
                string align = IsNumeric(col.DataType) ? "right" : "left";
                string style = isNull
                    ? $"padding:4px 12px;text-align:{align};color:#adb5bd;font-style:italic;"
                    : $"padding:4px 12px;text-align:{align};";
                sb.Append($"<td style=\"{style}\">{Escape(val)}</td>");
            }
            if (colsTruncated) sb.Append("<td style=\"padding:4px 12px;color:#adb5bd;\">...</td>");
            sb.AppendLine("</tr>");
        }

        if (rowsTruncated)
        {
            sb.Append("<tr style=\"background:#fff;\">");
            sb.Append("<td style=\"padding:4px 12px;color:#adb5bd;\">...</td>");
            for (int c = 0; c < displayCols; c++)
                sb.Append("<td style=\"padding:4px 12px;color:#adb5bd;\">...</td>");
            if (colsTruncated) sb.Append("<td></td>");
            sb.AppendLine("</tr>");
        }

        sb.AppendLine("</tbody>");
        sb.AppendLine("</table>");

        // Footer
        sb.Append("<div style=\"font-size:12px;color:#6c757d;padding:4px 0;\">");
        sb.Append($"{df.RowCount:N0} rows × {df.ColumnCount} columns");
        if (rowsTruncated) sb.Append($" (showing first {MaxRows})");
        sb.AppendLine("</div>");
        sb.AppendLine("</div>");

        return sb.ToString();
    }

    private static string FormatValue(object value, Type dtype) => value switch
    {
        double d when d == Math.Floor(d) && Math.Abs(d) < 1e15 => d.ToString("N0"),
        double d => d.ToString("G6"),
        float f => f.ToString("G6"),
        DateTime dt => dt.TimeOfDay == TimeSpan.Zero ? dt.ToString("yyyy-MM-dd") : dt.ToString("yyyy-MM-dd HH:mm:ss"),
        bool b => b ? "true" : "false",
        int i => i.ToString("N0"),
        long l => l.ToString("N0"),
        _ => value.ToString() ?? ""
    };

    private static string FormatDtype(Type type) => type switch
    {
        _ when type == typeof(int) => "int32",
        _ when type == typeof(long) => "int64",
        _ when type == typeof(double) => "float64",
        _ when type == typeof(float) => "float32",
        _ when type == typeof(bool) => "bool",
        _ when type == typeof(DateTime) => "datetime",
        _ when type == typeof(string) => "string",
        _ => type.Name
    };

    private static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal);

    private static string Escape(string text) =>
        text.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
}
