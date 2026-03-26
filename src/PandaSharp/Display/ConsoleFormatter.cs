using System.Text;

namespace PandaSharp.Display;

public static class ConsoleFormatter
{
    public static string Format(DataFrame df, int maxRows = 20)
    {
        if (df.ColumnCount == 0) return "(empty DataFrame)";

        // For large DataFrames, show first 5 + last 5 rows with "..." separator
        int headCount = maxRows;
        int tailCount = 0;
        bool showEllipsis = false;

        if (df.RowCount > maxRows && maxRows >= 10)
        {
            headCount = maxRows / 2;
            tailCount = maxRows / 2;
            showEllipsis = true;
        }
        else if (df.RowCount > maxRows)
        {
            headCount = maxRows;
            showEllipsis = true;
        }

        headCount = Math.Min(headCount, df.RowCount);
        if (showEllipsis && tailCount > 0)
            tailCount = Math.Min(tailCount, df.RowCount - headCount);

        // Calculate column widths from visible rows
        int indexWidth = Math.Max(df.RowCount.ToString().Length, 1);
        var colWidths = new int[df.ColumnCount];

        for (int c = 0; c < df.ColumnCount; c++)
        {
            colWidths[c] = df.ColumnNames[c].Length;
            var col = df[df.ColumnNames[c]];
            for (int r = 0; r < headCount; r++)
            {
                var val = FormatValue(col.GetObject(r));
                colWidths[c] = Math.Max(colWidths[c], val.Length);
            }
            if (tailCount > 0)
            {
                int tailStart = df.RowCount - tailCount;
                for (int r = tailStart; r < df.RowCount; r++)
                {
                    var val = FormatValue(col.GetObject(r));
                    colWidths[c] = Math.Max(colWidths[c], val.Length);
                }
            }
        }

        var sb = new StringBuilder();

        // Top border
        sb.Append('┌');
        sb.Append(new string('─', indexWidth + 2));
        for (int c = 0; c < df.ColumnCount; c++)
        {
            sb.Append('┬');
            sb.Append(new string('─', colWidths[c] + 2));
        }
        sb.AppendLine("┐");

        // Header
        sb.Append("│ ");
        sb.Append(new string(' ', indexWidth));
        sb.Append(" │");
        for (int c = 0; c < df.ColumnCount; c++)
        {
            sb.Append(' ');
            sb.Append(df.ColumnNames[c].PadRight(colWidths[c]));
            sb.Append(" │");
        }
        sb.AppendLine();

        // Header separator
        sb.Append('├');
        sb.Append(new string('─', indexWidth + 2));
        for (int c = 0; c < df.ColumnCount; c++)
        {
            sb.Append('┼');
            sb.Append(new string('─', colWidths[c] + 2));
        }
        sb.AppendLine("┤");

        // Head rows
        for (int r = 0; r < headCount; r++)
        {
            AppendDataRow(sb, df, r, indexWidth, colWidths);
        }

        // Ellipsis row
        if (showEllipsis)
        {
            sb.Append("│ ");
            sb.Append("...".PadLeft(indexWidth));
            sb.Append(" │");
            for (int c = 0; c < df.ColumnCount; c++)
            {
                sb.Append(' ');
                sb.Append("...".PadRight(colWidths[c]));
                sb.Append(" │");
            }
            sb.AppendLine();
        }

        // Tail rows
        if (tailCount > 0)
        {
            int tailStart = df.RowCount - tailCount;
            for (int r = tailStart; r < df.RowCount; r++)
            {
                AppendDataRow(sb, df, r, indexWidth, colWidths);
            }
        }

        // Bottom border
        sb.Append('└');
        sb.Append(new string('─', indexWidth + 2));
        for (int c = 0; c < df.ColumnCount; c++)
        {
            sb.Append('┴');
            sb.Append(new string('─', colWidths[c] + 2));
        }
        sb.AppendLine("┘");

        sb.Append($"[{df.RowCount} rows x {df.ColumnCount} columns]");

        return sb.ToString();
    }

    private static void AppendDataRow(StringBuilder sb, DataFrame df, int r, int indexWidth, int[] colWidths)
    {
        sb.Append("│ ");
        sb.Append(r.ToString().PadLeft(indexWidth));
        sb.Append(" │");
        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            var val = FormatValue(col.GetObject(r));
            bool rightAlign = IsNumeric(col.DataType);
            sb.Append(' ');
            sb.Append(rightAlign ? val.PadLeft(colWidths[c]) : val.PadRight(colWidths[c]));
            sb.Append(" │");
        }
        sb.AppendLine();
    }

    private static string FormatValue(object? value) => value switch
    {
        null => "null",
        double d => FormatDouble(d),
        float f => FormatDouble(f),
        DateTime dt => dt.ToString("yyyy-MM-dd HH:mm:ss").TrimEnd('0', ':').TrimEnd(' '),
        _ => value.ToString() ?? "null"
    };

    private static string FormatDouble(double d)
    {
        if (d == Math.Floor(d) && Math.Abs(d) < 1e15)
            return d.ToString("F0");
        return d.ToString("G10");
    }

    private static bool IsNumeric(Type type) =>
        type == typeof(int) || type == typeof(long) ||
        type == typeof(double) || type == typeof(float) ||
        type == typeof(decimal) || type == typeof(short) ||
        type == typeof(byte);
}
