using System.Numerics;
using System.Text;
using PandaSharp.Column;
using PandaSharp.Display;
using PandaSharp.Statistics;

namespace PandaSharp.Statistics;

/// <summary>
/// Comprehensive DataFrame profile — automated EDA (Exploratory Data Analysis).
/// Like Python's ydata-profiling: generates per-column statistics, correlations,
/// missing data analysis, and duplicate detection in a single call.
///
/// Usage:
///   var profile = df.Profile();
///   Console.WriteLine(profile);           // console summary
///   File.WriteAllText("report.html", profile.ToHtml());  // rich HTML report
/// </summary>
public class DataProfile
{
    public int RowCount { get; init; }
    public int ColumnCount { get; init; }
    public long MemoryBytes { get; init; }
    public int DuplicateRowCount { get; init; }
    public double DuplicateRowPercent { get; init; }
    public int TotalMissingValues { get; init; }
    public double MissingPercent { get; init; }
    public ColumnProfile[] Columns { get; init; } = [];
    public double[,]? CorrelationMatrix { get; init; }
    public string[] CorrelationColumns { get; init; } = [];

    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine("=== DataFrame Profile ===");
        sb.AppendLine($"Rows: {RowCount:N0}  Columns: {ColumnCount}  Memory: {FormatBytes(MemoryBytes)}");
        sb.AppendLine($"Duplicates: {DuplicateRowCount:N0} ({DuplicateRowPercent:F1}%)");
        sb.AppendLine($"Missing: {TotalMissingValues:N0} cells ({MissingPercent:F1}%)");
        sb.AppendLine();

        foreach (var col in Columns)
        {
            sb.AppendLine($"--- {col.Name} ({col.TypeName}) ---");
            sb.AppendLine($"  Non-null: {col.NonNullCount:N0}/{RowCount}  Null: {col.NullCount:N0} ({col.NullPercent:F1}%)  Unique: {col.UniqueCount:N0}");

            if (col.IsNumeric)
            {
                sb.AppendLine($"  Mean: {col.Mean:F4}  Std: {col.Std:F4}  Min: {col.Min:F4}  Max: {col.Max:F4}");
                sb.AppendLine($"  25%: {col.Q25:F4}  50%: {col.Median:F4}  75%: {col.Q75:F4}");
                sb.AppendLine($"  Skew: {col.Skew:F4}  Kurtosis: {col.Kurtosis:F4}  Zeros: {col.ZeroCount} ({col.ZeroPercent:F1}%)");
            }

            if (col.IsString)
            {
                sb.AppendLine($"  Min length: {col.MinLength}  Max length: {col.MaxLength}  Mean length: {col.MeanLength:F1}");
                sb.AppendLine($"  Empty strings: {col.EmptyStringCount}");
            }

            if (col.TopValues.Length > 0)
            {
                sb.Append("  Top values: ");
                sb.AppendLine(string.Join(", ", col.TopValues.Take(5).Select(t => $"{t.Value}({t.Count})")));
            }
        }

        if (CorrelationMatrix is not null && CorrelationColumns.Length > 0)
        {
            sb.AppendLine();
            sb.AppendLine("=== Correlation Matrix ===");
            sb.Append("         ");
            foreach (var c in CorrelationColumns) sb.Append($"{c,10}");
            sb.AppendLine();
            for (int r = 0; r < CorrelationColumns.Length; r++)
            {
                sb.Append($"{CorrelationColumns[r],9}");
                for (int c = 0; c < CorrelationColumns.Length; c++)
                    sb.Append($"{CorrelationMatrix[r, c],10:F3}");
                sb.AppendLine();
            }
        }

        return sb.ToString();
    }

    public string ToHtml()
    {
        var sb = new StringBuilder();
        sb.AppendLine("<!DOCTYPE html><html><head><meta charset='utf-8'>");
        sb.AppendLine("<title>DataFrame Profile</title>");
        sb.AppendLine("<style>");
        sb.AppendLine("body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:20px;background:#f5f5f5}");
        sb.AppendLine(".card{background:white;border-radius:8px;padding:20px;margin:16px 0;box-shadow:0 1px 3px rgba(0,0,0,.12)}");
        sb.AppendLine("h1{color:#1a1a2e}h2{color:#16213e;border-bottom:2px solid #0f3460;padding-bottom:8px}");
        sb.AppendLine("table{border-collapse:collapse;width:100%}th,td{padding:8px 12px;text-align:left;border-bottom:1px solid #eee}");
        sb.AppendLine("th{background:#f8f9fa;font-weight:600}.num{text-align:right;font-variant-numeric:tabular-nums}");
        sb.AppendLine(".warn{color:#e74c3c}.ok{color:#27ae60}.bar{background:#3498db;height:8px;border-radius:4px}");
        sb.AppendLine(".bar-bg{background:#ecf0f1;border-radius:4px;overflow:hidden;width:100px;display:inline-block}");
        sb.AppendLine("</style></head><body>");

        // Overview
        sb.AppendLine("<h1>DataFrame Profile Report</h1>");
        sb.AppendLine("<div class='card'><h2>Overview</h2><table>");
        sb.AppendLine($"<tr><td>Rows</td><td class='num'>{RowCount:N0}</td></tr>");
        sb.AppendLine($"<tr><td>Columns</td><td class='num'>{ColumnCount}</td></tr>");
        sb.AppendLine($"<tr><td>Memory</td><td class='num'>{FormatBytes(MemoryBytes)}</td></tr>");
        sb.AppendLine($"<tr><td>Duplicate rows</td><td class='num'>{DuplicateRowCount:N0} ({DuplicateRowPercent:F1}%)</td></tr>");
        sb.AppendLine($"<tr><td>Missing values</td><td class='num'>{TotalMissingValues:N0} ({MissingPercent:F1}%)</td></tr>");
        sb.AppendLine("</table></div>");

        // Column details
        foreach (var col in Columns)
        {
            sb.AppendLine($"<div class='card'><h2>{EscapeHtml(col.Name)} <small style='color:#888'>({col.TypeName})</small></h2>");
            sb.AppendLine("<table>");
            sb.AppendLine($"<tr><td>Non-null</td><td class='num'>{col.NonNullCount:N0}</td></tr>");

            var nullClass = col.NullPercent > 5 ? "warn" : "ok";
            sb.AppendLine($"<tr><td>Null</td><td class='num {nullClass}'>{col.NullCount:N0} ({col.NullPercent:F1}%)</td></tr>");
            sb.AppendLine($"<tr><td>Unique</td><td class='num'>{col.UniqueCount:N0}</td></tr>");

            if (col.IsNumeric)
            {
                sb.AppendLine($"<tr><td>Mean</td><td class='num'>{col.Mean:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Std</td><td class='num'>{col.Std:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Min</td><td class='num'>{col.Min:F4}</td></tr>");
                sb.AppendLine($"<tr><td>25%</td><td class='num'>{col.Q25:F4}</td></tr>");
                sb.AppendLine($"<tr><td>50% (Median)</td><td class='num'>{col.Median:F4}</td></tr>");
                sb.AppendLine($"<tr><td>75%</td><td class='num'>{col.Q75:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Max</td><td class='num'>{col.Max:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Skewness</td><td class='num'>{col.Skew:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Kurtosis</td><td class='num'>{col.Kurtosis:F4}</td></tr>");
                sb.AppendLine($"<tr><td>Zeros</td><td class='num'>{col.ZeroCount} ({col.ZeroPercent:F1}%)</td></tr>");
            }

            if (col.IsString)
            {
                sb.AppendLine($"<tr><td>Min length</td><td class='num'>{col.MinLength}</td></tr>");
                sb.AppendLine($"<tr><td>Max length</td><td class='num'>{col.MaxLength}</td></tr>");
                sb.AppendLine($"<tr><td>Mean length</td><td class='num'>{col.MeanLength:F1}</td></tr>");
                sb.AppendLine($"<tr><td>Empty strings</td><td class='num'>{col.EmptyStringCount}</td></tr>");
            }

            if (col.TopValues.Length > 0)
            {
                sb.AppendLine("<tr><td colspan='2'><strong>Top values</strong></td></tr>");
                foreach (var tv in col.TopValues.Take(10))
                {
                    double pct = RowCount > 0 ? tv.Count * 100.0 / RowCount : 0;
                    sb.AppendLine($"<tr><td>{EscapeHtml(tv.Value)}</td><td class='num'>{tv.Count} ({pct:F1}%) " +
                        $"<div class='bar-bg'><div class='bar' style='width:{Math.Min(pct, 100):F0}px'></div></div></td></tr>");
                }
            }

            sb.AppendLine("</table></div>");
        }

        // Correlation matrix
        if (CorrelationMatrix is not null && CorrelationColumns.Length > 1)
        {
            sb.AppendLine("<div class='card'><h2>Correlation Matrix</h2><table><tr><th></th>");
            foreach (var c in CorrelationColumns) sb.Append($"<th>{EscapeHtml(c)}</th>");
            sb.AppendLine("</tr>");
            for (int r = 0; r < CorrelationColumns.Length; r++)
            {
                sb.Append($"<tr><th>{EscapeHtml(CorrelationColumns[r])}</th>");
                for (int c = 0; c < CorrelationColumns.Length; c++)
                {
                    double val = CorrelationMatrix[r, c];
                    string color = val > 0.7 ? "#27ae60" : val < -0.7 ? "#e74c3c" : "#333";
                    sb.Append($"<td class='num' style='color:{color}'>{val:F3}</td>");
                }
                sb.AppendLine("</tr>");
            }
            sb.AppendLine("</table></div>");
        }

        sb.AppendLine("<footer style='text-align:center;color:#888;margin:20px'>");
        sb.AppendLine($"Generated by PandaSharp Profile at {DateTime.Now:yyyy-MM-dd HH:mm:ss}");
        sb.AppendLine("</footer></body></html>");
        return sb.ToString();
    }

    private static string FormatBytes(long bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1024 * 1024 => $"{bytes / 1024.0:F1} KB",
        < 1024 * 1024 * 1024 => $"{bytes / (1024.0 * 1024):F1} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):F2} GB"
    };

    private static string EscapeHtml(string s) =>
        s.Replace("&", "&amp;").Replace("<", "&lt;").Replace(">", "&gt;");
}

public record struct TopValue(string Value, int Count);

public record ColumnProfile
{
    public string Name { get; init; } = "";
    public string TypeName { get; init; } = "";
    public bool IsNumeric { get; init; }
    public bool IsString { get; init; }
    public int NonNullCount { get; init; }
    public int NullCount { get; init; }
    public double NullPercent { get; init; }
    public int UniqueCount { get; init; }
    public TopValue[] TopValues { get; init; } = [];

    // Numeric stats
    public double Mean { get; init; }
    public double Std { get; init; }
    public double Min { get; init; }
    public double Max { get; init; }
    public double Median { get; init; }
    public double Q25 { get; init; }
    public double Q75 { get; init; }
    public double Skew { get; init; }
    public double Kurtosis { get; init; }
    public int ZeroCount { get; init; }
    public double ZeroPercent { get; init; }

    // String stats
    public int MinLength { get; init; }
    public int MaxLength { get; init; }
    public double MeanLength { get; init; }
    public int EmptyStringCount { get; init; }
}
