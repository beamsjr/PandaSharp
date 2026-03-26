using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

/// <summary>
/// Lightweight column metadata for the execution plan.
/// Named ExplainColumnInfo to avoid collision with InfoResult.ColumnInfo.
/// </summary>
public record ExplainColumnInfo(
    string Name,
    Type DataType,
    int NullCount,
    double NullPercent,
    long MemoryBytes);

/// <summary>
/// Execution plan summary: column metadata + estimated memory.
/// O(columns) — never touches row data.
/// </summary>
public record ExecutionPlan(
    int RowCount,
    int ColumnCount,
    IReadOnlyList<ExplainColumnInfo> Columns,
    long EstimatedMemoryBytes)
{
    public override string ToString()
    {
        var sb = new StringBuilder();
        sb.AppendLine($"ExecutionPlan: {RowCount:N0} rows x {ColumnCount} columns  (~{FormatBytes(EstimatedMemoryBytes)})");
        sb.AppendLine($"{"Column",-24} {"Type",-14} {"Nulls",-8} {"Null%",-8} {"Memory",10}");
        sb.AppendLine(new string('-', 68));

        foreach (var col in Columns)
        {
            sb.AppendLine(
                $"{col.Name,-24} {col.DataType.Name,-14} {col.NullCount,-8} {col.NullPercent,6:F1}%  {FormatBytes(col.MemoryBytes),10}");
        }

        return sb.ToString();
    }

    private static string FormatBytes(long bytes) => bytes switch
    {
        < 1024 => $"{bytes} B",
        < 1024 * 1024 => $"{bytes / 1024.0:F1} KB",
        < 1024L * 1024 * 1024 => $"{bytes / (1024.0 * 1024):F1} MB",
        _ => $"{bytes / (1024.0 * 1024 * 1024):F2} GB"
    };
}

public static class ExplainExtensions
{
    /// <summary>
    /// Returns an ExecutionPlan with column metadata and memory estimates.
    /// O(columns) complexity — walks column metadata only, never materializes data.
    /// Uses Unsafe.SizeOf&lt;T&gt;() for accurate struct memory calculation.
    /// </summary>
    public static ExecutionPlan Explain(this DataFrame df)
    {
        var columns = new ExplainColumnInfo[df.ColumnCount];
        long totalMemory = 0;

        for (int i = 0; i < df.ColumnCount; i++)
        {
            var col = df[df.ColumnNames[i]];
            long memBytes = EstimateColumnMemory(col);
            totalMemory += memBytes;

            double nullPct = col.Length > 0
                ? col.NullCount * 100.0 / col.Length
                : 0.0;

            columns[i] = new ExplainColumnInfo(
                col.Name,
                col.DataType,
                col.NullCount,
                nullPct,
                memBytes);
        }

        return new ExecutionPlan(df.RowCount, df.ColumnCount, columns, totalMemory);
    }

    /// <summary>
    /// Estimate memory for a column using Unsafe.SizeOf for struct columns.
    /// For string columns, uses reference array overhead + average char estimate
    /// without iterating rows (uses NullCount metadata only).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long EstimateColumnMemory(IColumn col)
    {
        // String column: reference array + estimate 40 bytes per non-null string
        if (col is StringColumn)
        {
            long refArrayBytes = (long)col.Length * IntPtr.Size;
            long stringEstimate = (long)(col.Length - col.NullCount) * 40;
            return refArrayBytes + stringEstimate;
        }

        // Struct-backed columns: use Unsafe.SizeOf<T> via type dispatch
        long elementSize = GetElementSize(col.DataType);
        long dataBytes = elementSize * col.Length;

        // Null bitmask: ceil(length / 8) bytes (0 if all-valid sentinel)
        long bitmaskBytes = col.NullCount > 0 ? (col.Length + 7) / 8 : 0;

        return dataBytes + bitmaskBytes;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static long GetElementSize(Type type)
    {
        if (type == typeof(int)) return Unsafe.SizeOf<int>();
        if (type == typeof(long)) return Unsafe.SizeOf<long>();
        if (type == typeof(double)) return Unsafe.SizeOf<double>();
        if (type == typeof(float)) return Unsafe.SizeOf<float>();
        if (type == typeof(bool)) return Unsafe.SizeOf<bool>();
        if (type == typeof(DateTime)) return Unsafe.SizeOf<DateTime>();
        // Fallback for unknown struct types
        return Marshal.SizeOf(type);
    }
}
