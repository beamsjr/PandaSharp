using System.Runtime.InteropServices;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class InfoExtensions
{
    public static InfoResult Info(this DataFrame df)
    {
        var columns = new List<ColumnInfo>();

        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            long estimatedBytes = EstimateMemory(col);
            columns.Add(new ColumnInfo(
                Name: col.Name,
                DataType: FormatType(col.DataType),
                NonNullCount: col.Length - col.NullCount,
                NullCount: col.NullCount,
                EstimatedBytes: estimatedBytes
            ));
        }

        return new InfoResult
        {
            RowCount = df.RowCount,
            ColumnCount = df.ColumnCount,
            Columns = columns
        };
    }

    private static long EstimateMemory(IColumn col)
    {
        if (col is StringColumn sc)
        {
            long bytes = 0;
            for (int i = 0; i < sc.Length; i++)
                bytes += sc[i]?.Length * 2 ?? 0; // UTF-16
            return bytes + col.Length * IntPtr.Size; // reference array overhead
        }

        // Struct columns: element size * length + null bitmap
        int elementSize = col.DataType == typeof(bool) ? 1 : Marshal.SizeOf(col.DataType);
        return (long)elementSize * col.Length + (col.Length + 7) / 8;
    }

    private static string FormatType(Type type)
    {
        if (type == typeof(int)) return "int32";
        if (type == typeof(long)) return "int64";
        if (type == typeof(float)) return "float32";
        if (type == typeof(double)) return "float64";
        if (type == typeof(bool)) return "bool";
        if (type == typeof(DateTime)) return "datetime";
        if (type == typeof(string)) return "string";
        return type.Name;
    }
}
