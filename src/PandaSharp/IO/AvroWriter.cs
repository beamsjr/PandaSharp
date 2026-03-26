using System.Security.Cryptography;
using System.Text;
using PandaSharp.Column;

namespace PandaSharp.IO;

/// <summary>
/// Minimal Apache Avro binary format writer.
/// Supports: null, boolean, int, long, float, double, string.
/// Uses union ["null", "type"] for nullable columns.
/// </summary>
public static class AvroWriter
{
    private static readonly byte[] AvroMagic = "Obj\x01"u8.ToArray();

    public static void Write(DataFrame df, string path)
    {
        using var stream = new BufferedStream(File.Create(path), 65536);
        Write(df, stream);
    }

    public static void Write(DataFrame df, Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Write magic
        writer.Write(AvroMagic);

        // 2. Build schema JSON
        var schemaJson = BuildSchema(df);

        // 3. Write metadata map
        var metadata = new Dictionary<string, byte[]>
        {
            ["avro.schema"] = Encoding.UTF8.GetBytes(schemaJson),
            ["avro.codec"] = Encoding.UTF8.GetBytes("null")
        };
        WriteMap(writer, metadata);

        // 4. Write sync marker (16 random bytes)
        var syncMarker = RandomNumberGenerator.GetBytes(16);
        writer.Write(syncMarker);

        // 5. Write data as a single block
        if (df.RowCount > 0)
        {
            // Serialize all records into a block buffer
            using var blockBuffer = new MemoryStream(checked(df.RowCount * df.ColumnCount * 16));
            using var blockWriter = new BinaryWriter(blockBuffer, Encoding.UTF8, leaveOpen: true);

            for (int r = 0; r < df.RowCount; r++)
            {
                for (int c = 0; c < df.ColumnCount; c++)
                {
                    var col = df[df.ColumnNames[c]];
                    WriteValue(blockWriter, col, r);
                }
            }

            blockWriter.Flush();
            var blockData = blockBuffer.ToArray();

            // Write block: count, size, data, sync
            WriteZigZagLong(writer, df.RowCount);
            WriteZigZagLong(writer, blockData.Length);
            writer.Write(blockData);
            writer.Write(syncMarker);
        }

        writer.Flush();
    }

    private static void WriteValue(BinaryWriter writer, IColumn col, int row)
    {
        bool isNull = col.IsNull(row);
        bool isNullable = col.NullCount > 0 || IsNullableColumn(col);

        if (isNullable)
        {
            if (isNull)
            {
                WriteZigZagLong(writer, 0); // null branch index
                return;
            }
            WriteZigZagLong(writer, 1); // non-null branch index
        }

        var value = col.GetObject(row);

        switch (value)
        {
            case bool b:
                writer.Write((byte)(b ? 1 : 0));
                break;
            case int i:
                WriteZigZagLong(writer, i);
                break;
            case long l:
                WriteZigZagLong(writer, l);
                break;
            case float f:
                writer.Write(f);
                break;
            case double d:
                writer.Write(d);
                break;
            case string s:
                var bytes = Encoding.UTF8.GetBytes(s);
                WriteZigZagLong(writer, bytes.Length);
                writer.Write(bytes);
                break;
            default:
                throw new NotSupportedException($"Unsupported value type: {value?.GetType().Name ?? "null"}");
        }
    }

    private static bool IsNullableColumn(IColumn col)
    {
        // Check if column type is StringColumn (always nullable) or has any nulls
        return col is StringColumn;
    }

    private static string BuildSchema(DataFrame df)
    {
        var sb = new StringBuilder(256);
        sb.Append("{\"type\":\"record\",\"name\":\"DataFrame\",\"fields\":[");

        for (int i = 0; i < df.ColumnCount; i++)
        {
            if (i > 0) sb.Append(',');
            var col = df[df.ColumnNames[i]];
            var avroType = GetAvroType(col);
            var isNullable = col.NullCount > 0 || IsNullableColumn(col);

            sb.Append("{\"name\":\"");
            sb.Append(EscapeJsonString(col.Name));
            sb.Append("\",\"type\":");

            if (isNullable)
            {
                sb.Append("[\"null\",\"");
                sb.Append(avroType);
                sb.Append("\"]");
            }
            else
            {
                sb.Append('"');
                sb.Append(avroType);
                sb.Append('"');
            }

            sb.Append('}');
        }

        sb.Append("]}");
        return sb.ToString();
    }

    private static string GetAvroType(IColumn col)
    {
        var dt = col.DataType;
        if (dt == typeof(bool)) return "boolean";
        if (dt == typeof(int)) return "int";
        if (dt == typeof(long)) return "long";
        if (dt == typeof(float)) return "float";
        if (dt == typeof(double)) return "double";
        if (dt == typeof(string)) return "string";
        throw new NotSupportedException($"Unsupported column type: {dt.Name}");
    }

    private static string EscapeJsonString(string s)
    {
        return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
    }

    private static void WriteZigZagLong(BinaryWriter writer, long value)
    {
        // Zig-zag encode
        ulong encoded = (ulong)((value << 1) ^ (value >> 63));

        // Variable-length encode
        while (encoded > 0x7F)
        {
            writer.Write((byte)((encoded & 0x7F) | 0x80));
            encoded >>= 7;
        }
        writer.Write((byte)encoded);
    }

    private static void WriteMap(BinaryWriter writer, Dictionary<string, byte[]> map)
    {
        if (map.Count > 0)
        {
            WriteZigZagLong(writer, map.Count);
            foreach (var (key, value) in map)
            {
                // Write key as Avro string
                var keyBytes = Encoding.UTF8.GetBytes(key);
                WriteZigZagLong(writer, keyBytes.Length);
                writer.Write(keyBytes);

                // Write value as Avro bytes
                WriteZigZagLong(writer, value.Length);
                writer.Write(value);
            }
        }

        // End of map
        WriteZigZagLong(writer, 0);
    }
}
