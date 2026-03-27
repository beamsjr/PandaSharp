using System.Text;
using System.Text.Json;
using Cortex.Column;

namespace Cortex.IO;

/// <summary>
/// Minimal Apache Avro binary format reader.
/// Supports: null, boolean, int, long, float, double, string, bytes.
/// Handles nullable columns via union ["null", "type"].
/// </summary>
public static class AvroReader
{
    private static readonly byte[] AvraMagic = "Obj\x01"u8.ToArray();

    public static DataFrame Read(string path)
    {
        using var stream = new BufferedStream(File.OpenRead(path), 65536);
        return Read(stream);
    }

    public static DataFrame Read(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Read header: magic bytes
        var magic = reader.ReadBytes(4);
        if (magic.Length < 4 || magic[0] != AvraMagic[0] || magic[1] != AvraMagic[1] ||
            magic[2] != AvraMagic[2] || magic[3] != AvraMagic[3])
            throw new InvalidDataException("Not a valid Avro file: bad magic bytes.");

        // 2. Read file metadata (map of string->bytes)
        var metadata = ReadMap(reader);

        // 3. Parse schema from metadata
        if (!metadata.TryGetValue("avro.schema", out var schemaBytes))
            throw new InvalidDataException("Avro file missing schema in metadata.");

        var schemaJson = Encoding.UTF8.GetString(schemaBytes);
        var schema = JsonDocument.Parse(schemaJson);
        var fields = ParseSchema(schema);

        // 4. Read sync marker (16 bytes)
        var syncMarker = reader.ReadBytes(16);

        // 5. Read data blocks
        var columnBuilders = new List<List<object?>>(fields.Count);
        for (int i = 0; i < fields.Count; i++)
            columnBuilders.Add(new List<object?>());

        while (stream.Position < stream.Length)
        {
            long rowCount;
            try
            {
                rowCount = ReadZigZagLong(reader);
            }
            catch (EndOfStreamException)
            {
                break;
            }

            if (rowCount <= 0) break;

            var blockSize = ReadZigZagLong(reader);
            var blockData = reader.ReadBytes((int)blockSize);

            // Parse block records
            using var blockStream = new MemoryStream(blockData);
            using var blockReader = new BinaryReader(blockStream, Encoding.UTF8, leaveOpen: true);

            for (long r = 0; r < rowCount; r++)
            {
                for (int f = 0; f < fields.Count; f++)
                {
                    var field = fields[f];
                    var value = ReadValue(blockReader, field.AvroType, field.IsNullable);
                    columnBuilders[f].Add(value);
                }
            }

            // Read sync marker after block
            var blockSync = reader.ReadBytes(16);
            if (!blockSync.AsSpan().SequenceEqual(syncMarker))
                throw new InvalidDataException("Avro sync marker mismatch.");
        }

        // 6. Build DataFrame columns
        var columns = new IColumn[fields.Count];
        for (int i = 0; i < fields.Count; i++)
        {
            columns[i] = BuildColumn(fields[i], columnBuilders[i]);
        }

        return new DataFrame(columns);
    }

    private static object? ReadValue(BinaryReader reader, string avroType, bool isNullable)
    {
        if (isNullable)
        {
            // Union ["null", "type"] - read index
            var unionIndex = ReadZigZagLong(reader);
            if (unionIndex == 0) return null; // null branch
            // fall through to read actual type
        }

        return avroType switch
        {
            "boolean" => reader.ReadByte() != 0,
            "int" => (int)ReadZigZagLong(reader),
            "long" => ReadZigZagLong(reader),
            "float" => reader.ReadSingle(),
            "double" => reader.ReadDouble(),
            "string" => ReadAvroString(reader),
            "bytes" => ReadAvroBytes(reader),
            _ => throw new NotSupportedException($"Unsupported Avro type: {avroType}")
        };
    }

    private static string ReadAvroString(BinaryReader reader)
    {
        var len = (int)ReadZigZagLong(reader);
        if (len < 0 || len > 100_000_000)
            throw new InvalidDataException($"Avro string length {len} is out of valid range (0..100000000).");
        if (len == 0) return string.Empty;
        var bytes = reader.ReadBytes(len);
        return Encoding.UTF8.GetString(bytes);
    }

    private static byte[] ReadAvroBytes(BinaryReader reader)
    {
        var len = (int)ReadZigZagLong(reader);
        return reader.ReadBytes(len);
    }

    private static long ReadZigZagLong(BinaryReader reader)
    {
        // Variable-length zig-zag encoding
        long result = 0;
        int shift = 0;
        byte b;
        do
        {
            b = reader.ReadByte();
            result |= (long)(b & 0x7F) << shift;
            shift += 7;
        } while ((b & 0x80) != 0);

        // Zig-zag decode
        return (long)((ulong)result >> 1) ^ -(result & 1);
    }

    private static Dictionary<string, byte[]> ReadMap(BinaryReader reader)
    {
        var map = new Dictionary<string, byte[]>();
        while (true)
        {
            var count = ReadZigZagLong(reader);
            if (count == 0) break;
            if (count < 0)
            {
                count = -count;
                ReadZigZagLong(reader); // skip byte size
            }

            for (long i = 0; i < count; i++)
            {
                var key = ReadAvroString(reader);
                var value = ReadAvroBytes(reader);
                map[key] = value;
            }
        }

        return map;
    }

    private record FieldInfo(string Name, string AvroType, bool IsNullable);

    private static List<FieldInfo> ParseSchema(JsonDocument schema)
    {
        var root = schema.RootElement;
        if (root.GetProperty("type").GetString() != "record")
            throw new InvalidDataException("Avro schema must be a record type.");

        var fields = new List<FieldInfo>();
        foreach (var field in root.GetProperty("fields").EnumerateArray())
        {
            var name = field.GetProperty("name").GetString()!;
            var typeElement = field.GetProperty("type");

            if (typeElement.ValueKind == JsonValueKind.String)
            {
                fields.Add(new FieldInfo(name, typeElement.GetString()!, false));
            }
            else if (typeElement.ValueKind == JsonValueKind.Array)
            {
                // Union type - expect ["null", "actualType"]
                var types = new List<string>();
                foreach (var t in typeElement.EnumerateArray())
                    types.Add(t.GetString()!);

                var nonNullType = types.First(t => t != "null");
                fields.Add(new FieldInfo(name, nonNullType, types.Contains("null")));
            }
            else
            {
                throw new InvalidDataException($"Unsupported Avro type element for field '{name}'.");
            }
        }

        return fields;
    }

    private static IColumn BuildColumn(FieldInfo field, List<object?> values)
    {
        int count = values.Count;

        if (field.IsNullable)
        {
            return field.AvroType switch
            {
                "int" => Column<int>.FromNullable(field.Name,
                    values.Select(v => v is null ? (int?)null : Convert.ToInt32(v)).ToArray()),
                "long" => Column<long>.FromNullable(field.Name,
                    values.Select(v => v is null ? (long?)null : Convert.ToInt64(v)).ToArray()),
                "float" => Column<float>.FromNullable(field.Name,
                    values.Select(v => v is null ? (float?)null : Convert.ToSingle(v)).ToArray()),
                "double" => Column<double>.FromNullable(field.Name,
                    values.Select(v => v is null ? (double?)null : Convert.ToDouble(v)).ToArray()),
                "boolean" => Column<bool>.FromNullable(field.Name,
                    values.Select(v => v is null ? (bool?)null : Convert.ToBoolean(v)).ToArray()),
                "string" => new StringColumn(field.Name,
                    values.Select(v => v as string).ToArray()),
                _ => throw new NotSupportedException($"Unsupported Avro type: {field.AvroType}")
            };
        }

        return field.AvroType switch
        {
            "int" => new Column<int>(field.Name,
                values.Select(v => Convert.ToInt32(v!)).ToArray()),
            "long" => new Column<long>(field.Name,
                values.Select(v => Convert.ToInt64(v!)).ToArray()),
            "float" => new Column<float>(field.Name,
                values.Select(v => Convert.ToSingle(v!)).ToArray()),
            "double" => new Column<double>(field.Name,
                values.Select(v => Convert.ToDouble(v!)).ToArray()),
            "boolean" => new Column<bool>(field.Name,
                values.Select(v => Convert.ToBoolean(v!)).ToArray()),
            "string" => new StringColumn(field.Name,
                values.Select(v => v as string).ToArray()),
            _ => throw new NotSupportedException($"Unsupported Avro type: {field.AvroType}")
        };
    }
}
