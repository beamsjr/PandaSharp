using System.Text;
using PandaSharp.Column;

namespace PandaSharp.IO;

/// <summary>
/// Minimal Apache ORC format reader.
/// Reads files written by OrcWriter.
/// Format: Header("ORC") + Stripes + File Footer + Postscript + 1-byte PS length.
/// Supports: boolean, byte, short, int, long, float, double, string.
/// </summary>
public static class OrcReader
{
    private static readonly byte[] OrcMagic = "ORC"u8.ToArray();

    public static DataFrame Read(string path)
    {
        using var stream = new BufferedStream(File.OpenRead(path), 65536);
        return Read(stream);
    }

    internal static DataFrame Read(Stream stream)
    {
        // We need to be able to seek; if not seekable, copy to MemoryStream
        if (!stream.CanSeek)
        {
            using var ms = new MemoryStream();
            stream.CopyTo(ms);
            ms.Position = 0;
            return Read(ms);
        }

        using var reader = new BinaryReader(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Verify header
        var magic = reader.ReadBytes(3);
        if (!magic.AsSpan().SequenceEqual(OrcMagic))
            throw new InvalidDataException("Not a valid ORC file: bad magic bytes.");

        // 2. Read postscript length (last byte of file)
        stream.Seek(-1, SeekOrigin.End);
        var postscriptLen = reader.ReadByte();

        // 3. Read postscript
        stream.Seek(-1 - postscriptLen, SeekOrigin.End);
        var footerLength = reader.ReadInt32();
        var endMagic = reader.ReadBytes(3);
        if (!endMagic.AsSpan().SequenceEqual(OrcMagic))
            throw new InvalidDataException("ORC postscript magic mismatch.");

        // 4. Read file footer
        long footerStart = stream.Length - 1 - postscriptLen - footerLength;
        stream.Seek(footerStart, SeekOrigin.Begin);

        var rowCount = reader.ReadInt32();
        var columnCount = reader.ReadInt32();

        // Read schema
        var columnNames = new string[columnCount];
        var typeCodes = new int[columnCount];
        for (int c = 0; c < columnCount; c++)
        {
            var nameLen = reader.ReadInt32();
            var nameBytes = reader.ReadBytes(nameLen);
            columnNames[c] = Encoding.UTF8.GetString(nameBytes);
            typeCodes[c] = reader.ReadInt32();
        }

        // Read stripe info
        var stripeCount = reader.ReadInt32();
        var stripeInfos = new StripeInfo[stripeCount];
        for (int s = 0; s < stripeCount; s++)
        {
            stripeInfos[s] = new StripeInfo(
                reader.ReadInt64(),  // offset
                reader.ReadInt32(),  // data length
                reader.ReadInt32(),  // footer length
                reader.ReadInt32()   // row count
            );
        }

        // 5. Read stripe data
        // Pre-allocate column arrays
        var columns = new IColumn[columnCount];

        foreach (var stripe in stripeInfos)
        {
            // Read stripe footer first to get stream offsets
            long stripeFooterPos = stripe.Offset + stripe.DataLength;
            stream.Seek(stripeFooterPos, SeekOrigin.Begin);

            var sfColumnCount = reader.ReadInt32();
            var streamInfos = new ColumnStreamInfo[sfColumnCount];
            for (int c = 0; c < sfColumnCount; c++)
            {
                streamInfos[c] = new ColumnStreamInfo(
                    reader.ReadInt32(),  // presence offset
                    reader.ReadInt32(),  // presence length
                    reader.ReadInt32(),  // data offset
                    reader.ReadInt32()   // data length
                );
            }

            // Read each column
            for (int c = 0; c < columnCount; c++)
            {
                var info = streamInfos[c];

                // Read presence stream
                stream.Seek(stripe.Offset + info.PresenceOffset, SeekOrigin.Begin);
                var presenceBytes = reader.ReadBytes(info.PresenceLength);

                // Read data stream
                stream.Seek(stripe.Offset + info.DataOffset, SeekOrigin.Begin);
                var dataBytes = reader.ReadBytes(info.DataLength);

                columns[c] = BuildColumn(columnNames[c], typeCodes[c],
                    presenceBytes, dataBytes, stripe.RowCount);
            }
        }

        return new DataFrame(columns);
    }

    private static IColumn BuildColumn(string name, int typeCode,
        byte[] presenceBytes, byte[] dataBytes, int rowCount)
    {
        if (presenceBytes.Length < rowCount)
            throw new InvalidDataException($"Presence bytes length ({presenceBytes.Length}) is less than row count ({rowCount}).");

        using var dataStream = new MemoryStream(dataBytes);
        using var dataReader = new BinaryReader(dataStream, Encoding.UTF8, leaveOpen: true);

        bool hasNulls = false;
        for (int r = 0; r < rowCount; r++)
        {
            if (presenceBytes[r] == 0) { hasNulls = true; break; }
        }

        return typeCode switch
        {
            0 => BuildBoolColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            1 => BuildByteColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            2 => BuildShortColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            3 => BuildIntColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            4 => BuildLongColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            5 => BuildFloatColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            6 => BuildDoubleColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            7 => BuildStringColumn(name, presenceBytes, dataReader, rowCount),
            8 => BuildDateTimeColumn(name, presenceBytes, dataReader, rowCount, hasNulls),
            _ => throw new NotSupportedException($"Unsupported ORC type code: {typeCode}")
        };
    }

    private static IColumn BuildBoolColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new bool?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var b = reader.ReadByte();
                values[r] = presence[r] == 0 ? null : b != 0;
            }
            return Column<bool>.FromNullable(name, values);
        }

        var arr = new bool[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadByte() != 0;
        return new Column<bool>(name, arr);
    }

    private static IColumn BuildByteColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        // Map ORC byte type to int column (PandaSharp doesn't have Column<byte>)
        if (hasNulls)
        {
            var values = new int?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var b = reader.ReadByte();
                values[r] = presence[r] == 0 ? null : (int)b;
            }
            return Column<int>.FromNullable(name, values);
        }

        var arr = new int[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadByte();
        return new Column<int>(name, arr);
    }

    private static IColumn BuildShortColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        // Map ORC short to int column
        if (hasNulls)
        {
            var values = new int?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var s = reader.ReadInt16();
                values[r] = presence[r] == 0 ? null : (int)s;
            }
            return Column<int>.FromNullable(name, values);
        }

        var arr = new int[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadInt16();
        return new Column<int>(name, arr);
    }

    private static IColumn BuildIntColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new int?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var v = reader.ReadInt32();
                values[r] = presence[r] == 0 ? null : v;
            }
            return Column<int>.FromNullable(name, values);
        }

        var arr = new int[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadInt32();
        return new Column<int>(name, arr);
    }

    private static IColumn BuildLongColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new long?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var v = reader.ReadInt64();
                values[r] = presence[r] == 0 ? null : v;
            }
            return Column<long>.FromNullable(name, values);
        }

        var arr = new long[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadInt64();
        return new Column<long>(name, arr);
    }

    private static IColumn BuildFloatColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new float?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var v = reader.ReadSingle();
                values[r] = presence[r] == 0 ? null : v;
            }
            return Column<float>.FromNullable(name, values);
        }

        var arr = new float[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadSingle();
        return new Column<float>(name, arr);
    }

    private static IColumn BuildDoubleColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new double?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var v = reader.ReadDouble();
                values[r] = presence[r] == 0 ? null : v;
            }
            return Column<double>.FromNullable(name, values);
        }

        var arr = new double[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = reader.ReadDouble();
        return new Column<double>(name, arr);
    }

    private static IColumn BuildDateTimeColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount, bool hasNulls)
    {
        if (hasNulls)
        {
            var values = new DateTime?[rowCount];
            for (int r = 0; r < rowCount; r++)
            {
                var ticks = reader.ReadInt64();
                values[r] = presence[r] == 0 ? null : new DateTime(ticks, DateTimeKind.Utc);
            }
            return Column<DateTime>.FromNullable(name, values);
        }

        var arr = new DateTime[rowCount];
        for (int r = 0; r < rowCount; r++)
            arr[r] = new DateTime(reader.ReadInt64(), DateTimeKind.Utc);
        return new Column<DateTime>(name, arr);
    }

    private static IColumn BuildStringColumn(string name, byte[] presence,
        BinaryReader reader, int rowCount)
    {
        var values = new string?[rowCount];
        for (int r = 0; r < rowCount; r++)
        {
            var len = reader.ReadInt32();
            if (presence[r] == 0)
            {
                values[r] = null;
            }
            else if (len == 0)
            {
                values[r] = string.Empty;
            }
            else
            {
                var bytes = reader.ReadBytes(len);
                values[r] = Encoding.UTF8.GetString(bytes);
            }
        }
        return new StringColumn(name, values);
    }

    private record StripeInfo(long Offset, int DataLength, int FooterLength, int RowCount);
    private record ColumnStreamInfo(int PresenceOffset, int PresenceLength,
        int DataOffset, int DataLength);
}
