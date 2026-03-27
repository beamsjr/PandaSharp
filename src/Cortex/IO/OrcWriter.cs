using System.Text;
using Cortex.Column;

namespace Cortex.IO;

/// <summary>
/// Minimal Apache ORC format writer.
/// Format: Header("ORC") + Stripes + File Footer + Postscript + 1-byte PS length.
/// Supports: boolean, int, long, float, double, string.
/// Uses a simplified single-stripe layout with direct encoding.
/// </summary>
public static class OrcWriter
{
    private static readonly byte[] OrcMagic = "ORC"u8.ToArray();

    public static void Write(DataFrame df, string path)
    {
        using var stream = new BufferedStream(File.Create(path), 65536);
        Write(df, stream);
    }

    internal static void Write(DataFrame df, Stream stream)
    {
        using var writer = new BinaryWriter(stream, Encoding.UTF8, leaveOpen: true);

        // 1. Write header: "ORC"
        writer.Write(OrcMagic);
        long headerEnd = stream.Position;

        // 2. Write stripe data (all columns in a single stripe)
        long stripeStart = stream.Position;
        var columnStreamInfos = new List<ColumnStreamInfo>();

        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            var streamStart = stream.Position;

            // Write presence stream (null bitmap) - 1 byte per row for simplicity
            var presenceStart = stream.Position;
            for (int r = 0; r < df.RowCount; r++)
                writer.Write((byte)(col.IsNull(r) ? 0 : 1));
            var presenceLen = stream.Position - presenceStart;

            // Write data stream
            var dataStart = stream.Position;
            WriteColumnData(writer, col, df.RowCount);
            var dataLen = stream.Position - dataStart;

            columnStreamInfos.Add(new ColumnStreamInfo(
                (int)(presenceStart - stripeStart), (int)presenceLen,
                (int)(dataStart - stripeStart), (int)dataLen));
        }

        long stripeEnd = stream.Position;

        // 3. Write stripe footer (column encoding info)
        var stripeFooterStart = stream.Position;
        WriteStripeFooter(writer, df, columnStreamInfos);
        var stripeFooterLen = (int)(stream.Position - stripeFooterStart);

        // 4. Write file footer
        var fileFooterStart = stream.Position;
        WriteFileFooter(writer, df, headerEnd, (int)(stripeEnd - stripeStart),
            stripeFooterLen, df.RowCount);
        var fileFooterLen = (int)(stream.Position - fileFooterStart);

        // 5. Write postscript
        var postscriptStart = stream.Position;
        WritePostscript(writer, fileFooterLen);
        var postscriptLen = (int)(stream.Position - postscriptStart);

        // 6. Write postscript length (1 byte, last byte of file)
        writer.Write((byte)postscriptLen);

        writer.Flush();
    }

    private static void WriteColumnData(BinaryWriter writer, IColumn col, int rowCount)
    {
        var dt = col.DataType;

        for (int r = 0; r < rowCount; r++)
        {
            if (col.IsNull(r))
            {
                // Write zero-value placeholder for null
                if (dt == typeof(bool)) writer.Write((byte)0);
                else if (dt == typeof(int)) writer.Write(0);
                else if (dt == typeof(long)) writer.Write(0L);
                else if (dt == typeof(float)) writer.Write(0f);
                else if (dt == typeof(double)) writer.Write(0d);
                else if (dt == typeof(DateTime)) writer.Write(0L); // DateTime stored as ticks (long)
                else if (dt == typeof(string)) writer.Write(0); // length = 0
                else if (dt == typeof(short)) writer.Write((short)0);
                else if (dt == typeof(byte)) writer.Write((byte)0);
                continue;
            }

            var value = col.GetObject(r)!;
            switch (value)
            {
                case bool b:
                    writer.Write((byte)(b ? 1 : 0));
                    break;
                case byte by:
                    writer.Write(by);
                    break;
                case short s:
                    writer.Write(s);
                    break;
                case int i:
                    writer.Write(i);
                    break;
                case long l:
                    writer.Write(l);
                    break;
                case float f:
                    writer.Write(f);
                    break;
                case double d:
                    writer.Write(d);
                    break;
                case DateTime dtVal:
                    writer.Write(dtVal.Ticks);
                    break;
                case string str:
                    var bytes = Encoding.UTF8.GetBytes(str);
                    writer.Write(bytes.Length);
                    writer.Write(bytes);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported ORC column type: {value.GetType().Name}");
            }
        }
    }

    private static void WriteStripeFooter(BinaryWriter writer, DataFrame df,
        List<ColumnStreamInfo> columnStreams)
    {
        // Write number of columns
        writer.Write(df.ColumnCount);

        // For each column: presence offset, presence length, data offset, data length
        foreach (var info in columnStreams)
        {
            writer.Write(info.PresenceOffset);
            writer.Write(info.PresenceLength);
            writer.Write(info.DataOffset);
            writer.Write(info.DataLength);
        }
    }

    private static void WriteFileFooter(BinaryWriter writer, DataFrame df,
        long stripeOffset, int stripeDataLen, int stripeFooterLen, int rowCount)
    {
        // Number of rows
        writer.Write(rowCount);

        // Number of columns
        writer.Write(df.ColumnCount);

        // Schema: for each column, write name and type code
        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            var nameBytes = Encoding.UTF8.GetBytes(col.Name);
            writer.Write(nameBytes.Length);
            writer.Write(nameBytes);
            writer.Write(GetOrcTypeCode(col.DataType));
        }

        // Stripe info: offset, data length, footer length, row count
        writer.Write(1); // number of stripes
        writer.Write(stripeOffset);
        writer.Write(stripeDataLen);
        writer.Write(stripeFooterLen);
        writer.Write(rowCount);
    }

    private static void WritePostscript(BinaryWriter writer, int footerLength)
    {
        writer.Write(footerLength);
        // Write ORC magic at end of postscript for validation
        writer.Write(OrcMagic);
    }

    private static int GetOrcTypeCode(Type type)
    {
        if (type == typeof(bool)) return 0;
        if (type == typeof(byte)) return 1;
        if (type == typeof(short)) return 2;
        if (type == typeof(int)) return 3;
        if (type == typeof(long)) return 4;
        if (type == typeof(float)) return 5;
        if (type == typeof(double)) return 6;
        if (type == typeof(string)) return 7;
        if (type == typeof(DateTime)) return 8; // DateTime stored as ticks (long)
        throw new NotSupportedException($"Unsupported ORC type: {type.Name}");
    }

    private record ColumnStreamInfo(int PresenceOffset, int PresenceLength,
        int DataOffset, int DataLength);
}
