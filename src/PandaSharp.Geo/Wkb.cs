using System.Buffers.Binary;

namespace PandaSharp.Geo;

/// <summary>
/// Well-Known Binary (WKB) encoding and decoding for geometry types.
/// Follows the OGC Simple Features specification.
/// </summary>
public static class Wkb
{
    private const byte LittleEndian = 1;
    private const int WkbPoint = 1;
    private const int PointSize = 21; // 1 (byte order) + 4 (type) + 8 (x) + 8 (y)

    /// <summary>Encode a GeoPoint as WKB bytes (21 bytes, little-endian).</summary>
    public static byte[] EncodePoint(GeoPoint point)
    {
        var bytes = new byte[PointSize];
        bytes[0] = LittleEndian;
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(1), WkbPoint);
        BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(5), point.Longitude); // X = lon
        BinaryPrimitives.WriteDoubleLittleEndian(bytes.AsSpan(13), point.Latitude); // Y = lat
        return bytes;
    }

    /// <summary>Decode a GeoPoint from WKB bytes.</summary>
    public static GeoPoint DecodePoint(ReadOnlySpan<byte> bytes)
    {
        if (bytes.Length < PointSize)
            throw new ArgumentException($"WKB point requires {PointSize} bytes, got {bytes.Length}.");

        bool littleEndian = bytes[0] == LittleEndian;
        int geomType = littleEndian
            ? BinaryPrimitives.ReadInt32LittleEndian(bytes[1..])
            : BinaryPrimitives.ReadInt32BigEndian(bytes[1..]);

        if (geomType != WkbPoint)
            throw new ArgumentException($"Expected WKB Point (type 1), got type {geomType}.");

        double x, y;
        if (littleEndian)
        {
            x = BinaryPrimitives.ReadDoubleLittleEndian(bytes[5..]);
            y = BinaryPrimitives.ReadDoubleLittleEndian(bytes[13..]);
        }
        else
        {
            x = BinaryPrimitives.ReadDoubleBigEndian(bytes[5..]);
            y = BinaryPrimitives.ReadDoubleBigEndian(bytes[13..]);
        }

        return new GeoPoint(Latitude: y, Longitude: x);
    }

    /// <summary>Encode an array of GeoPoints as a contiguous WKB byte array.</summary>
    public static byte[] EncodePoints(GeoColumn column)
    {
        var bytes = new byte[column.Length * PointSize];
        for (int i = 0; i < column.Length; i++)
        {
            var pointBytes = EncodePoint(column[i]);
            pointBytes.CopyTo(bytes.AsSpan(i * PointSize));
        }
        return bytes;
    }

    /// <summary>Decode a contiguous WKB byte array into a GeoColumn.</summary>
    public static GeoColumn DecodePoints(string name, byte[] bytes)
    {
        int count = bytes.Length / PointSize;
        if (bytes.Length % PointSize != 0)
            throw new ArgumentException($"WKB byte array length ({bytes.Length}) is not a multiple of {PointSize}.");

        var lats = new double[count];
        var lons = new double[count];
        for (int i = 0; i < count; i++)
        {
            var point = DecodePoint(bytes.AsSpan(i * PointSize, PointSize));
            lats[i] = point.Latitude;
            lons[i] = point.Longitude;
        }
        return new GeoColumn(name, lats, lons);
    }

    /// <summary>Decode individual WKB byte arrays (one per row) into a GeoColumn.</summary>
    public static GeoColumn DecodePointArray(string name, byte[][] wkbArray)
    {
        var lats = new double[wkbArray.Length];
        var lons = new double[wkbArray.Length];
        for (int i = 0; i < wkbArray.Length; i++)
        {
            var point = DecodePoint(wkbArray[i]);
            lats[i] = point.Latitude;
            lons[i] = point.Longitude;
        }
        return new GeoColumn(name, lats, lons);
    }
}
