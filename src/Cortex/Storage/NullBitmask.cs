using System.Numerics;

namespace Cortex.Storage;

/// <summary>
/// Tracks null/valid status per element using a compact bitmask.
/// A set bit (1) means the value is valid (present). A cleared bit (0) means null.
/// </summary>
internal class NullBitmask
{
    public static readonly NullBitmask AllValid = new(Array.Empty<byte>(), 0, 0, isAllValid: true);

    private readonly byte[] _bitmap;
    private readonly int _offset;
    private readonly int _length;
    private readonly bool _isAllValid;

    public int NullCount { get; }

    private NullBitmask(byte[] bitmap, int offset, int length, bool isAllValid)
    {
        _bitmap = bitmap;
        _offset = offset;
        _length = length;
        _isAllValid = isAllValid;

        if (isAllValid)
        {
            NullCount = 0;
        }
        else
        {
            NullCount = length - CountValidBits(bitmap, offset, length);
        }
    }

    // Pre-computed NullCount constructor (avoids re-counting)
    private NullBitmask(byte[] bitmap, int offset, int length, bool isAllValid, int nullCount)
    {
        _bitmap = bitmap;
        _offset = offset;
        _length = length;
        _isAllValid = isAllValid;
        NullCount = nullCount;
    }

    /// <summary>
    /// Count valid (set) bits using hardware PopCount when available.
    /// </summary>
    private static int CountValidBits(byte[] bitmap, int offset, int length)
    {
        if (length == 0) return 0;

        // For aligned, full-byte ranges, use PopCount
        if (offset == 0)
        {
            int fullBytes = length / 8;
            int validCount = 0;
            for (int i = 0; i < fullBytes; i++)
                validCount += BitOperations.PopCount(bitmap[i]);

            // Handle remaining bits
            int remaining = length & 7;
            if (remaining > 0 && fullBytes < bitmap.Length)
            {
                byte lastByte = bitmap[fullBytes];
                byte mask = (byte)((1 << remaining) - 1);
                validCount += BitOperations.PopCount((uint)(lastByte & mask));
            }
            return validCount;
        }

        // Fallback for offset slices
        int count = 0;
        for (int i = 0; i < length; i++)
        {
            int pos = offset + i;
            if ((bitmap[pos >> 3] & (1 << (pos & 7))) != 0) count++;
        }
        return count;
    }

    /// <summary>
    /// Creates a bitmask from a nullable array. Single pass: builds bitmap and tracks nulls simultaneously.
    /// </summary>
    public static NullBitmask FromNullables<T>(T?[] values) where T : struct
    {
        int byteCount = (values.Length + 7) / 8;
        var bitmap = new byte[byteCount];
        int nullCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (values[i].HasValue)
                bitmap[i >> 3] |= (byte)(1 << (i & 7));
            else
                nullCount++;
        }

        if (nullCount == 0)
            return new NullBitmask(Array.Empty<byte>(), 0, values.Length, isAllValid: true, 0);

        return new NullBitmask(bitmap, 0, values.Length, isAllValid: false, nullCount);
    }

    /// <summary>
    /// Creates a bitmask from a bool[] indicating which entries are null.
    /// </summary>
    public static NullBitmask FromNullFlags(bool[] isNull, int length)
    {
        int nullCount = 0;
        for (int i = 0; i < length; i++)
            if (isNull[i]) nullCount++;

        if (nullCount == 0)
            return new NullBitmask(Array.Empty<byte>(), 0, length, isAllValid: true, 0);

        int byteCount = (length + 7) / 8;
        var bitmap = new byte[byteCount];

        for (int i = 0; i < length; i++)
        {
            if (!isNull[i])
                bitmap[i >> 3] |= (byte)(1 << (i & 7));
        }

        return new NullBitmask(bitmap, 0, length, isAllValid: false, nullCount);
    }

    /// <summary>Deep copy the bitmask.</summary>
    public NullBitmask DeepCopy()
    {
        if (_isAllValid) return this; // immutable singleton-like, safe to share
        var copy = new byte[_bitmap.Length];
        Array.Copy(_bitmap, copy, _bitmap.Length);
        return new NullBitmask(copy, _offset, _length, false, NullCount);
    }

    public bool IsNull(int index)
    {
        if (_isAllValid) return false;
        int pos = _offset + index;
        return (_bitmap[pos >> 3] & (1 << (pos & 7))) == 0;
    }

    public bool IsValid(int index) => !IsNull(index);

    public NullBitmask Slice(int offset, int length)
    {
        if (_isAllValid)
            return new NullBitmask(Array.Empty<byte>(), 0, length, isAllValid: true, 0);
        if (offset < 0 || length < 0 || offset + length > _length)
            throw new ArgumentOutOfRangeException();
        return new NullBitmask(_bitmap, _offset + offset, length, isAllValid: false);
    }

    public NullBitmask TakeRows(ReadOnlySpan<int> indices)
    {
        if (_isAllValid)
            return new NullBitmask(Array.Empty<byte>(), 0, indices.Length, isAllValid: true, 0);

        int byteCount = (indices.Length + 7) / 8;
        var bitmap = new byte[byteCount];
        int nullCount = 0;
        for (int i = 0; i < indices.Length; i++)
        {
            if (IsValid(indices[i]))
                bitmap[i >> 3] |= (byte)(1 << (i & 7));
            else
                nullCount++;
        }
        return new NullBitmask(bitmap, 0, indices.Length, isAllValid: false, nullCount);
    }

    public NullBitmask Filter(ReadOnlySpan<bool> mask)
    {
        if (_isAllValid)
        {
            int count = 0;
            for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;
            return new NullBitmask(Array.Empty<byte>(), 0, count, isAllValid: true, 0);
        }

        int resultCount = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) resultCount++;

        int byteCount = (resultCount + 7) / 8;
        var bitmap = new byte[byteCount];
        int nullCount = 0;
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            if (mask[i])
            {
                if (IsValid(i))
                    bitmap[j >> 3] |= (byte)(1 << (j & 7));
                else
                    nullCount++;
                j++;
            }
        }
        return new NullBitmask(bitmap, 0, resultCount, isAllValid: false, nullCount);
    }
}
