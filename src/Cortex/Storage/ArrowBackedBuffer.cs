using System.Buffers;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Apache.Arrow;

namespace Cortex.Storage;

/// <summary>
/// Zero-copy wrapper over an Apache Arrow buffer providing typed Span access.
/// Uses ArrayPool for intermediate allocations to reduce GC pressure.
/// </summary>
internal class ArrowBackedBuffer<T> where T : struct
{
    private readonly ArrowBuffer _buffer;
    private readonly int _offset;
    private readonly int _length;
    // Cached byte[] reference for fast span creation (avoids ArrowBuffer.Span indirection)
    private readonly byte[]? _rawBytes;

    public int Length => _length;

    public ArrowBackedBuffer(T[] values)
    {
        var bytes = new byte[values.Length * Unsafe.SizeOf<T>()];
        MemoryMarshal.AsBytes(values.AsSpan()).CopyTo(bytes);
        _buffer = new ArrowBuffer(bytes);
        _rawBytes = bytes;
        _offset = 0;
        _length = values.Length;
    }

    /// <summary>
    /// Wrap a pre-computed byte array directly as an Arrow buffer. No copy.
    /// </summary>
    internal static ArrowBackedBuffer<T> WrapBytes(byte[] bytes, int length)
    {
        return new ArrowBackedBuffer<T>(new ArrowBuffer(bytes), 0, length, bytes);
    }

    public ArrowBackedBuffer(ArrowBuffer buffer, int offset, int length)
    {
        _buffer = buffer;
        _offset = offset;
        _length = length;
        _rawBytes = null;
    }

    private ArrowBackedBuffer(ArrowBuffer buffer, int offset, int length, byte[]? rawBytes)
    {
        _buffer = buffer;
        _offset = offset;
        _length = length;
        _rawBytes = rawBytes;
    }

    public ReadOnlySpan<T> Span
    {
        get
        {
            // Fast path: direct byte[] access (no ArrowBuffer.Span overhead)
            if (_rawBytes is not null && _offset == 0)
                return MemoryMarshal.Cast<byte, T>(_rawBytes.AsSpan(0, _length * Unsafe.SizeOf<T>()));
            return MemoryMarshal.Cast<byte, T>(_buffer.Span).Slice(_offset, _length);
        }
    }

    /// <summary>
    /// Get the raw backing byte[] for native pinning (no copy).
    /// Returns null if this is a sliced view without a cached byte[] reference.
    /// </summary>
    internal byte[]? RawBytes => _offset == 0 ? _rawBytes : null;

    public T this[int index]
    {
        get
        {
            if ((uint)index >= (uint)_length)
                throw new IndexOutOfRangeException($"Index {index} is out of range for buffer of length {_length}.");
            return Span[index];
        }
    }

    /// <summary>
    /// Deep copy: copies the underlying data into a new independent buffer.
    /// </summary>
    public ArrowBackedBuffer<T> DeepCopy()
    {
        var span = Span;
        var copy = new T[span.Length];
        span.CopyTo(copy);
        return new ArrowBackedBuffer<T>(copy);
    }

    public ArrowBackedBuffer<T> Slice(int offset, int length)
    {
        if (offset < 0 || length < 0 || offset + length > _length)
            throw new ArgumentOutOfRangeException();
        return new ArrowBackedBuffer<T>(_buffer, _offset + offset, length);
    }

    /// <summary>
    /// Creates a new buffer containing only the rows at the specified indices.
    /// Writes directly into the output byte[] — single allocation, no intermediate copy.
    /// </summary>
    public ArrowBackedBuffer<T> TakeRows(ReadOnlySpan<int> indices)
    {
        int count = indices.Length;
        int byteLen = count * Unsafe.SizeOf<T>();
        var bytes = new byte[byteLen];
        var output = MemoryMarshal.Cast<byte, T>(bytes.AsSpan());
        var span = Span;
        for (int i = 0; i < count; i++)
        {
            int idx = indices[i];
            if ((uint)idx >= (uint)_length)
                throw new ArgumentOutOfRangeException(nameof(indices),
                    $"Index {idx} is out of range for buffer of length {_length}.");
            output[i] = span[idx];
        }
        return new ArrowBackedBuffer<T>(new ArrowBuffer(bytes), 0, count);
    }

    /// <summary>
    /// Creates a new buffer containing only the rows where mask is true.
    /// Uses ArrayPool for the intermediate gather to reduce GC pressure.
    /// </summary>
    public ArrowBackedBuffer<T> Filter(ReadOnlySpan<bool> mask)
    {
        if (mask.Length != _length)
            throw new ArgumentException("Mask length must match buffer length.");

        // Count true values
        int count = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) count++;

        if (count == 0)
            return new ArrowBackedBuffer<T>(new ArrowBuffer(System.Array.Empty<byte>()), 0, 0);

        // Write directly into output byte[] — single allocation
        int byteLen = count * Unsafe.SizeOf<T>();
        var bytes = new byte[byteLen];
        var output = MemoryMarshal.Cast<byte, T>(bytes.AsSpan());
        var span = Span;
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) output[j++] = span[i];
        return new ArrowBackedBuffer<T>(new ArrowBuffer(bytes), 0, count);
    }
}
