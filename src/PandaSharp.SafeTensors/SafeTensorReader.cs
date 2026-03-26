using System.Buffers.Binary;
using System.IO.MemoryMappedFiles;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Text.Json;
using PandaSharp.ML.Tensors;

namespace PandaSharp.SafeTensors;

/// <summary>
/// Reads tensors from the HuggingFace SafeTensors binary format.
/// Supports memory-mapped file access for efficient loading of large models.
/// </summary>
public sealed class SafeTensorReader : IDisposable
{
    private readonly MemoryMappedFile? _mmf;
    private readonly MemoryMappedViewAccessor? _accessor;
    private readonly byte[]? _data;
    private readonly long _dataOffset;
    private readonly long _totalDataSize;
    private readonly Dictionary<string, TensorInfo> _tensors;
    private readonly Dictionary<string, string> _metadata;
    private bool _disposed;

    // Cached unsafe pointer for fast mmap reads
    private readonly unsafe byte* _mmapPtr;

    private unsafe SafeTensorReader(
        MemoryMappedFile? mmf,
        MemoryMappedViewAccessor? accessor,
        byte[]? data,
        long dataOffset,
        long totalDataSize,
        Dictionary<string, TensorInfo> tensors,
        Dictionary<string, string> metadata)
    {
        _mmf = mmf;
        _accessor = accessor;
        _data = data;
        _dataOffset = dataOffset;
        _totalDataSize = totalDataSize;
        _tensors = tensors;
        _metadata = metadata;

        // Acquire raw pointer for mmap — avoids per-read overhead of ReadArray
        if (_accessor != null)
        {
            byte* ptr = null;
            try
            {
                _accessor.SafeMemoryMappedViewHandle.AcquirePointer(ref ptr);
                _mmapPtr = ptr;
            }
            catch
            {
                if (ptr != null)
                    _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
                _accessor.Dispose();
                _mmf?.Dispose();
                throw;
            }
        }
    }

    /// <summary>
    /// Opens a SafeTensors file from the specified path using memory-mapped I/O for efficient access.
    /// </summary>
    public static SafeTensorReader Open(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("SafeTensors requires little-endian architecture.");
        var fileStream = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read);

        // Read header size (8-byte little-endian uint64)
        Span<byte> headerSizeBytes = stackalloc byte[8];
        fileStream.ReadExactly(headerSizeBytes);
        long headerSize = (long)BinaryPrimitives.ReadUInt64LittleEndian(headerSizeBytes);

        if (headerSize <= 0 || headerSize > fileStream.Length - 8)
            throw new InvalidDataException($"Invalid SafeTensors header size: {headerSize}");

        // Read header JSON
        var headerBytes = new byte[headerSize];
        fileStream.ReadExactly(headerBytes);
        var (tensors, metadata) = ParseHeader(headerBytes);

        long dataOffset = 8 + headerSize;

        // Memory-map the file for efficient tensor data access
        var mmf = MemoryMappedFile.CreateFromFile(
            fileStream, null, 0, MemoryMappedFileAccess.Read, HandleInheritability.None, leaveOpen: false);
        var accessor = mmf.CreateViewAccessor(0, 0, MemoryMappedFileAccess.Read);

        long totalDataSize = fileStream.Length - dataOffset;
        return new SafeTensorReader(mmf, accessor, null, dataOffset, totalDataSize, tensors, metadata);
    }

    /// <summary>
    /// Opens a SafeTensors file from a stream, reading all data into memory.
    /// </summary>
    public static SafeTensorReader Open(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("SafeTensors requires little-endian architecture.");
        Span<byte> headerSizeBytes = stackalloc byte[8];
        stream.ReadExactly(headerSizeBytes);
        long headerSize = (long)BinaryPrimitives.ReadUInt64LittleEndian(headerSizeBytes);

        if (headerSize <= 0 || headerSize > int.MaxValue)
            throw new InvalidDataException($"Invalid SafeTensors header size: {headerSize}");

        var headerBytes = new byte[headerSize];
        stream.ReadExactly(headerBytes);
        var (tensors, metadata) = ParseHeader(headerBytes);

        using var ms = new MemoryStream();
        stream.CopyTo(ms);
        var data = ms.ToArray();

        return new SafeTensorReader(null, null, data, 0, data.Length, tensors, metadata);
    }

    /// <summary>
    /// Returns the names of all tensors in the file.
    /// </summary>
    public string[] GetTensorNames() => [.. _tensors.Keys];

    /// <summary>
    /// Returns a copy of the file-level metadata dictionary.
    /// </summary>
    public Dictionary<string, string> GetMetadata() => new(_metadata);

    /// <summary>
    /// Reads and returns a tensor by name, converting to the requested numeric type.
    /// Float16 and BFloat16 data are automatically promoted to Float32.
    /// </summary>
    public unsafe Tensor<T> GetTensor<T>(string name) where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(name);
        if (!_tensors.TryGetValue(name, out var info))
            throw new KeyNotFoundException($"Tensor '{name}' not found. Available: {string.Join(", ", _tensors.Keys)}");

        long byteStart = info.DataOffsets[0];
        long byteEnd = info.DataOffsets[1];
        if (byteStart < 0 || byteEnd < byteStart || byteEnd > _totalDataSize)
            throw new InvalidDataException(
                $"Tensor '{name}' has invalid data offsets [{byteStart}, {byteEnd}) exceeding data size {_totalDataSize}.");
        int byteLength = (int)(byteEnd - byteStart);

        T[] data;

        // Fast path: direct memory copy from mmap when types match exactly
        if (_mmapPtr != null && CanDirectCopy<T>(info.Dtype))
        {
            int elementSize = Unsafe.SizeOf<T>();
            int elementCount = byteLength / elementSize;
            data = new T[elementCount];

            fixed (T* dest = data)
            {
                Buffer.MemoryCopy(
                    _mmapPtr + _dataOffset + byteStart,
                    dest,
                    byteLength,
                    byteLength);
            }
        }
        else if (_data != null && CanDirectCopy<T>(info.Dtype))
        {
            // Fast path: direct copy from byte array
            int elementSize = Unsafe.SizeOf<T>();
            int elementCount = byteLength / elementSize;
            data = new T[elementCount];
            Buffer.BlockCopy(_data, (int)byteStart, data, 0, byteLength);
        }
        else
        {
            // Slow path: read raw bytes then convert
            var raw = new byte[byteLength];
            ReadBytes(byteStart, raw);
            data = ConvertData<T>(raw, info.Dtype);
        }

        return new Tensor<T>(data, info.Shape);
    }

    /// <summary>
    /// Releases all resources held by this reader.
    /// </summary>
    public unsafe void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        if (_accessor != null && _mmapPtr != null)
            _accessor.SafeMemoryMappedViewHandle.ReleasePointer();
        _accessor?.Dispose();
        _mmf?.Dispose();
        GC.SuppressFinalize(this);
    }

    // -- Private helpers --

    /// <summary>
    /// Checks if we can do a direct memory copy (dtype matches T exactly).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool CanDirectCopy<T>(string dtype) where T : struct
    {
        return dtype switch
        {
            "F32" => typeof(T) == typeof(float),
            "F64" => typeof(T) == typeof(double),
            "I32" => typeof(T) == typeof(int),
            "I64" => typeof(T) == typeof(long),
            "U8" or "BOOL" => typeof(T) == typeof(byte),
            "I8" => typeof(T) == typeof(sbyte),
            "I16" => typeof(T) == typeof(short),
            "U16" => typeof(T) == typeof(ushort),
            "U32" => typeof(T) == typeof(uint),
            "U64" => typeof(T) == typeof(ulong),
            "F16" => typeof(T) == typeof(Half),
            _ => false
        };
    }

    private unsafe void ReadBytes(long offset, byte[] buffer)
    {
        if (_mmapPtr != null)
        {
            fixed (byte* dest = buffer)
            {
                Buffer.MemoryCopy(
                    _mmapPtr + _dataOffset + offset,
                    dest,
                    buffer.Length,
                    buffer.Length);
            }
        }
        else if (_data != null)
        {
            Array.Copy(_data, offset, buffer, 0, buffer.Length);
        }
        else
        {
            throw new InvalidOperationException("No data source available.");
        }
    }

    private static (Dictionary<string, TensorInfo> Tensors, Dictionary<string, string> Metadata) ParseHeader(byte[] headerBytes)
    {
        var tensors = new Dictionary<string, TensorInfo>();
        var metadata = new Dictionary<string, string>();

        using var doc = JsonDocument.Parse(headerBytes);
        var root = doc.RootElement;

        foreach (var prop in root.EnumerateObject())
        {
            if (prop.Name == "__metadata__")
            {
                foreach (var meta in prop.Value.EnumerateObject())
                {
                    metadata[meta.Name] = meta.Value.GetString() ?? "";
                }
                continue;
            }

            var dtype = prop.Value.GetProperty("dtype").GetString()
                ?? throw new InvalidDataException($"Missing dtype for tensor '{prop.Name}'");

            var shapeArr = prop.Value.GetProperty("shape");
            var shape = new int[shapeArr.GetArrayLength()];
            int idx = 0;
            foreach (var dim in shapeArr.EnumerateArray())
                shape[idx++] = dim.GetInt32();

            var offsetsArr = prop.Value.GetProperty("data_offsets");
            var offsets = new long[offsetsArr.GetArrayLength()];
            idx = 0;
            foreach (var off in offsetsArr.EnumerateArray())
                offsets[idx++] = off.GetInt64();

            tensors[prop.Name] = new TensorInfo(dtype, shape, offsets);
        }

        return (tensors, metadata);
    }

    private static T[] ConvertData<T>(byte[] raw, string dtype) where T : struct, INumber<T>
    {
        return dtype switch
        {
            "F16" => ConvertFloat16ToTarget<T>(raw),
            "BF16" => ConvertBFloat16ToTarget<T>(raw),
            "F32" => ReinterpretCast<float, T>(raw),
            "F64" => ReinterpretCast<double, T>(raw),
            "I32" => ReinterpretCast<int, T>(raw),
            "I64" => ReinterpretCast<long, T>(raw),
            "U8" => ReinterpretCast<byte, T>(raw),
            "I8" => ReinterpretCast<sbyte, T>(raw),
            "BOOL" => ReinterpretCast<byte, T>(raw),
            "I16" => ReinterpretCast<short, T>(raw),
            "U16" => ReinterpretCast<ushort, T>(raw),
            "U32" => ReinterpretCast<uint, T>(raw),
            "U64" => ReinterpretCast<ulong, T>(raw),
            _ => throw new NotSupportedException($"Unsupported SafeTensors dtype: {dtype}")
        };
    }

    private static T[] ReinterpretCast<TSrc, T>(byte[] raw)
        where TSrc : struct, INumber<TSrc>
        where T : struct, INumber<T>
    {
        // Fast path: same type, just reinterpret bytes
        if (typeof(TSrc) == typeof(T))
        {
            var result = new T[raw.Length / Unsafe.SizeOf<T>()];
            Buffer.BlockCopy(raw, 0, result, 0, raw.Length);
            return result;
        }

        var src = MemoryMarshal.Cast<byte, TSrc>(raw);
        var output = new T[src.Length];
        for (int i = 0; i < src.Length; i++)
            output[i] = T.CreateChecked(src[i]);
        return output;
    }

    /// <summary>Converts IEEE 754 float16 values to the target type (promoted through float32).</summary>
    private static T[] ConvertFloat16ToTarget<T>(byte[] raw) where T : struct, INumber<T>
    {
        int count = raw.Length / 2;
        var result = new T[count];
        for (int i = 0; i < count; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * 2));
            float value = HalfToFloat(bits);
            result[i] = T.CreateChecked(value);
        }
        return result;
    }

    /// <summary>Converts BFloat16 values to the target type (promoted through float32).</summary>
    private static T[] ConvertBFloat16ToTarget<T>(byte[] raw) where T : struct, INumber<T>
    {
        int count = raw.Length / 2;
        var result = new T[count];
        for (int i = 0; i < count; i++)
        {
            ushort bits = BinaryPrimitives.ReadUInt16LittleEndian(raw.AsSpan(i * 2));
            uint floatBits = (uint)bits << 16;
            float value = BitConverter.Int32BitsToSingle((int)floatBits);
            result[i] = T.CreateChecked(value);
        }
        return result;
    }

    /// <summary>Converts IEEE 754 half-precision (float16) bits to float32.</summary>
    private static float HalfToFloat(ushort halfBits)
    {
        int sign = (halfBits >> 15) & 1;
        int exponent = (halfBits >> 10) & 0x1F;
        int mantissa = halfBits & 0x3FF;

        if (exponent == 0)
        {
            if (mantissa == 0)
                return sign == 1 ? -0.0f : 0.0f;
            float value = mantissa / 1024.0f * MathF.Pow(2, -14);
            return sign == 1 ? -value : value;
        }

        if (exponent == 31)
        {
            if (mantissa == 0)
                return sign == 1 ? float.NegativeInfinity : float.PositiveInfinity;
            return float.NaN;
        }

        int floatExponent = exponent - 15 + 127;
        uint floatBitsVal = ((uint)sign << 31) | ((uint)floatExponent << 23) | ((uint)mantissa << 13);
        return BitConverter.Int32BitsToSingle((int)floatBitsVal);
    }
}

/// <summary>
/// Metadata about a single tensor stored in a SafeTensors file.
/// </summary>
internal record TensorInfo(string Dtype, int[] Shape, long[] DataOffsets);
