using System.Buffers.Binary;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using Cortex.ML.Tensors;

namespace Cortex.SafeTensors;

/// <summary>
/// Builds and writes tensors in the HuggingFace SafeTensors binary format.
/// Supports writing multiple named tensors with optional metadata.
/// </summary>
public class SafeTensorWriter
{
    private readonly List<TensorEntry> _entries = [];
    private readonly Dictionary<string, string> _metadata = [];

    /// <summary>
    /// Adds a named tensor to be written. The tensor dtype is inferred from <typeparamref name="T"/>.
    /// </summary>
    public SafeTensorWriter Add<T>(string name, Tensor<T> tensor) where T : struct, INumber<T>
    {
        ArgumentNullException.ThrowIfNull(name);
        ArgumentNullException.ThrowIfNull(tensor);
        if (_entries.Any(e => e.Name == name))
            throw new ArgumentException($"Tensor '{name}' has already been added.", nameof(name));

        var dtype = InferDtype<T>();
        int elementSize = Marshal.SizeOf<T>();
        int byteCount = tensor.Span.Length * elementSize;
        _entries.Add(new TensorEntry(name, dtype, (int[])tensor.Shape.Clone(), tensor, byteCount));
        return this;
    }

    /// <summary>
    /// Adds a metadata key-value pair to be written in the file header.
    /// </summary>
    public SafeTensorWriter AddMetadata(string key, string value)
    {
        ArgumentNullException.ThrowIfNull(key);
        ArgumentNullException.ThrowIfNull(value);
        _metadata[key] = value;
        return this;
    }

    /// <summary>
    /// Writes the SafeTensors file to the specified path.
    /// </summary>
    public void Save(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var fs = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.None,
            bufferSize: 1024 * 1024); // 1MB buffer for sequential writes
        Save(fs);
    }

    /// <summary>
    /// Writes the SafeTensors data to the specified stream.
    /// </summary>
    public void Save(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        if (!BitConverter.IsLittleEndian)
            throw new PlatformNotSupportedException("SafeTensors requires little-endian architecture.");
        // Compute data offsets
        long currentOffset = 0;
        var offsets = new (long Start, long End)[_entries.Count];
        for (int i = 0; i < _entries.Count; i++)
        {
            offsets[i] = (currentOffset, currentOffset + _entries[i].ByteCount);
            currentOffset = offsets[i].End;
        }

        // Build header JSON directly to UTF8 bytes
        var headerBytes = BuildHeaderBytes(offsets);

        // Write header size (8-byte LE uint64)
        Span<byte> headerSizeBytes = stackalloc byte[8];
        BinaryPrimitives.WriteUInt64LittleEndian(headerSizeBytes, (ulong)headerBytes.Length);
        stream.Write(headerSizeBytes);

        // Write header
        stream.Write(headerBytes);

        // Write tensor data directly from spans — no intermediate copy
        foreach (var entry in _entries)
        {
            entry.WriteDataTo(stream);
        }
    }

    // -- Private helpers --

    private byte[] BuildHeaderBytes(ReadOnlySpan<(long Start, long End)> offsets)
    {
        using var ms = new MemoryStream();
        using var writer = new Utf8JsonWriter(ms, new JsonWriterOptions { Indented = false });

        writer.WriteStartObject();

        for (int i = 0; i < _entries.Count; i++)
        {
            var entry = _entries[i];
            var (start, end) = offsets[i];

            writer.WritePropertyName(entry.Name);
            writer.WriteStartObject();
            writer.WriteString("dtype", entry.Dtype);

            writer.WritePropertyName("shape");
            writer.WriteStartArray();
            foreach (var dim in entry.Shape)
                writer.WriteNumberValue(dim);
            writer.WriteEndArray();

            writer.WritePropertyName("data_offsets");
            writer.WriteStartArray();
            writer.WriteNumberValue(start);
            writer.WriteNumberValue(end);
            writer.WriteEndArray();

            writer.WriteEndObject();
        }

        if (_metadata.Count > 0)
        {
            writer.WritePropertyName("__metadata__");
            writer.WriteStartObject();
            foreach (var (key, value) in _metadata)
                writer.WriteString(key, value);
            writer.WriteEndObject();
        }

        writer.WriteEndObject();
        writer.Flush();

        return ms.ToArray();
    }

    private static string InferDtype<T>() where T : struct, INumber<T>
    {
        return typeof(T) switch
        {
            var t when t == typeof(float) => "F32",
            var t when t == typeof(double) => "F64",
            var t when t == typeof(int) => "I32",
            var t when t == typeof(long) => "I64",
            var t when t == typeof(byte) => "U8",
            var t when t == typeof(sbyte) => "I8",
            var t when t == typeof(short) => "I16",
            var t when t == typeof(ushort) => "U16",
            var t when t == typeof(uint) => "U32",
            var t when t == typeof(ulong) => "U64",
            var t when t == typeof(Half) => "F16",
            _ => throw new NotSupportedException($"Type {typeof(T).Name} is not supported for SafeTensors serialization.")
        };
    }

    /// <summary>
    /// Type-erased entry that writes tensor bytes directly to a stream.
    /// </summary>
    private sealed class TensorEntry
    {
        public string Name { get; }
        public string Dtype { get; }
        public int[] Shape { get; }
        public int ByteCount { get; }
        private readonly object _tensor;

        public TensorEntry(string name, string dtype, int[] shape, object tensor, int byteCount)
        {
            Name = name;
            Dtype = dtype;
            Shape = shape;
            _tensor = tensor;
            ByteCount = byteCount;
        }

        public void WriteDataTo(Stream stream)
        {
            // Use dynamic dispatch to get the typed span and write bytes directly
            switch (_tensor)
            {
                case Tensor<float> tf:
                    WriteTensorBytes(stream, tf.Span);
                    break;
                case Tensor<double> td:
                    WriteTensorBytes(stream, td.Span);
                    break;
                case Tensor<int> ti:
                    WriteTensorBytes(stream, ti.Span);
                    break;
                case Tensor<long> tl:
                    WriteTensorBytes(stream, tl.Span);
                    break;
                case Tensor<byte> tb:
                    WriteTensorBytes(stream, tb.Span);
                    break;
                case Tensor<sbyte> tsb:
                    WriteTensorBytes(stream, tsb.Span);
                    break;
                case Tensor<short> ts:
                    WriteTensorBytes(stream, ts.Span);
                    break;
                case Tensor<ushort> tus:
                    WriteTensorBytes(stream, tus.Span);
                    break;
                case Tensor<Half> th:
                    WriteTensorBytes(stream, th.Span);
                    break;
                default:
                    throw new NotSupportedException($"Unsupported tensor type: {_tensor.GetType()}");
            }
        }

        private static void WriteTensorBytes<T>(Stream stream, ReadOnlySpan<T> span) where T : struct
        {
            var bytes = MemoryMarshal.AsBytes(span);
            stream.Write(bytes);
        }
    }
}
