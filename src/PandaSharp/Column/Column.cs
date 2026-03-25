using PandaSharp.Storage;

namespace PandaSharp.Column;

/// <summary>
/// Strongly-typed, Arrow-backed columnar data storage.
/// </summary>
public class Column<T> : IColumn where T : struct
{
    internal readonly ArrowBackedBuffer<T> Buffer;
    internal readonly NullBitmask Nulls;

    public string Name { get; }
    public Type DataType => typeof(T);
    public int Length => Buffer.Length;

    /// <summary>
    /// Get the raw backing byte[] for zero-copy native pinning.
    /// Returns null if the column is a sliced view.
    /// </summary>
    internal byte[]? RawBytes => Buffer.RawBytes;
    public int NullCount => Nulls.NullCount;

    public Column(string name, T[] values)
    {
        Name = name;
        Buffer = new ArrowBackedBuffer<T>(values);
        Nulls = NullBitmask.AllValid;
    }

    /// <summary>
    /// Fast constructor: wraps pre-computed result bytes directly. No copy.
    /// Used by SIMD arithmetic to avoid the T[]→byte[] copy.
    /// </summary>
    internal static Column<T> WrapResult(string name, byte[] bytes, int length)
    {
        return new Column<T>(name, ArrowBackedBuffer<T>.WrapBytes(bytes, length), NullBitmask.AllValid);
    }

    public static Column<T> FromNullable(string name, T?[] values)
    {
        var nonNull = new T[values.Length];
        for (int i = 0; i < values.Length; i++)
            nonNull[i] = values[i].GetValueOrDefault();
        var buffer = new ArrowBackedBuffer<T>(nonNull);
        var nulls = NullBitmask.FromNullables(values);
        return new Column<T>(name, buffer, nulls);
    }

    internal Column(string name, ArrowBackedBuffer<T> buffer, NullBitmask nulls)
    {
        Name = name;
        Buffer = buffer;
        Nulls = nulls;
    }

    public bool IsNull(int index) => Nulls.IsNull(index);

    public T? this[int index] => Nulls.IsNull(index) ? null : Buffer[index];

    public ReadOnlySpan<T> Values => Buffer.Span;

    public object? GetObject(int index) => Nulls.IsNull(index) ? null : Buffer[index];

    public IColumn Slice(int offset, int length) =>
        new Column<T>(Name, Buffer.Slice(offset, length), Nulls.Slice(offset, length));

    public IColumn Clone(string? newName = null) =>
        new Column<T>(newName ?? Name, Buffer.DeepCopy(), Nulls.DeepCopy());

    /// <summary>Shallow rename — shares buffers. Safe because columns are immutable.</summary>
    internal IColumn ShallowRename(string newName) => new Column<T>(newName, Buffer, Nulls);

    public IColumn Filter(ReadOnlySpan<bool> mask) =>
        new Column<T>(Name, Buffer.Filter(mask), Nulls.Filter(mask));

    public IColumn TakeRows(ReadOnlySpan<int> indices) =>
        new Column<T>(Name, Buffer.TakeRows(indices), Nulls.TakeRows(indices));

    public int Count() => Length - NullCount;
}
