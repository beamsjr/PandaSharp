using PandaSharp.Accessors;

namespace PandaSharp.Column;

/// <summary>
/// Column for variable-length string data. Backed by a simple array
/// (Arrow StringArray integration can be added later for IPC/zero-copy).
/// </summary>
public class StringColumn : IColumn
{
    private readonly string?[] _values;

    /// <summary>
    /// Vectorized string operations accessor.
    /// </summary>
    public StringAccessor Str => new(this);

    public string Name { get; }
    public Type DataType => typeof(string);
    public int Length => _values.Length;
    public int NullCount { get; }

    public StringColumn(string name, string?[] values)
    {
        Name = name;
        _values = (string?[])values.Clone();
        int nulls = 0;
        for (int i = 0; i < _values.Length; i++)
            if (_values[i] is null) nulls++;
        NullCount = nulls;
    }

    /// <summary>
    /// Internal factory that takes ownership of the array without copying.
    /// Callers must not retain or modify the array after passing it.
    /// </summary>
    internal static StringColumn CreateOwned(string name, string?[] values)
    {
        return new StringColumn(name, values, nullCount: CountNulls(values));
    }

    private StringColumn(string name, string?[] values, int nullCount)
    {
        Name = name;
        _values = values;
        NullCount = nullCount;
    }

    private static int CountNulls(string?[] values)
    {
        int nulls = 0;
        for (int i = 0; i < values.Length; i++)
            if (values[i] is null) nulls++;
        return nulls;
    }

    public string? this[int index] => _values[index];

    public bool IsNull(int index) => _values[index] is null;

    public object? GetObject(int index) => _values[index];

    public IColumn Slice(int offset, int length)
    {
        var slice = new string?[length];
        Array.Copy(_values, offset, slice, 0, length);
        return CreateOwned(Name, slice);
    }

    public IColumn Clone(string? newName = null) =>
        CreateOwned(newName ?? Name, (string?[])_values.Clone());

    public IColumn Filter(ReadOnlySpan<bool> mask)
    {
        if (mask.Length != Length)
            throw new ArgumentException("Mask length must match column length.");

        int count = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;

        var result = new string?[count];
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) result[j++] = _values[i];

        return CreateOwned(Name, result);
    }

    public IColumn TakeRows(ReadOnlySpan<int> indices)
    {
        var result = new string?[indices.Length];
        for (int i = 0; i < indices.Length; i++)
            result[i] = _values[indices[i]];
        return CreateOwned(Name, result);
    }

    // -- String-specific operations --

    public bool[] Contains(string substring)
    {
        var result = new bool[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _values[i]?.Contains(substring) ?? false;
        return result;
    }

    public bool[] Eq(string value)
    {
        var result = new bool[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _values[i] == value;
        return result;
    }

    internal string?[] GetValues() => _values;

    /// <summary>Returns a copy of the backing array for mutation.</summary>
    internal string?[] GetOwnedCopy() => (string?[])_values.Clone();
}
