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
    /// Vectorized string operations accessor. Reuses cached dict encoding.
    /// </summary>
    public StringAccessor Str => new(this, _cachedDict);

    internal DictEncoding? _cachedDict;

    /// <summary>Cache dict encoding for reuse across multiple Str operations.</summary>
    internal void CacheDictEncoding(DictEncoding dict) => _cachedDict = dict;

    /// <summary>
    /// Get dictionary-encoded group IDs and unique values.
    /// Each row maps to an int code (0..K-1) where K is the number of unique values.
    /// Reuses cached encoding from Str operations if available.
    /// </summary>
    public (int[] Codes, string[] Uniques) GetDictCodes()
    {
        var cached = _cachedDict;
        if (cached is not null)
            return (cached.Codes, cached.Uniques);
        var dict = DictEncoding.Encode(this);
        // Benign race: if another thread computed simultaneously, use whichever wins
        var winner = Interlocked.CompareExchange(ref _cachedDict, dict, null) ?? dict;
        return (winner.Codes, winner.Uniques);
    }

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

    /// <summary>
    /// Internal factory for arrays known to have no nulls (e.g., dict-encoded transforms).
    /// Skips the O(N) null counting pass.
    /// </summary>
    internal static StringColumn CreateOwnedNoNulls(string name, string?[] values)
    {
        return new StringColumn(name, values, nullCount: 0);
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

    /// <summary>Convert all string values to uppercase.</summary>
    public StringColumn ToUpper()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _values[i]?.ToUpperInvariant();
        return CreateOwned(Name, result);
    }

    /// <summary>Convert all string values to lowercase.</summary>
    public StringColumn ToLower()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _values[i]?.ToLowerInvariant();
        return CreateOwned(Name, result);
    }

    /// <summary>Trim whitespace from both ends of all string values.</summary>
    public StringColumn Trim()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _values[i]?.Trim();
        return CreateOwned(Name, result);
    }

    /// <summary>Extract a substring from each string value.</summary>
    public StringColumn Substring(int startIndex, int length)
    {
        if (startIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(startIndex), "startIndex must be non-negative.");
        if (length < 0)
            throw new ArgumentOutOfRangeException(nameof(length), "length must be non-negative.");
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            var s = _values[i];
            if (s is null) { result[i] = null; continue; }
            if (startIndex >= s.Length) { result[i] = ""; continue; }
            int actualLen = Math.Min(length, s.Length - startIndex);
            result[i] = s.Substring(startIndex, actualLen);
        }
        return CreateOwned(Name, result);
    }

    internal string?[] GetValues() => _values;

    /// <summary>Returns a copy of the backing array for mutation.</summary>
    internal string?[] GetOwnedCopy() => (string?[])_values.Clone();

    // ── Packed byte buffer for native bulk operations ──

    private byte[]? _packedBytes;
    private int[]? _packedOffsets;
    private int[]? _packedLengths;

    /// <summary>
    /// Get or build a contiguous ASCII byte buffer containing all strings.
    /// Enables single-call native C string operations without per-string marshalling.
    /// </summary>
    internal (byte[] Bytes, int[] Offsets, int[] Lengths) GetPackedBuffer()
    {
        if (_packedBytes is not null)
            return (_packedBytes, _packedOffsets!, _packedLengths!);

        _packedOffsets = new int[Length];
        _packedLengths = new int[Length];
        int total = 0;
        for (int i = 0; i < Length; i++)
        {
            _packedOffsets[i] = total;
            _packedLengths[i] = _values[i]?.Length ?? 0;
            total += _packedLengths[i];
        }

        _packedBytes = new byte[total];
        for (int i = 0; i < Length; i++)
        {
            if (_values[i] is not null)
                System.Text.Encoding.ASCII.GetBytes(_values[i]!, 0, _packedLengths[i], _packedBytes, _packedOffsets[i]);
        }

        return (_packedBytes, _packedOffsets, _packedLengths);
    }

    /// <summary>
    /// Apply a native byte-level transform (upper/lower) on the packed buffer.
    /// Returns a new StringColumn with transformed strings — one native call for ALL strings.
    /// </summary>
    internal StringColumn ApplyPackedTransform(Action<IntPtr, IntPtr, int> nativeOp)
    {
        var (bytes, offsets, lengths) = GetPackedBuffer();
        var outBytes = new byte[bytes.Length];

        unsafe
        {
            fixed (byte* pIn = bytes, pOut = outBytes)
                nativeOp((IntPtr)pIn, (IntPtr)pOut, bytes.Length);
        }

        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            if (_values[i] is null) { result[i] = null; continue; }
            result[i] = System.Text.Encoding.ASCII.GetString(outBytes, offsets[i], lengths[i]);
        }
        return CreateOwned(Name, result);
    }
}
