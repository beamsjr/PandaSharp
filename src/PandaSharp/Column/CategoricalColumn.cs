namespace PandaSharp.Column;

/// <summary>
/// Dictionary-encoded column for low-cardinality data. Stores a dictionary of unique values
/// and integer codes, dramatically reducing memory for repeated strings.
/// </summary>
public class CategoricalColumn : IColumn
{
    private readonly string?[] _dictionary;  // unique values
    private readonly int[] _codes;           // index into dictionary per row (-1 = null)

    public string Name { get; }
    public Type DataType => typeof(string);
    public int Length => _codes.Length;
    public int NullCount { get; }
    public int CategoryCount => _dictionary.Length;
    public IReadOnlyList<string?> Categories => _dictionary;

    public CategoricalColumn(string name, string?[] values)
    {
        Name = name;

        // Build dictionary
        var dict = new Dictionary<string, int>();
        var categories = new List<string?>();
        var codes = new int[values.Length];
        int nullCount = 0;

        for (int i = 0; i < values.Length; i++)
        {
            if (values[i] is null)
            {
                codes[i] = -1;
                nullCount++;
            }
            else
            {
                if (!dict.TryGetValue(values[i]!, out int code))
                {
                    code = categories.Count;
                    dict[values[i]!] = code;
                    categories.Add(values[i]);
                }
                codes[i] = code;
            }
        }

        _dictionary = categories.ToArray();
        _codes = codes;
        NullCount = nullCount;
    }

    private CategoricalColumn(string name, string?[] dictionary, int[] codes, int nullCount)
    {
        Name = name;
        _dictionary = dictionary;
        _codes = codes;
        NullCount = nullCount;
    }

    public string? this[int index] => _codes[index] < 0 ? null : _dictionary[_codes[index]];

    public bool IsNull(int index) => _codes[index] < 0;

    public object? GetObject(int index) => this[index];

    public IColumn Slice(int offset, int length)
    {
        var codes = new int[length];
        Array.Copy(_codes, offset, codes, 0, length);
        int nulls = codes.Count(c => c < 0);
        return new CategoricalColumn(Name, _dictionary, codes, nulls);
    }

    public IColumn Clone(string? newName = null) =>
        new CategoricalColumn(newName ?? Name, _dictionary, (int[])_codes.Clone(), NullCount);

    public IColumn Filter(ReadOnlySpan<bool> mask)
    {
        if (mask.Length != Length) throw new ArgumentException("Mask length must match column length.");
        int count = 0;
        for (int i = 0; i < mask.Length; i++) if (mask[i]) count++;
        var codes = new int[count];
        int j = 0;
        for (int i = 0; i < mask.Length; i++)
            if (mask[i]) codes[j++] = _codes[i];
        int nulls = codes.Count(c => c < 0);
        return new CategoricalColumn(Name, _dictionary, codes, nulls);
    }

    public IColumn TakeRows(ReadOnlySpan<int> indices)
    {
        var codes = new int[indices.Length];
        for (int i = 0; i < indices.Length; i++)
            codes[i] = _codes[indices[i]];
        int nulls = codes.Count(c => c < 0);
        return new CategoricalColumn(Name, _dictionary, codes, nulls);
    }

    /// <summary>
    /// Convert to a regular StringColumn (expands dictionary).
    /// </summary>
    public StringColumn ToStringColumn()
    {
        var values = new string?[Length];
        for (int i = 0; i < Length; i++)
            values[i] = this[i];
        return new StringColumn(Name, values);
    }

    /// <summary>
    /// Estimated memory usage in bytes.
    /// Dictionary: sum of string lengths * 2 (UTF-16) + references
    /// Codes: 4 bytes per row (int32)
    /// </summary>
    public long EstimatedBytes
    {
        get
        {
            long dictBytes = _dictionary.Sum(s => (s?.Length ?? 0) * 2 + IntPtr.Size);
            long codeBytes = _codes.Length * 4L;
            return dictBytes + codeBytes;
        }
    }
}

public static class CategoricalExtensions
{
    /// <summary>
    /// Convert a StringColumn to a CategoricalColumn for memory savings.
    /// </summary>
    public static CategoricalColumn AsCategorical(this StringColumn column) =>
        new(column.Name, column.GetValues());
}
