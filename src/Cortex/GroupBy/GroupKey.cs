namespace Cortex.GroupBy;

/// <summary>
/// Composite key for groupby operations. Supports single or multi-column keys.
/// </summary>
public readonly struct GroupKey : IEquatable<GroupKey>
{
    private readonly object?[] _values;
    private readonly int _hash;

    public GroupKey(object?[] values)
    {
        _values = values;
        _hash = ComputeHash(values);
    }

    public object? this[int index] => _values[index];
    public int Count => _values.Length;

    public bool Equals(GroupKey other)
    {
        if (_values.Length != other._values.Length) return false;
        for (int i = 0; i < _values.Length; i++)
        {
            if (!Equals(_values[i], other._values[i])) return false;
        }
        return true;
    }

    public override bool Equals(object? obj) => obj is GroupKey other && Equals(other);
    public override int GetHashCode() => _hash;

    private static int ComputeHash(object?[] values)
    {
        var hash = new HashCode();
        foreach (var v in values)
            hash.Add(v);
        return hash.ToHashCode();
    }

    public override string ToString() =>
        _values.Length == 1
            ? _values[0]?.ToString() ?? "null"
            : $"({string.Join(", ", _values.Select(v => v?.ToString() ?? "null"))})";
}
