using Cortex.Column;

namespace Cortex.Index;

/// <summary>
/// Hierarchical multi-level index. Each level is a column of labels.
/// Used for grouped/pivoted DataFrames with composite row identifiers.
/// </summary>
public class MultiIndex : IIndex
{
    private readonly IColumn[] _levels;
    private readonly string[] _names;

    public int Length { get; }
    public int NLevels => _levels.Length;
    public IReadOnlyList<string> Names => _names;

    public MultiIndex(IColumn[] levels, string[]? names = null)
    {
        if (levels.Length == 0)
            throw new ArgumentException("MultiIndex requires at least one level.");

        int length = levels[0].Length;
        for (int i = 1; i < levels.Length; i++)
        {
            if (levels[i].Length != length)
                throw new ArgumentException("All levels must have the same length.");
        }

        _levels = levels;
        _names = names ?? levels.Select((_, i) => $"level_{i}").ToArray();
        Length = length;
    }

    /// <summary>Create from arrays of labels.</summary>
    public static MultiIndex FromArrays(params (string Name, IColumn Labels)[] levels)
    {
        return new MultiIndex(
            levels.Select(l => l.Labels).ToArray(),
            levels.Select(l => l.Name).ToArray()
        );
    }

    /// <summary>Create from tuples — each tuple is one row's multi-level key.</summary>
    public static MultiIndex FromTuples(string[] names, params object?[][] tuples)
    {
        int nLevels = names.Length;
        var builders = new List<object?>[nLevels];
        for (int l = 0; l < nLevels; l++)
            builders[l] = new List<object?>();

        foreach (var tuple in tuples)
        {
            for (int l = 0; l < nLevels; l++)
                builders[l].Add(l < tuple.Length ? tuple[l] : null);
        }

        var levels = new IColumn[nLevels];
        for (int l = 0; l < nLevels; l++)
            levels[l] = new StringColumn(names[l], builders[l].Select(v => v?.ToString()).ToArray());

        return new MultiIndex(levels, names);
    }

    /// <summary>Get the label tuple at a position.</summary>
    public object? GetLabel(int position)
    {
        if (_levels.Length == 1) return _levels[0].GetObject(position);
        var values = new object?[_levels.Length];
        for (int l = 0; l < _levels.Length; l++)
            values[l] = _levels[l].GetObject(position);
        return values;
    }

    /// <summary>Get a specific level's value at a position.</summary>
    public object? GetLevel(int level, int position) => _levels[level].GetObject(position);

    /// <summary>Get the column for a level.</summary>
    public IColumn GetLevelValues(int level) => _levels[level];
    public IColumn GetLevelValues(string name)
    {
        int idx = Array.IndexOf(_names, name);
        if (idx < 0) throw new KeyNotFoundException($"Level '{name}' not found.");
        return _levels[idx];
    }

    /// <summary>Find all row positions where a level matches a value.</summary>
    public int[] GetLocations(int level, object? value)
    {
        var col = _levels[level];
        var indices = new List<int>();
        for (int i = 0; i < Length; i++)
        {
            if (Equals(col.GetObject(i), value))
                indices.Add(i);
        }
        return indices.ToArray();
    }

    public IIndex Slice(int offset, int length)
    {
        var sliced = _levels.Select(l => l.Slice(offset, length)).ToArray();
        return new MultiIndex(sliced, _names);
    }

    public IIndex TakeRows(ReadOnlySpan<int> indices)
    {
        var taken = new IColumn[_levels.Length];
        for (int i = 0; i < _levels.Length; i++)
            taken[i] = _levels[i].TakeRows(indices);
        return new MultiIndex(taken, _names);
    }

    public IIndex Filter(ReadOnlySpan<bool> mask)
    {
        var filtered = new IColumn[_levels.Length];
        for (int i = 0; i < _levels.Length; i++)
            filtered[i] = _levels[i].Filter(mask);
        return new MultiIndex(filtered, _names);
    }
}
