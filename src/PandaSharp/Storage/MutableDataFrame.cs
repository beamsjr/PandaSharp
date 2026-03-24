using PandaSharp.Column;

namespace PandaSharp.Storage;

/// <summary>
/// A mutable view over a DataFrame with Copy-on-Write semantics.
/// Modifications only copy the affected column when it's shared.
/// </summary>
public class MutableDataFrame
{
    private readonly List<IColumn> _columns;
    private readonly Dictionary<string, int> _columnIndex;
    private readonly int[] _refCounts; // track sharing per column

    public int RowCount { get; }
    public int ColumnCount => _columns.Count;
    public IReadOnlyList<string> ColumnNames { get; }

    public MutableDataFrame(DataFrame source)
    {
        _columns = new List<IColumn>();
        _columnIndex = new Dictionary<string, int>();
        var names = new List<string>();

        foreach (var name in source.ColumnNames)
        {
            _columnIndex[name] = _columns.Count;
            _columns.Add(source[name]); // share the column, don't copy
            names.Add(name);
        }

        ColumnNames = names.AsReadOnly();
        RowCount = source.RowCount;
        _refCounts = new int[_columns.Count];
        Array.Fill(_refCounts, 1); // initially all shared with source
    }

    public IColumn this[string name] => _columns[_columnIndex[name]];

    /// <summary>
    /// Set a value at a specific row and column. Uses CoW: only copies the column if shared.
    /// </summary>
    public void SetValue<T>(string columnName, int rowIndex, T? value) where T : struct
    {
        if (!_columnIndex.TryGetValue(columnName, out int colIdx))
            throw new KeyNotFoundException($"Column '{columnName}' not found.");
        if ((uint)rowIndex >= (uint)RowCount)
            throw new IndexOutOfRangeException();

        if (_columns[colIdx] is not Column<T> oldCol)
            throw new InvalidOperationException(
                $"Column '{columnName}' is {_columns[colIdx].DataType.Name}, not {typeof(T).Name}.");

        // Copy-on-Write: only allocate a new column if shared with the source
        if (_refCounts[colIdx] > 0)
        {
            var values = new T?[RowCount];
            for (int i = 0; i < RowCount; i++)
                values[i] = oldCol[i];
            values[rowIndex] = value;
            _columns[colIdx] = Column<T>.FromNullable(columnName, values);
            _refCounts[colIdx] = 0;
            return;
        }

        // Already exclusively owned from a previous SetValue — still need to rebuild
        // the column (Arrow buffers are immutable), but we can reuse the nullable array
        // stored in _ownedBuffers if we tracked it. For now, rebuild minimally.
        var vals = new T?[RowCount];
        for (int i = 0; i < RowCount; i++)
            vals[i] = oldCol[i];
        vals[rowIndex] = value;
        _columns[colIdx] = Column<T>.FromNullable(columnName, vals);
    }

    public void SetStringValue(string columnName, int rowIndex, string? value)
    {
        if (!_columnIndex.TryGetValue(columnName, out int colIdx))
            throw new KeyNotFoundException($"Column '{columnName}' not found.");
        if ((uint)rowIndex >= (uint)RowCount)
            throw new IndexOutOfRangeException();

        var oldCol = (StringColumn)_columns[colIdx];

        // Copy-on-Write: only clone when shared
        if (_refCounts[colIdx] > 0)
        {
            var values = oldCol.GetOwnedCopy();
            values[rowIndex] = value;
            _columns[colIdx] = StringColumn.CreateOwned(columnName, values);
            _refCounts[colIdx] = 0;
            return;
        }

        // Already exclusively owned — mutate the backing array directly
        oldCol.GetValues()[rowIndex] = value;
    }

    /// <summary>
    /// Freeze back to an immutable DataFrame.
    /// </summary>
    public DataFrame ToDataFrame() => new(_columns);
}

public static class MutableDataFrameExtensions
{
    /// <summary>
    /// Convert to a mutable DataFrame with Copy-on-Write semantics.
    /// </summary>
    public static MutableDataFrame ToMutable(this DataFrame df) => new(df);
}
