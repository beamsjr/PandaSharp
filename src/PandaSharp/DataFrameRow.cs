using PandaSharp.Column;

namespace PandaSharp;

/// <summary>
/// Lightweight row view into a DataFrame. No data is copied.
/// </summary>
public readonly struct DataFrameRow
{
    private readonly IReadOnlyList<IColumn> _columns;
    private readonly IReadOnlyDictionary<string, int> _columnIndex;
    private readonly int _rowIndex;

    internal DataFrameRow(IReadOnlyList<IColumn> columns, IReadOnlyDictionary<string, int> columnIndex, int rowIndex)
    {
        _columns = columns;
        _columnIndex = columnIndex;
        _rowIndex = rowIndex;
    }

    public object? this[string columnName]
    {
        get
        {
            if (!_columnIndex.TryGetValue(columnName, out int idx))
                throw new KeyNotFoundException($"Column '{columnName}' not found.");
            return _columns[idx].GetObject(_rowIndex);
        }
    }

    public object? this[int columnIndex] => _columns[columnIndex].GetObject(_rowIndex);

    public T? Get<T>(string columnName) where T : struct
    {
        var col = _columns[_columnIndex[columnName]];
        if (col.IsNull(_rowIndex)) return null;
        return (T)col.GetObject(_rowIndex)!;
    }

    public string? GetString(string columnName) =>
        (string?)_columns[_columnIndex[columnName]].GetObject(_rowIndex);
}
