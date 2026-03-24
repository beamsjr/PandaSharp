namespace PandaSharp.Indexing;

/// <summary>
/// Fast scalar access by row/column label. Like pandas df.at[row, col].
/// Returns a single value without constructing a DataFrameRow.
/// </summary>
public class AtAccessor
{
    private readonly DataFrame _df;
    internal AtAccessor(DataFrame df) => _df = df;

    /// <summary>Access a single value by row index and column name.</summary>
    public object? this[int row, string column]
    {
        get
        {
            if ((uint)row >= (uint)_df.RowCount)
                throw new IndexOutOfRangeException($"Row {row} out of range (0..{_df.RowCount - 1}).");
            return _df[column].GetObject(row);
        }
    }
}

/// <summary>
/// Fast scalar access by integer position. Like pandas df.iat[row, col].
/// Returns a single value by pure position — no label lookup.
/// </summary>
public class IAtAccessor
{
    private readonly DataFrame _df;
    internal IAtAccessor(DataFrame df) => _df = df;

    /// <summary>Access a single value by row position and column position.</summary>
    public object? this[int row, int col]
    {
        get
        {
            if ((uint)row >= (uint)_df.RowCount)
                throw new IndexOutOfRangeException($"Row {row} out of range (0..{_df.RowCount - 1}).");
            if ((uint)col >= (uint)_df.ColumnCount)
                throw new IndexOutOfRangeException($"Column {col} out of range (0..{_df.ColumnCount - 1}).");
            return _df[_df.ColumnNames[col]].GetObject(row);
        }
    }
}
