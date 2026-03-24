using PandaSharp.Column;

namespace PandaSharp.Indexing;

/// <summary>
/// Label-based indexing accessor (like pandas .loc).
/// Supports integer position, Range, and boolean mask access.
/// </summary>
public class LocAccessor
{
    private readonly DataFrame _df;

    internal LocAccessor(DataFrame df) => _df = df;

    /// <summary>
    /// Access a single row by position.
    /// </summary>
    public DataFrameRow this[int index] => _df[index];

    /// <summary>
    /// Access rows by Range (e.g., df.Loc[1..3]).
    /// </summary>
    public DataFrame this[Range range]
    {
        get
        {
            var (offset, length) = range.GetOffsetAndLength(_df.RowCount);
            return _df.Slice(offset, length);
        }
    }

    /// <summary>
    /// Access rows by boolean mask.
    /// </summary>
    public DataFrame this[bool[] mask] => _df.Filter(mask);

    /// <summary>
    /// Access rows by boolean column.
    /// </summary>
    public DataFrame this[Column<bool> mask]
    {
        get
        {
            var boolMask = new bool[_df.RowCount];
            for (int i = 0; i < _df.RowCount; i++)
                boolMask[i] = mask[i] ?? false;
            return _df.Filter(boolMask);
        }
    }
}

/// <summary>
/// Position-based indexing accessor (like pandas .iloc).
/// </summary>
public class ILocAccessor
{
    private readonly DataFrame _df;

    internal ILocAccessor(DataFrame df) => _df = df;

    /// <summary>
    /// Access a single row by position.
    /// </summary>
    public DataFrameRow this[int index] => _df[index];

    /// <summary>
    /// Access rows by Range (e.g., df.ILoc[1..3]).
    /// </summary>
    public DataFrame this[Range range]
    {
        get
        {
            var (offset, length) = range.GetOffsetAndLength(_df.RowCount);
            return _df.Slice(offset, length);
        }
    }

    /// <summary>
    /// Access rows by explicit indices.
    /// </summary>
    public DataFrame this[int[] indices]
    {
        get
        {
            return new DataFrame(
                _df.ColumnNames.Select(name => _df[name].TakeRows(indices)));
        }
    }
}
