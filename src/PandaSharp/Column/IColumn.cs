namespace PandaSharp.Column;

/// <summary>
/// Untyped column interface used by DataFrame to hold heterogeneous columns.
/// All column types (Column&lt;T&gt;, StringColumn, CategoricalColumn) implement this interface.
/// </summary>
public interface IColumn
{
    /// <summary>Column name.</summary>
    string Name { get; }
    /// <summary>The .NET type of elements in this column.</summary>
    Type DataType { get; }
    /// <summary>Number of elements (rows).</summary>
    int Length { get; }
    /// <summary>Number of null elements.</summary>
    int NullCount { get; }
    /// <summary>Check if element at index is null.</summary>
    bool IsNull(int index);
    /// <summary>Get element as boxed object (null if null).</summary>
    object? GetObject(int index);
    /// <summary>Zero-copy slice returning a view over a range of rows.</summary>
    IColumn Slice(int offset, int length);
    /// <summary>Clone the column, optionally with a new name.</summary>
    IColumn Clone(string? newName = null);
    /// <summary>Filter rows by boolean mask.</summary>
    IColumn Filter(ReadOnlySpan<bool> mask);
    /// <summary>Take rows at specified indices.</summary>
    IColumn TakeRows(ReadOnlySpan<int> indices);
}
