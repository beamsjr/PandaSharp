namespace Cortex.Column;

internal static class ColumnRenameHelper
{
    /// <summary>
    /// Rename without deep copy. For use after TakeRows/Filter which already create new data.
    /// Falls back to Clone for unknown column types.
    /// </summary>
    internal static IColumn RenameOrKeep(this IColumn col, string outputName)
    {
        if (col.Name == outputName) return col;

        return col switch
        {
            Column<int> c => new Column<int>(outputName, c.Buffer, c.Nulls),
            Column<long> c => new Column<long>(outputName, c.Buffer, c.Nulls),
            Column<double> c => new Column<double>(outputName, c.Buffer, c.Nulls),
            Column<float> c => new Column<float>(outputName, c.Buffer, c.Nulls),
            Column<bool> c => new Column<bool>(outputName, c.Buffer, c.Nulls),
            Column<DateTime> c => new Column<DateTime>(outputName, c.Buffer, c.Nulls),
            StringColumn sc => new StringColumn(outputName, sc.GetValues()),
            _ => col.Clone(outputName) // fallback
        };
    }
}
