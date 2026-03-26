namespace PandaSharp.GroupBy;

public static class GroupByExtensions
{
    public static GroupedDataFrame GroupBy(this DataFrame df, params string[] keyColumns)
    {
        if (keyColumns.Length == 0)
            throw new ArgumentException("At least one key column is required.");

        foreach (var key in keyColumns)
        {
            if (!df.ColumnNames.Contains(key))
                throw new KeyNotFoundException(
                    $"GroupBy column '{key}' not found. Available columns: [{string.Join(", ", df.ColumnNames.Select(c => $"'{c}'"))}]");
        }

        return new GroupedDataFrame(df, keyColumns);
    }

    /// <summary>
    /// GroupBy with explicit null handling mode.
    /// Usage: df.GroupBy(new[] { "Category" }, NullGroupingMode.Exclude)
    /// </summary>
    public static GroupedDataFrame GroupBy(this DataFrame df, string[] keyColumns, NullGroupingMode nullMode)
    {
        if (keyColumns.Length == 0)
            throw new ArgumentException("At least one key column is required.");

        foreach (var key in keyColumns)
        {
            if (!df.ColumnNames.Contains(key))
                throw new KeyNotFoundException(
                    $"GroupBy column '{key}' not found. Available columns: [{string.Join(", ", df.ColumnNames.Select(c => $"'{c}'"))}]");
        }

        return new GroupedDataFrame(df, keyColumns, nullMode);
    }
}
