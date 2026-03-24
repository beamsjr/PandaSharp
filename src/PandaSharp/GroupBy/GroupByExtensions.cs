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
                throw new KeyNotFoundException($"Column '{key}' not found.");
        }

        return new GroupedDataFrame(df, keyColumns);
    }
}
