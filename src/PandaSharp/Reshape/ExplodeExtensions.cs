using PandaSharp.Column;

namespace PandaSharp.Reshape;

public static class ExplodeExtensions
{
    /// <summary>
    /// Explode a string column by splitting on a separator, creating one row per element.
    /// Other columns are repeated for each split element.
    /// Usage: df.Explode("Tags", separator: ",")
    /// </summary>
    public static DataFrame Explode(this DataFrame df, string column, string separator = ",")
    {
        var col = df[column];
        if (col is not StringColumn strCol)
            throw new ArgumentException($"Explode currently supports StringColumn only. Column '{column}' is {col.DataType.Name}.");

        // Determine how many rows each original row expands to
        var expansions = new List<(int OrigRow, string? Value)>();
        for (int i = 0; i < df.RowCount; i++)
        {
            var val = strCol[i];
            if (val is null)
            {
                expansions.Add((i, null));
            }
            else
            {
                var parts = val.Split(separator);
                foreach (var part in parts)
                    expansions.Add((i, part.Trim()));
            }
        }

        // Build result columns
        var columns = new List<IColumn>();
        foreach (var name in df.ColumnNames)
        {
            if (name == column)
            {
                // The exploded column
                var values = expansions.Select(e => e.Value).ToArray();
                columns.Add(new StringColumn(name, values));
            }
            else
            {
                // Repeat values for the original row indices
                var srcCol = df[name];
                var values = new object?[expansions.Count];
                for (int i = 0; i < expansions.Count; i++)
                    values[i] = srcCol.GetObject(expansions[i].OrigRow);
                columns.Add(BuildColumn(name, srcCol.DataType, values));
            }
        }

        return new DataFrame(columns);
    }

    private static IColumn BuildColumn(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(bool)) return BuildTyped<bool>(name, values);
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, typed);
    }
}
