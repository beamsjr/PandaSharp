using PandaSharp.Column;

namespace PandaSharp.Concat;

public static class ConcatExtensions
{
    /// <summary>
    /// Concatenate DataFrames along rows (axis=0) or columns (axis=1).
    /// For axis=0, missing columns are filled with nulls.
    /// </summary>
    public static DataFrame Concat(params DataFrame[] frames) => Concat(0, frames);

    public static DataFrame Concat(int axis, params DataFrame[] frames)
    {
        if (frames.Length == 0) return new DataFrame();
        if (frames.Length == 1) return frames[0];

        return axis == 0 ? ConcatRows(frames) : ConcatColumns(frames);
    }

    private static DataFrame ConcatRows(DataFrame[] frames)
    {
        // Collect all unique column names in order
        var allColumns = new List<string>();
        var seen = new HashSet<string>();
        foreach (var df in frames)
        {
            foreach (var name in df.ColumnNames)
            {
                if (seen.Add(name)) allColumns.Add(name);
            }
        }

        long totalRowsLong = 0;
        foreach (var f in frames) totalRowsLong += f.RowCount;
        if (totalRowsLong > int.MaxValue)
            throw new OverflowException($"Combined row count ({totalRowsLong}) exceeds int.MaxValue. Cannot concatenate this many rows.");
        int totalRows = (int)totalRowsLong;
        var columns = new List<IColumn>();

        // Pre-build column name sets for O(1) lookup
        var columnSets = frames.Select(df => new HashSet<string>(df.ColumnNames)).ToArray();

        foreach (var colName in allColumns)
        {
            // Find the widest numeric type across all frames for this column
            Type? colType = null;
            for (int f = 0; f < frames.Length; f++)
            {
                if (columnSets[f].Contains(colName))
                {
                    var frameType = frames[f][colName].DataType;
                    colType = colType is null ? frameType : WidenType(colType, frameType);
                }
            }

            var values = new object?[totalRows];
            int offset = 0;
            for (int f = 0; f < frames.Length; f++)
            {
                var df = frames[f];
                if (columnSets[f].Contains(colName))
                {
                    var col = df[colName];
                    for (int i = 0; i < col.Length; i++)
                        values[offset + i] = col.GetObject(i);
                }
                // else: values stay null (missing column)
                offset += df.RowCount;
            }

            columns.Add(BuildColumnFromObjects(colName, colType ?? typeof(string), values));
        }

        return new DataFrame(columns);
    }

    private static DataFrame ConcatColumns(DataFrame[] frames)
    {
        // All frames must have the same number of rows
        int rowCount = frames[0].RowCount;
        foreach (var df in frames)
        {
            if (df.RowCount != rowCount)
                throw new ArgumentException(
                    $"Cannot concat along columns (axis=1): DataFrame at index {Array.IndexOf(frames, df)} has {df.RowCount} rows but first DataFrame has {rowCount} rows. All DataFrames must have the same row count for column-wise concatenation.");
        }

        var columns = new List<IColumn>();
        var seen = new HashSet<string>();
        foreach (var df in frames)
        {
            foreach (var name in df.ColumnNames)
            {
                string outputName = name;
                int suffix = 1;
                while (!seen.Add(outputName))
                    outputName = $"{name}_{suffix++}";

                columns.Add(outputName == name ? df[name] : df[name].Clone(outputName));
            }
        }

        return new DataFrame(columns);
    }

    private static IColumn BuildColumnFromObjects(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
        if (type == typeof(bool)) return BuildTyped<bool>(name, values);
        if (type == typeof(DateTime)) return BuildTyped<DateTime>(name, values);
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, typed);
    }

    private static readonly Dictionary<Type, int> NumericPriority = new()
    {
        [typeof(byte)] = 1,
        [typeof(short)] = 2,
        [typeof(int)] = 3,
        [typeof(long)] = 4,
        [typeof(float)] = 5,
        [typeof(double)] = 6,
    };

    /// <summary>
    /// Return the wider of two numeric types. If both are numeric, pick the higher-priority one.
    /// If either is non-numeric (string, bool, DateTime), return the first type (no widening).
    /// </summary>
    private static Type WidenType(Type a, Type b)
    {
        if (a == b) return a;
        if (NumericPriority.TryGetValue(a, out int pa) && NumericPriority.TryGetValue(b, out int pb))
            return pa >= pb ? a : b;
        return a; // non-numeric or mixed: keep first type
    }
}
