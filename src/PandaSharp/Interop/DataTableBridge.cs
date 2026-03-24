using System.Data;
using PandaSharp.Column;

namespace PandaSharp.Interop;

public static class DataTableBridge
{
    /// <summary>
    /// Create a DataFrame from a System.Data.DataTable.
    /// </summary>
    public static DataFrame FromDataTable(DataTable table)
    {
        var columns = new List<IColumn>();

        foreach (DataColumn dc in table.Columns)
        {
            var values = new object?[table.Rows.Count];
            for (int r = 0; r < table.Rows.Count; r++)
            {
                var val = table.Rows[r][dc];
                values[r] = val == DBNull.Value ? null : val;
            }
            columns.Add(BuildColumn(dc.ColumnName, dc.DataType, values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Convert a DataFrame to a System.Data.DataTable.
    /// </summary>
    public static DataTable ToDataTable(this DataFrame df, string tableName = "DataFrame")
    {
        var table = new DataTable(tableName);

        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var type = col.DataType == typeof(string) ? typeof(string) : col.DataType;
            table.Columns.Add(name, type);
        }

        for (int r = 0; r < df.RowCount; r++)
        {
            var row = table.NewRow();
            foreach (var name in df.ColumnNames)
            {
                var val = df[name].GetObject(r);
                row[name] = val ?? DBNull.Value;
            }
            table.Rows.Add(row);
        }

        return table;
    }

    /// <summary>
    /// Create a DataFrame from a DbDataReader (streaming).
    /// </summary>
    public static DataFrame FromDataReader(IDataReader reader)
    {
        var columnNames = new string[reader.FieldCount];
        var columnTypes = new Type[reader.FieldCount];
        var data = new List<object?[]>();

        for (int c = 0; c < reader.FieldCount; c++)
        {
            columnNames[c] = reader.GetName(c);
            columnTypes[c] = reader.GetFieldType(c);
        }

        while (reader.Read())
        {
            var row = new object?[reader.FieldCount];
            for (int c = 0; c < reader.FieldCount; c++)
                row[c] = reader.IsDBNull(c) ? null : reader.GetValue(c);
            data.Add(row);
        }

        var columns = new List<IColumn>();
        for (int c = 0; c < columnNames.Length; c++)
        {
            var values = data.Select(r => r[c]).ToArray();
            columns.Add(BuildColumn(columnNames[c], columnTypes[c], values));
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
        if (type == typeof(DateTime)) return BuildTyped<DateTime>(name, values);
        if (type == typeof(decimal))
        {
            var typed = new double?[values.Length];
            for (int i = 0; i < values.Length; i++)
                typed[i] = values[i] is null ? null : Convert.ToDouble(values[i]);
            return Column<double>.FromNullable(name, typed);
        }
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
