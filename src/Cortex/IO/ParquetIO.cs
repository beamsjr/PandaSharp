using Parquet;
using Parquet.Data;
using Parquet.Schema;
using Cortex.Column;

namespace Cortex.IO;

public class ParquetReadOptions
{
    /// <summary>Only read these columns (null = all).</summary>
    public string[]? Columns { get; set; }
}

public static class ParquetIO
{
    public static async Task<DataFrame> ReadParquetAsync(string path, ParquetReadOptions? options = null)
    {
        using var stream = File.OpenRead(path);
        return await ReadParquetAsync(stream, options);
    }

    public static async Task<DataFrame> ReadParquetAsync(Stream stream, ParquetReadOptions? options = null)
    {
        using var reader = await ParquetReader.CreateAsync(stream);
        var schema = reader.Schema;
        var pruneSet = options?.Columns is not null ? new HashSet<string>(options.Columns) : null;

        var fields = schema.DataFields
            .Where(f => pruneSet is null || pruneSet.Contains(f.Name))
            .ToArray();

        if (reader.RowGroupCount == 1)
        {
            // Fast path: single row group — read typed arrays directly, zero boxing
            using var rgReader = reader.OpenRowGroupReader(0);
            var columns = new IColumn[fields.Length];
            for (int i = 0; i < fields.Length; i++)
            {
                var field = fields[i];
                var col = await rgReader.ReadColumnAsync(field);
                columns[i] = BuildColumnFromTypedArray(field.Name, field.ClrType, col.Data, field.IsNullable);
            }
            return new DataFrame(columns);
        }

        // Multi row-group path: accumulate typed arrays, then concat
        var builders = new List<Array>[fields.Length];
        for (int i = 0; i < fields.Length; i++)
            builders[i] = new List<Array>();

        int totalRows = 0;
        for (int rg = 0; rg < reader.RowGroupCount; rg++)
        {
            using var rgReader = reader.OpenRowGroupReader(rg);
            for (int i = 0; i < fields.Length; i++)
            {
                var col = await rgReader.ReadColumnAsync(fields[i]);
                builders[i].Add(col.Data);
                if (i == 0) // count rows once per row group, from first field only
                    totalRows += col.Data.Length;
            }
        }

        var result = new IColumn[fields.Length];
        for (int i = 0; i < fields.Length; i++)
        {
            var field = fields[i];
            var merged = ConcatArrays(builders[i], field.ClrType, totalRows);
            result[i] = BuildColumnFromTypedArray(field.Name, field.ClrType, merged, field.IsNullable);
        }

        return new DataFrame(result);
    }

    /// <summary>
    /// Build a column directly from the typed array returned by Parquet.Net — zero boxing.
    /// </summary>
    private static IColumn BuildColumnFromTypedArray(string name, Type clrType, Array data, bool hasNulls)
    {
        // Non-nullable columns: cast directly to T[]
        if (!hasNulls)
        {
            if (clrType == typeof(int) && data is int[] intArr)
                return new Column<int>(name, intArr);
            if (clrType == typeof(long) && data is long[] longArr)
                return new Column<long>(name, longArr);
            if (clrType == typeof(double) && data is double[] dblArr)
                return new Column<double>(name, dblArr);
            if (clrType == typeof(float) && data is float[] fltArr)
                return new Column<float>(name, fltArr);
            if (clrType == typeof(bool) && data is bool[] boolArr)
                return new Column<bool>(name, boolArr);
            if (clrType == typeof(DateTime) && data is DateTime[] dtArr)
                return new Column<DateTime>(name, dtArr);
        }

        // Nullable columns: Parquet.Net returns T?[] for nullable fields
        if ((clrType == typeof(int) || clrType == typeof(int?)) && data is int?[] nullIntArr)
            return Column<int>.FromNullable(name, nullIntArr);
        if ((clrType == typeof(long) || clrType == typeof(long?)) && data is long?[] nullLongArr)
            return Column<long>.FromNullable(name, nullLongArr);
        if ((clrType == typeof(double) || clrType == typeof(double?)) && data is double?[] nullDblArr)
            return Column<double>.FromNullable(name, nullDblArr);
        if ((clrType == typeof(float) || clrType == typeof(float?)) && data is float?[] nullFltArr)
            return Column<float>.FromNullable(name, nullFltArr);
        if ((clrType == typeof(bool) || clrType == typeof(bool?)) && data is bool?[] nullBoolArr)
            return Column<bool>.FromNullable(name, nullBoolArr);
        if ((clrType == typeof(DateTime) || clrType == typeof(DateTime?)) && data is DateTime?[] nullDtArr)
            return Column<DateTime>.FromNullable(name, nullDtArr);

        // Non-nullable numeric stored as object[] (fallback)
        if (data is int[] intArr2 && hasNulls)
            return new Column<int>(name, intArr2);
        if (data is long[] longArr2 && hasNulls)
            return new Column<long>(name, longArr2);
        if (data is double[] dblArr2 && hasNulls)
            return new Column<double>(name, dblArr2);

        // String columns
        if (data is string[] strArr)
            return StringColumn.CreateOwned(name, strArr!);
        if (data is string?[] nullStrArr)
            return StringColumn.CreateOwned(name, nullStrArr);

        // Final fallback: boxing path for unknown types
        var values = new object?[data.Length];
        for (int i = 0; i < data.Length; i++)
        {
            var val = data.GetValue(i);
            values[i] = val is DBNull ? null : val;
        }
        return BuildColumn(name, clrType, values);
    }

    /// <summary>Concatenate multiple typed arrays into one.</summary>
    private static Array ConcatArrays(List<Array> arrays, Type clrType, int totalRows)
    {
        if (clrType == typeof(int)) return ConcatTyped<int>(arrays, totalRows);
        if (clrType == typeof(int?)) return ConcatTyped<int?>(arrays, totalRows);
        if (clrType == typeof(long)) return ConcatTyped<long>(arrays, totalRows);
        if (clrType == typeof(long?)) return ConcatTyped<long?>(arrays, totalRows);
        if (clrType == typeof(double)) return ConcatTyped<double>(arrays, totalRows);
        if (clrType == typeof(double?)) return ConcatTyped<double?>(arrays, totalRows);
        if (clrType == typeof(float)) return ConcatTyped<float>(arrays, totalRows);
        if (clrType == typeof(float?)) return ConcatTyped<float?>(arrays, totalRows);
        if (clrType == typeof(bool)) return ConcatTyped<bool>(arrays, totalRows);
        if (clrType == typeof(bool?)) return ConcatTyped<bool?>(arrays, totalRows);
        if (clrType == typeof(DateTime)) return ConcatTyped<DateTime>(arrays, totalRows);
        if (clrType == typeof(DateTime?)) return ConcatTyped<DateTime?>(arrays, totalRows);
        if (clrType == typeof(string)) return ConcatTyped<string?>(arrays, totalRows);

        // Fallback
        var result = new object?[totalRows];
        int offset = 0;
        foreach (var arr in arrays)
        {
            Array.Copy(arr, 0, result, offset, arr.Length);
            offset += arr.Length;
        }
        return result;
    }

    private static T[] ConcatTyped<T>(List<Array> arrays, int totalRows)
    {
        var result = new T[totalRows];
        int offset = 0;
        foreach (var arr in arrays)
        {
            var typed = (T[])arr;
            Array.Copy(typed, 0, result, offset, typed.Length);
            offset += typed.Length;
        }
        return result;
    }

    public static DataFrame ReadParquet(string path, ParquetReadOptions? options = null)
        => ReadParquetAsync(path, options).GetAwaiter().GetResult();

    public static DataFrame ReadParquet(Stream stream, ParquetReadOptions? options = null)
        => ReadParquetAsync(stream, options).GetAwaiter().GetResult();

    // ===== Hive-style partitioned reads =====

    /// <summary>
    /// Read a Hive-style partitioned Parquet dataset from a directory tree.
    /// Discovers partition columns from directory names (e.g., year=2024/month=01/).
    /// All .parquet files are read and concatenated, with partition columns added.
    /// </summary>
    public static DataFrame ReadPartitioned(string directoryPath, ParquetReadOptions? options = null)
    {
        if (!Directory.Exists(directoryPath))
            throw new DirectoryNotFoundException($"Partition root not found: {directoryPath}");

        var files = DiscoverPartitionedFiles(directoryPath);
        if (files.Count == 0)
            throw new InvalidDataException($"No .parquet files found under: {directoryPath}");

        var frames = new List<DataFrame>();

        foreach (var (filePath, partitions) in files)
        {
            var df = ReadParquet(filePath, options);

            // Add partition columns
            foreach (var (key, value) in partitions)
            {
                if (df.ColumnNames.Contains(key)) continue; // data column takes precedence

                // Try to parse as int, then double, fallback to string
                if (int.TryParse(value, out int intVal))
                {
                    var vals = new int[df.RowCount];
                    Array.Fill(vals, intVal);
                    df = df.AddColumn(new Column<int>(key, vals));
                }
                else if (double.TryParse(value, System.Globalization.CultureInfo.InvariantCulture, out double dblVal))
                {
                    var vals = new double[df.RowCount];
                    Array.Fill(vals, dblVal);
                    df = df.AddColumn(new Column<double>(key, vals));
                }
                else
                {
                    var vals = new string?[df.RowCount];
                    Array.Fill(vals, value);
                    df = df.AddColumn(StringColumn.CreateOwned(key, vals));
                }
            }

            frames.Add(df);
        }

        if (frames.Count == 1)
            return frames[0];

        return Concat.ConcatExtensions.Concat(frames.ToArray());
    }

    /// <summary>
    /// Write a DataFrame as a Hive-style partitioned Parquet dataset.
    /// Creates directories like basePath/col=value/ and writes a .parquet file per partition.
    /// </summary>
    public static void WritePartitioned(DataFrame df, string basePath, params string[] partitionColumns)
    {
        if (partitionColumns.Length == 0)
            throw new ArgumentException("At least one partition column is required.");

        // Group rows by partition key values
        var groups = new Dictionary<string, List<int>>();
        for (int r = 0; r < df.RowCount; r++)
        {
            var parts = new List<string>();
            foreach (var col in partitionColumns)
            {
                var val = df[col].GetObject(r)?.ToString() ?? "__null__";
                parts.Add($"{col}={val}");
            }
            var dirKey = string.Join(Path.DirectorySeparatorChar, parts);
            if (!groups.TryGetValue(dirKey, out var list))
            {
                list = new List<int>();
                groups[dirKey] = list;
            }
            list.Add(r);
        }

        // Non-partition columns
        var dataCols = df.ColumnNames.Where(c => !partitionColumns.Contains(c)).ToArray();

        foreach (var (dirKey, indices) in groups)
        {
            var dirPath = Path.Combine(basePath, dirKey);
            Directory.CreateDirectory(dirPath);

            // Build a DataFrame for this partition (data columns only)
            int[] idx = indices.ToArray();
            var cols = dataCols.Select(name => df[name].TakeRows(idx)).ToList();
            var partDf = new DataFrame(cols);

            var filePath = Path.Combine(dirPath, "part-0.parquet");
            WriteParquet(partDf, filePath);
        }
    }

    /// <summary>
    /// Discover all .parquet files under a directory tree, extracting Hive partition key=value pairs.
    /// </summary>
    private static List<(string FilePath, List<(string Key, string Value)> Partitions)> DiscoverPartitionedFiles(
        string rootPath)
    {
        var result = new List<(string, List<(string, string)>)>();

        foreach (var file in Directory.EnumerateFiles(rootPath, "*.parquet", SearchOption.AllDirectories))
        {
            var relativePath = Path.GetRelativePath(rootPath, file);
            var dirParts = Path.GetDirectoryName(relativePath)?.Split(Path.DirectorySeparatorChar)
                ?? Array.Empty<string>();

            var partitions = new List<(string Key, string Value)>();
            foreach (var part in dirParts)
            {
                var eqIdx = part.IndexOf('=');
                if (eqIdx > 0)
                    partitions.Add((part[..eqIdx], part[(eqIdx + 1)..]));
            }

            result.Add((file, partitions));
        }

        return result;
    }

    public static async Task WriteParquetAsync(DataFrame df, string path)
    {
        using var stream = File.Create(path);
        await WriteParquetAsync(df, stream);
    }

    public static async Task WriteParquetAsync(DataFrame df, Stream stream)
    {
        var fields = new List<DataField>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            fields.Add(new DataField(name, ToParquetType(col.DataType), col.NullCount > 0));
        }
        var schema = new ParquetSchema(fields.ToArray());

        using var writer = await ParquetWriter.CreateAsync(schema, stream);
        using var rgWriter = writer.CreateRowGroup();

        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var field = fields.First(f => f.Name == name);

            if (col.DataType == typeof(int))
                await WriteTypedColumn<int>(rgWriter, field, col);
            else if (col.DataType == typeof(long))
                await WriteTypedColumn<long>(rgWriter, field, col);
            else if (col.DataType == typeof(double))
                await WriteTypedColumn<double>(rgWriter, field, col);
            else if (col.DataType == typeof(float))
                await WriteTypedColumn<float>(rgWriter, field, col);
            else if (col.DataType == typeof(bool))
                await WriteTypedColumn<bool>(rgWriter, field, col);
            else if (col.DataType == typeof(DateTime))
                await WriteTypedColumn<DateTime>(rgWriter, field, col);
            else
                await WriteStringColumn(rgWriter, field, col);
        }
    }

    public static void WriteParquet(DataFrame df, string path)
        => WriteParquetAsync(df, path).GetAwaiter().GetResult();

    private static async Task WriteTypedColumn<T>(ParquetRowGroupWriter rgWriter, DataField field, IColumn col)
        where T : struct
    {
        var typed = (Column<T>)col;
        if (col.NullCount == 0)
        {
            var values = typed.Values.ToArray();
            await rgWriter.WriteColumnAsync(new DataColumn(field, values));
        }
        else
        {
            var values = new T?[col.Length];
            for (int i = 0; i < col.Length; i++)
                values[i] = typed[i];
            await rgWriter.WriteColumnAsync(new DataColumn(field, values));
        }
    }

    private static async Task WriteStringColumn(ParquetRowGroupWriter rgWriter, DataField field, IColumn col)
    {
        var values = new string?[col.Length];
        for (int i = 0; i < col.Length; i++)
            values[i] = col.GetObject(i)?.ToString();
        await rgWriter.WriteColumnAsync(new DataColumn(field, values));
    }

    private static Type ToParquetType(Type type)
    {
        if (type == typeof(int)) return typeof(int);
        if (type == typeof(long)) return typeof(long);
        if (type == typeof(double)) return typeof(double);
        if (type == typeof(float)) return typeof(float);
        if (type == typeof(bool)) return typeof(bool);
        if (type == typeof(DateTime)) return typeof(DateTime);
        return typeof(string);
    }

    private static IColumn BuildColumn(string name, Type clrType, object?[] values)
    {
        if (clrType == typeof(int) || clrType == typeof(int?))
            return BuildTyped<int>(name, values);
        if (clrType == typeof(long) || clrType == typeof(long?))
            return BuildTyped<long>(name, values);
        if (clrType == typeof(double) || clrType == typeof(double?))
            return BuildTyped<double>(name, values);
        if (clrType == typeof(float) || clrType == typeof(float?))
            return BuildTyped<float>(name, values);
        if (clrType == typeof(bool) || clrType == typeof(bool?))
            return BuildTyped<bool>(name, values);
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
