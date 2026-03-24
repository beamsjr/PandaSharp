using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.GroupBy;

/// <summary>
/// Result of a GroupBy operation. Stores group keys and their row indices.
/// </summary>
public class GroupedDataFrame
{
    private readonly DataFrame _source;
    private readonly string[] _keyColumns;
    private readonly Dictionary<GroupKey, List<int>> _groups;

    internal GroupedDataFrame(DataFrame source, string[] keyColumns)
    {
        _source = source;
        _keyColumns = keyColumns;
        _groups = BuildGroups();
    }

    public int GroupCount => _groups.Count;
    public IEnumerable<GroupKey> Keys => _groups.Keys;

    /// <summary>
    /// Gets the DataFrame for a specific group.
    /// </summary>
    public DataFrame GetGroup(GroupKey key)
    {
        if (!_groups.TryGetValue(key, out var indices))
            throw new KeyNotFoundException($"Group '{key}' not found.");
        int[] idx = indices.ToArray();
        return new DataFrame(_source.ColumnNames.Select(name => _source[name].TakeRows(idx)));
    }

    // -- Built-in aggregates that return a single DataFrame --

    public DataFrame Sum() => Aggregate(AggFunc.Sum);
    public DataFrame Mean() => Aggregate(AggFunc.Mean);
    public DataFrame Median() => Aggregate(AggFunc.Median);
    public DataFrame Std() => Aggregate(AggFunc.Std);
    public DataFrame Var() => Aggregate(AggFunc.Var);
    public DataFrame Min() => Aggregate(AggFunc.Min);
    public DataFrame Max() => Aggregate(AggFunc.Max);
    public DataFrame Count() => Aggregate(AggFunc.Count);
    public DataFrame First() => Aggregate(AggFunc.First);
    public DataFrame Last() => Aggregate(AggFunc.Last);

    /// <summary>
    /// Return the nth row from each group.
    /// </summary>
    public DataFrame Nth(int n)
    {
        var indices = new List<int>();
        foreach (var (_, groupIndices) in _groups)
        {
            if (n >= 0 && n < groupIndices.Count)
                indices.Add(groupIndices[n]);
            else if (n < 0 && groupIndices.Count + n >= 0)
                indices.Add(groupIndices[groupIndices.Count + n]);
        }
        int[] idx = indices.ToArray();
        return new DataFrame(_source.ColumnNames.Select(name => _source[name].TakeRows(idx)));
    }

    /// <summary>
    /// Returns a column with cumulative count within each group (0-based).
    /// The result has the same length as the source DataFrame with values ordered by original row position.
    /// </summary>
    public Column.Column<int> Cumcount()
    {
        var result = new int?[_source.RowCount];
        foreach (var (_, indices) in _groups)
        {
            for (int i = 0; i < indices.Count; i++)
                result[indices[i]] = i;
        }
        return Column.Column<int>.FromNullable("cumcount", result);
    }

    /// <summary>
    /// Returns the group number (0-based) for each row.
    /// </summary>
    public Column.Column<int> Ngroup()
    {
        var result = new int[_source.RowCount];
        int groupNum = 0;
        foreach (var (_, indices) in _groups)
        {
            foreach (var idx in indices)
                result[idx] = groupNum;
            groupNum++;
        }
        return new Column.Column<int>("ngroup", result);
    }

    /// <summary>
    /// Named aggregation: apply different functions to different columns.
    /// </summary>
    public DataFrame Agg(Action<AggregationBuilder> configure)
    {
        var builder = new AggregationBuilder();
        configure(builder);
        return ExecuteNamedAgg(builder.Build());
    }

    /// <summary>
    /// Transform: apply a function to each group, returning a same-shaped DataFrame.
    /// The function receives a group DataFrame and must return a DataFrame with the same row count.
    /// </summary>
    public DataFrame Transform(Func<DataFrame, DataFrame> func)
    {
        var nonKeyColumns = _source.ColumnNames.Where(c => !_keyColumns.Contains(c)).ToList();
        var transformedGroups = new List<(List<int> Indices, DataFrame Result)>();

        foreach (var (key, indices) in _groups)
        {
            var groupDf = GetGroup(key);
            var transformed = func(groupDf);

            if (transformed.RowCount != indices.Count)
                throw new InvalidOperationException(
                    $"Transform function must return same number of rows as group. Expected {indices.Count}, got {transformed.RowCount}.");

            transformedGroups.Add((indices, transformed));
        }

        // Build result columns by scattering group results to original positions
        var cols = new List<IColumn>();
        foreach (var keyCol in _keyColumns)
            cols.Add(_source[keyCol]);

        foreach (var colName in nonKeyColumns)
        {
            // Check first group to see if this column exists in the transform output
            if (transformedGroups.Count == 0 || !transformedGroups[0].Result.ColumnNames.Contains(colName))
            {
                cols.Add(_source[colName]); // keep original
                continue;
            }

            // Typed scatter for common types to avoid boxing
            var firstCol = transformedGroups[0].Result[colName];

            if (firstCol is Column.Column<double>)
            {
                var vals = new double[_source.RowCount];
                foreach (var (indices, result) in transformedGroups)
                {
                    var src = (Column.Column<double>)result[colName];
                    var span = src.Buffer.Span;
                    for (int i = 0; i < indices.Count; i++)
                        vals[indices[i]] = span[i];
                }
                cols.Add(new Column.Column<double>(colName, vals));
            }
            else if (firstCol is Column.Column<int>)
            {
                var vals = new int[_source.RowCount];
                foreach (var (indices, result) in transformedGroups)
                {
                    var src = (Column.Column<int>)result[colName];
                    var span = src.Buffer.Span;
                    for (int i = 0; i < indices.Count; i++)
                        vals[indices[i]] = span[i];
                }
                cols.Add(new Column.Column<int>(colName, vals));
            }
            else
            {
                // Fallback: boxing path
                var resultValues = new object?[_source.RowCount];
                foreach (var (indices, result) in transformedGroups)
                {
                    var srcCol = result[colName];
                    for (int i = 0; i < indices.Count; i++)
                        resultValues[indices[i]] = srcCol.GetObject(i);
                }
                cols.Add(BuildColumnFromObjects(colName, firstCol.DataType, resultValues));
            }
        }

        return new DataFrame(cols);
    }

    /// <summary>
    /// Filter: keep only groups where the predicate returns true.
    /// </summary>
    public DataFrame Filter(Func<DataFrame, bool> predicate)
    {
        var keepIndices = new List<int>();
        foreach (var (key, indices) in _groups)
        {
            var groupDf = GetGroup(key);
            if (predicate(groupDf))
                keepIndices.AddRange(indices);
        }
        keepIndices.Sort();
        int[] idx = keepIndices.ToArray();
        return new DataFrame(_source.ColumnNames.Select(name => _source[name].TakeRows(idx)));
    }

    /// <summary>
    /// Apply: run an arbitrary function on each group, concatenate results.
    /// </summary>
    public DataFrame Apply(Func<DataFrame, DataFrame> func)
    {
        var results = new List<DataFrame>();
        foreach (var key in _groups.Keys)
        {
            var groupDf = GetGroup(key);
            var result = func(groupDf);
            results.Add(result);
        }

        if (results.Count == 0)
            return new DataFrame();

        // Concatenate all results vertically
        return ConcatDataFrames(results);
    }

    /// <summary>
    /// Like Sum() but aggregates columns in parallel for better throughput on wide DataFrames.
    /// </summary>
    public DataFrame SumParallel() => AggregateParallel(AggFunc.Sum);
    public DataFrame MeanParallel() => AggregateParallel(AggFunc.Mean);

    private DataFrame AggregateParallel(AggFunc func)
    {
        var nonKeyColumns = _source.ColumnNames
            .Where(c => !_keyColumns.Contains(c) && IsNumericColumn(_source[c]))
            .ToList();

        var groupKeys = _groups.Keys.ToList();
        var columns = new IColumn[nonKeyColumns.Count];

        // Aggregate each column in parallel
        Parallel.For(0, nonKeyColumns.Count, colIdx =>
        {
            var srcColName = nonKeyColumns[colIdx];
            var srcCol = _source[srcColName];
            var aggValues = new object?[groupKeys.Count];

            for (int g = 0; g < groupKeys.Count; g++)
            {
                var indices = _groups[groupKeys[g]];
                aggValues[g] = ComputeAggregate(srcCol, indices, func);
            }

            columns[colIdx] = BuildColumnFromObjects(srcColName,
                GetAggOutputType(srcCol.DataType, func), aggValues);
        });

        // Build key columns (sequential — they're small)
        var result = new List<IColumn>();
        foreach (var keyCol in _keyColumns)
        {
            var srcCol = _source[keyCol];
            var keyValues = groupKeys.Select(k => k[Array.IndexOf(_keyColumns, keyCol)]).ToArray();
            result.Add(BuildColumnFromObjects(keyCol, srcCol.DataType, keyValues));
        }
        result.AddRange(columns);

        return new DataFrame(result);
    }

    // -- Internal implementation --

    private Dictionary<GroupKey, List<int>> BuildGroups()
    {
        var groups = new Dictionary<GroupKey, List<int>>();
        var keyCols = _keyColumns.Select(k => _source[k]).ToArray();

        // Reuse a single array for key extraction, copy only when creating new groups
        var keyBuffer = new object?[keyCols.Length];

        for (int r = 0; r < _source.RowCount; r++)
        {
            for (int k = 0; k < keyCols.Length; k++)
                keyBuffer[k] = keyCols[k].GetObject(r);

            var probeKey = new GroupKey(keyBuffer);
            if (groups.TryGetValue(probeKey, out var list))
            {
                list.Add(r);
            }
            else
            {
                // Only allocate a new key array when we find a new group
                var ownedKey = new GroupKey((object?[])keyBuffer.Clone());
                list = new List<int> { r };
                groups[ownedKey] = list;
            }
        }

        return groups;
    }

    private DataFrame Aggregate(AggFunc func)
    {
        var nonKeyColumns = _source.ColumnNames
            .Where(c => !_keyColumns.Contains(c))
            .ToList();

        // For Count, include all columns. For others, only numeric.
        if (func != AggFunc.Count && func != AggFunc.First && func != AggFunc.Last)
        {
            nonKeyColumns = nonKeyColumns
                .Where(c => IsNumericColumn(_source[c]))
                .ToList();
        }

        return AggregateColumns(nonKeyColumns.Select(c => (c, c, func)).ToList());
    }

    private DataFrame ExecuteNamedAgg(List<(string SourceColumn, string OutputName, AggFunc Func)> specs)
    {
        return AggregateColumns(specs);
    }

    private DataFrame AggregateColumns(List<(string SourceColumn, string OutputName, AggFunc Func)> specs)
    {
        var groupKeys = _groups.Keys.ToList();
        var columns = new List<IColumn>();

        // Key columns
        foreach (var keyCol in _keyColumns)
        {
            var srcCol = _source[keyCol];
            var keyValues = groupKeys.Select(k => k[Array.IndexOf(_keyColumns, keyCol)]).ToArray();
            columns.Add(BuildColumnFromObjects(keyCol, srcCol.DataType, keyValues));
        }

        // Aggregated columns — use typed fast path when possible
        foreach (var (srcColName, outputName, func) in specs)
        {
            var srcCol = _source[srcColName];

            // Typed fast path: avoid boxing for common double/int Sum/Mean
            if (func == AggFunc.Sum && srcCol is Column.Column<double> dColSum)
            {
                var vals = new double[groupKeys.Count];
                var span = dColSum.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double sum = 0;
                    foreach (var idx in _groups[groupKeys[g]])
                        if (!dColSum.Nulls.IsNull(idx)) sum += span[idx];
                    vals[g] = sum;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            if (func == AggFunc.Mean && srcCol is Column.Column<double> dColMean)
            {
                var vals = new double[groupKeys.Count];
                var span = dColMean.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double sum = 0; int count = 0;
                    foreach (var idx in _groups[groupKeys[g]])
                    {
                        if (!dColMean.Nulls.IsNull(idx)) { sum += span[idx]; count++; }
                    }
                    vals[g] = count > 0 ? sum / count : double.NaN;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            if (func == AggFunc.Mean && srcCol is Column.Column<int> iColMean)
            {
                var vals = new double[groupKeys.Count];
                var span = iColMean.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double sum = 0; int count = 0;
                    foreach (var idx in _groups[groupKeys[g]])
                    {
                        if (!iColMean.Nulls.IsNull(idx)) { sum += span[idx]; count++; }
                    }
                    vals[g] = count > 0 ? sum / count : double.NaN;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            if (func == AggFunc.Sum && srcCol is Column.Column<int> iColSum)
            {
                var vals = new double[groupKeys.Count];
                var span = iColSum.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double sum = 0;
                    foreach (var idx in _groups[groupKeys[g]])
                        if (!iColSum.Nulls.IsNull(idx)) sum += span[idx];
                    vals[g] = sum;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            if (func == AggFunc.Count)
            {
                var vals = new int[groupKeys.Count];
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    int count = 0;
                    foreach (var idx in _groups[groupKeys[g]])
                        if (!srcCol.IsNull(idx)) count++;
                    vals[g] = count;
                }
                columns.Add(new Column.Column<int>(outputName, vals));
                continue;
            }

            // Fallback: generic path with boxing
            var aggValues = new object?[groupKeys.Count];
            for (int g = 0; g < groupKeys.Count; g++)
            {
                var indices = _groups[groupKeys[g]];
                aggValues[g] = ComputeAggregate(srcCol, indices, func);
            }
            columns.Add(BuildColumnFromObjects(outputName, GetAggOutputType(srcCol.DataType, func), aggValues));
        }

        return new DataFrame(columns);
    }

    private static object? ComputeAggregate(IColumn col, List<int> indices, AggFunc func)
    {
        if (indices.Count == 0) return null;

        return func switch
        {
            AggFunc.Count => indices.Count(i => !col.IsNull(i)),
            AggFunc.First => col.GetObject(indices[0]),
            AggFunc.Last => col.GetObject(indices[^1]),
            _ => ComputeNumericAggregate(col, indices, func)
        };
    }

    private static object? ComputeNumericAggregate(IColumn col, List<int> indices, AggFunc func)
    {
        var values = new List<double>();
        foreach (var i in indices)
        {
            if (col.IsNull(i)) continue;
            values.Add(Convert.ToDouble(col.GetObject(i)));
        }

        if (values.Count == 0) return null;

        return func switch
        {
            AggFunc.Sum => values.Sum(),
            AggFunc.Mean => values.Average(),
            AggFunc.Median => ComputeMedian(values),
            AggFunc.Std => ComputeStd(values),
            AggFunc.Var => ComputeVar(values),
            AggFunc.Min => values.Min(),
            AggFunc.Max => values.Max(),
            _ => throw new ArgumentException($"Unknown aggregate function: {func}")
        };
    }

    private static double ComputeMedian(List<double> values)
    {
        values.Sort();
        int mid = values.Count / 2;
        return values.Count % 2 == 0 ? (values[mid - 1] + values[mid]) / 2.0 : values[mid];
    }

    private static double ComputeStd(List<double> values)
    {
        if (values.Count <= 1) return 0;
        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Count - 1));
    }

    private static double ComputeVar(List<double> values)
    {
        if (values.Count <= 1) return 0;
        double mean = values.Average();
        return values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1);
    }

    private static Type GetAggOutputType(Type sourceType, AggFunc func)
    {
        return func switch
        {
            AggFunc.Count => typeof(int),
            AggFunc.First or AggFunc.Last => sourceType,
            _ => typeof(double) // all numeric aggregates produce double
        };
    }

    private static bool IsNumericColumn(IColumn col)
    {
        var t = col.DataType;
        return t == typeof(int) || t == typeof(long) || t == typeof(float) || t == typeof(double);
    }

    private static IColumn BuildColumnFromObjects(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int))
            return BuildTypedColumn<int>(name, values);
        if (type == typeof(long))
            return BuildTypedColumn<long>(name, values);
        if (type == typeof(float))
            return BuildTypedColumn<float>(name, values);
        if (type == typeof(double))
            return BuildTypedColumn<double>(name, values);
        if (type == typeof(bool))
            return BuildTypedColumn<bool>(name, values);
        if (type == typeof(DateTime))
            return BuildTypedColumn<DateTime>(name, values);

        // Fallback: store as strings
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTypedColumn<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, typed);
    }

    private static DataFrame ConcatDataFrames(List<DataFrame> frames)
    {
        if (frames.Count == 0) return new DataFrame();
        var columnNames = frames[0].ColumnNames;
        var columns = new List<IColumn>();

        int totalRows = 0;
        foreach (var f in frames) totalRows += f.RowCount;

        foreach (var colName in columnNames)
        {
            var allValues = new List<object?>(totalRows); // pre-sized
            foreach (var frame in frames)
            {
                var col = frame[colName];
                for (int i = 0; i < col.Length; i++)
                    allValues.Add(col.GetObject(i));
            }
            var type = frames[0][colName].DataType;
            columns.Add(BuildColumnFromObjects(colName, type, allValues.ToArray()));
        }

        return new DataFrame(columns);
    }
}
