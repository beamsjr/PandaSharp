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
    private readonly NullGroupingMode _nullMode;
    private List<int>[]? _cachedGroupEntries;

    internal GroupedDataFrame(DataFrame source, string[] keyColumns,
        NullGroupingMode nullMode = NullGroupingMode.Include)
    {
        _source = source;
        _keyColumns = keyColumns;
        _nullMode = nullMode;
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

    /// <summary>
    /// Build a group ID array: maps each row to its group index (0..K-1).
    /// Enables native C aggregation that processes all groups in a single pass.
    /// </summary>
    public (int[] GroupIds, int NumGroups) BuildGroupIdArray()
    {
        var groupIds = new int[_source.RowCount];
        var groupList = _groups.Keys.ToList();
        var groupMap = new Dictionary<GroupKey, int>(groupList.Count);
        for (int g = 0; g < groupList.Count; g++)
            groupMap[groupList[g]] = g;

        foreach (var (key, indices) in _groups)
        {
            int gid = groupMap[key];
            foreach (var idx in indices)
                groupIds[idx] = gid;
        }
        return (groupIds, groupList.Count);
    }

    /// <summary>
    /// Get the row indices for a specific group without copying any data.
    /// Use with source column access for zero-copy per-group operations.
    /// </summary>
    public List<int> GetGroupIndices(GroupKey key)
    {
        if (!_groups.TryGetValue(key, out var indices))
            throw new KeyNotFoundException($"Group '{key}' not found.");
        return indices;
    }

    /// <summary>
    /// Extract a typed column's values for a group without creating a sub-DataFrame.
    /// Much faster than GetGroup(key).GetColumn&lt;T&gt;(name) for per-group operations.
    /// </summary>
    public double[] GetGroupDoubles(GroupKey key, string columnName)
    {
        var indices = GetGroupIndices(key);
        if (!_source.ColumnNames.Contains(columnName))
            throw new ArgumentException($"Column '{columnName}' does not exist in the DataFrame. Available columns: {string.Join(", ", _source.ColumnNames)}", nameof(columnName));
        var col = _source[columnName];

        if (col is Column.Column<double> dc)
        {
            var span = dc.Buffer.Span;
            var result = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                result[i] = span[indices[i]];
            return result;
        }

        if (col is Column.Column<int> ic)
        {
            var ispan = ic.Buffer.Span;
            var iresult = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                iresult[i] = ispan[indices[i]];
            return iresult;
        }

        if (col is Column.Column<float> fc)
        {
            var fspan = fc.Buffer.Span;
            var fresult = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                fresult[i] = fspan[indices[i]];
            return fresult;
        }

        if (col is Column.Column<long> lc)
        {
            var lspan = lc.Buffer.Span;
            var lresult = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                lresult[i] = lspan[indices[i]];
            return lresult;
        }

        // Fallback with conversion
        var fallback = new double[indices.Count];
        for (int i = 0; i < indices.Count; i++)
            fallback[i] = Convert.ToDouble(col.GetObject(indices[i]) ?? 0);
        return fallback;
    }

    /// <summary>
    /// Apply a function to each group's typed column values without copying full DataFrames.
    /// Returns results scattered back to original row positions.
    /// This is the zero-copy equivalent of pandas groupby().transform().
    /// </summary>
    /// <summary>
    /// TransformDouble with pre-copied data array. Avoids repeated 118MB copies
    /// when multiple transforms operate on the same column.
    /// </summary>
    private List<int>[] GetGroupEntries()
    {
        return _cachedGroupEntries ??= _groups.Values.ToArray();
    }

    /// <summary>
    /// Fused multi-transform: apply multiple functions in a single parallel pass.
    /// Extracts group data once, runs all transforms, scatters all results.
    /// Saves one full gather+scatter pass per additional function.
    /// </summary>
    public (Column.Column<double>, Column.Column<double>) TransformDoubleMulti(
        string columnName, double[] data,
        Func<double[], double[]> func1, Func<double[], double[]> func2)
    {
        int n = _source.RowCount;
        var result1 = new double[n];
        var result2 = new double[n];
        var groupEntries = GetGroupEntries();
        Parallel.For(0, groupEntries.Length, gi =>
        {
            var indices = groupEntries[gi];
            // Single gather
            var groupVals = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                groupVals[i] = data[indices[i]];
            // Both transforms on same extracted data
            var t1 = func1(groupVals);
            var t2 = func2(groupVals);
            // Dual scatter
            for (int i = 0; i < indices.Count; i++)
            {
                int idx = indices[i];
                result1[idx] = t1[i];
                result2[idx] = t2[i];
            }
        });
        return (new Column.Column<double>(columnName, result1),
                new Column.Column<double>(columnName, result2));
    }

    public Column.Column<double> TransformDoublePreCopied(string columnName, double[] data, Func<double[], double[]> func)
    {
        int n = _source.RowCount;
        var result = new double[n];
        var groupEntries = GetGroupEntries();
        Parallel.For(0, groupEntries.Length, gi =>
        {
            var indices = groupEntries[gi];
            var groupVals = new double[indices.Count];
            for (int i = 0; i < indices.Count; i++)
                groupVals[i] = data[indices[i]];
            var transformed = func(groupVals);
            for (int i = 0; i < indices.Count; i++)
                result[indices[i]] = transformed[i];
        });
        return new Column.Column<double>(columnName, result);
    }

    public Column.Column<double> TransformDouble(string columnName, Func<double[], double[]> func)
    {
        var result = new double[_source.RowCount];
        var col = _source[columnName];

        if (col is Column.Column<double> dc)
        {
            var data = dc.Buffer.Span.ToArray(); // single copy for thread-safe parallel access
            var groupEntries = GetGroupEntries();

            // Process groups in parallel — each group writes to non-overlapping indices
            Parallel.For(0, groupEntries.Length, gi =>
            {
                var indices = groupEntries[gi];
                var groupVals = new double[indices.Count];
                for (int i = 0; i < indices.Count; i++)
                    groupVals[i] = data[indices[i]];

                var transformed = func(groupVals);

                for (int i = 0; i < indices.Count; i++)
                    result[indices[i]] = transformed[i];
            });
        }
        else
        {
            foreach (var (_, indices) in _groups)
            {
                var groupVals = new double[indices.Count];
                for (int i = 0; i < indices.Count; i++)
                    groupVals[i] = Convert.ToDouble(col.GetObject(indices[i]) ?? 0);

                var transformed = func(groupVals);
                for (int i = 0; i < indices.Count; i++)
                    result[indices[i]] = transformed[i];
            }
        }

        return new Column.Column<double>(columnName, result);
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
    /// Shift values within each group by the specified number of periods.
    /// Boundary positions are filled with NaN.
    /// </summary>
    public Column.Column<double> Shift(string column, int periods = 1)
    {
        int n = _source.RowCount;
        var result = new double[n];
        Array.Fill(result, double.NaN);

        foreach (var (key, indices) in _groups)
        {
            int groupSize = indices.Count;
            if (groupSize == 0) continue;

            // Gather source values for this group
            var groupVals = GetGroupDoubles(key, column);

            if (periods >= 0)
            {
                // Shift forward: copy [0..groupSize-periods) to positions [periods..groupSize)
                int copyCount = groupSize - Math.Min(periods, groupSize);
                for (int i = 0; i < copyCount; i++)
                    result[indices[i + periods]] = groupVals[i];
            }
            else
            {
                // Shift backward: copy [-periods..groupSize) to positions [0..groupSize+periods)
                int absPeriods = -periods;
                int copyCount = groupSize - Math.Min(absPeriods, groupSize);
                for (int i = 0; i < copyCount; i++)
                    result[indices[i]] = groupVals[i + absPeriods];
            }
        }

        return new Column.Column<double>(column, result);
    }

    /// <summary>
    /// Compute percentage change within each group. Fused shift+division in a single pass.
    /// Uses SIMD Vector&lt;double&gt; for the division loop when possible.
    /// </summary>
    public Column.Column<double> PctChange(string column, int periods = 1)
    {
        int n = _source.RowCount;
        var result = new double[n];
        Array.Fill(result, double.NaN);

        foreach (var (key, indices) in _groups)
        {
            int groupSize = indices.Count;
            if (groupSize == 0) continue;

            var groupVals = GetGroupDoubles(key, column);

            int absPeriods = Math.Abs(periods);
            if (absPeriods >= groupSize) continue; // all NaN

            if (periods >= 0)
            {
                // Fused: pct_change[i] = (current[i] - prev[i-periods]) / prev[i-periods]
                int count = groupSize - absPeriods;

                // SIMD path for the division
                int vecSize = Vector<double>.Count;
                int simdEnd = count - (count % vecSize);

                // Build current and previous arrays for SIMD
                var current = new double[count];
                var previous = new double[count];
                for (int i = 0; i < count; i++)
                {
                    current[i] = groupVals[i + absPeriods];
                    previous[i] = groupVals[i];
                }

                int j = 0;
                for (; j < simdEnd; j += vecSize)
                {
                    var curVec = new Vector<double>(current, j);
                    var prevVec = new Vector<double>(previous, j);
                    var diff = (curVec - prevVec) / prevVec;
                    diff.CopyTo(current, j); // reuse current array for output
                }
                for (; j < count; j++)
                {
                    current[j] = (current[j] - previous[j]) / previous[j];
                }

                // Scatter results back
                for (int i = 0; i < count; i++)
                    result[indices[i + absPeriods]] = current[i];
            }
            else
            {
                // Negative periods: compare current to future values
                int count = groupSize - absPeriods;

                var current = new double[count];
                var future = new double[count];
                for (int i = 0; i < count; i++)
                {
                    current[i] = groupVals[i];
                    future[i] = groupVals[i + absPeriods];
                }

                int vecSize = Vector<double>.Count;
                int simdEnd = count - (count % vecSize);
                int j = 0;
                for (; j < simdEnd; j += vecSize)
                {
                    var curVec = new Vector<double>(current, j);
                    var futVec = new Vector<double>(future, j);
                    var diff = (curVec - futVec) / futVec;
                    diff.CopyTo(current, j);
                }
                for (; j < count; j++)
                {
                    current[j] = (current[j] - future[j]) / future[j];
                }

                for (int i = 0; i < count; i++)
                    result[indices[i]] = current[i];
            }
        }

        return new Column.Column<double>(column, result);
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
    /// Multi-column aggregation with tuple syntax.
    /// Usage: grouped.Agg(("Salary", AggFunc.Sum), ("Age", AggFunc.Mean), ("Name", AggFunc.Count))
    /// </summary>
    public DataFrame Agg(params (string Column, AggFunc Func)[] aggregations)
    {
        if (aggregations.Length == 0)
            throw new ArgumentException("At least one aggregation is required.");

        var specs = new List<(string SourceColumn, string OutputName, AggFunc Func)>();
        foreach (var (column, func) in aggregations)
        {
            if (!_source.ColumnNames.Contains(column))
                throw new ArgumentException(
                    $"Column '{column}' not found in DataFrame. Available columns: [{string.Join(", ", _source.ColumnNames.Select(c => $"'{c}'"))}]");
            string outputName = $"{column}_{func.ToString().ToLower()}";
            specs.Add((column, outputName, func));
        }
        return ExecuteNamedAgg(specs);
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
        // Fast path: single string key column — avoid all boxing
        if (_keyColumns.Length == 1 && _source[_keyColumns[0]] is Column.StringColumn sc)
            return BuildStringGroups(sc);

        // Fast path: single int key column
        if (_keyColumns.Length == 1 && _source[_keyColumns[0]] is Column.Column<int> ic)
            return BuildIntGroups(ic);

        // Fast path: 2 string key columns — dict-encode both, composite int key
        if (_keyColumns.Length == 2 &&
            _source[_keyColumns[0]] is Column.StringColumn sc2a &&
            _source[_keyColumns[1]] is Column.StringColumn sc2b)
            return Build2StringGroups(sc2a, sc2b);

        var groups = new Dictionary<GroupKey, List<int>>();
        var keyCols = _keyColumns.Select(k => _source[k]).ToArray();
        var keyBuffer = new object?[keyCols.Length];

        for (int r = 0; r < _source.RowCount; r++)
        {
            bool hasNull = false;
            for (int k = 0; k < keyCols.Length; k++)
            {
                keyBuffer[k] = keyCols[k].GetObject(r);
                if (keyBuffer[k] is null) hasNull = true;
            }

            // Skip rows with null keys when Exclude mode is active
            if (hasNull && _nullMode == NullGroupingMode.Exclude)
                continue;

            var probeKey = new GroupKey(keyBuffer);
            if (groups.TryGetValue(probeKey, out var list))
            {
                list.Add(r);
            }
            else
            {
                var ownedKey = new GroupKey((object?[])keyBuffer.Clone());
                list = new List<int> { r };
                groups[ownedKey] = list;
            }
        }

        return groups;
    }

    private Dictionary<GroupKey, List<int>> BuildStringGroups(Column.StringColumn col)
    {
        // Fast path: use cached dict codes (int lookups instead of string hashing)
        if (col._cachedDict is { } dict)
        {
            var codes = dict.Codes;
            var uniques = dict.Uniques;
            int nGroups = uniques.Length;
            // Pre-allocate lists with estimated size
            var groupLists = new List<int>[nGroups];
            int avgSize = Math.Max(1, _source.RowCount / Math.Max(nGroups, 1));
            for (int g = 0; g < nGroups; g++)
                groupLists[g] = new List<int>(avgSize + avgSize / 4);
            List<int>? dictNullGroup = null;
            for (int r = 0; r < _source.RowCount; r++)
            {
                if (codes[r] < 0)
                {
                    // null value: code is -1
                    if (_nullMode == NullGroupingMode.Exclude)
                        continue;
                    dictNullGroup ??= new List<int>();
                    dictNullGroup.Add(r);
                }
                else
                {
                    groupLists[codes[r]].Add(r);
                }
            }
            var result = new Dictionary<GroupKey, List<int>>(nGroups + (dictNullGroup is not null ? 1 : 0));
            for (int g = 0; g < nGroups; g++)
                result[new GroupKey(new object?[] { uniques[g] })] = groupLists[g];
            if (dictNullGroup is not null)
                result[new GroupKey(new object?[] { null })] = dictNullGroup;
            return result;
        }

        var vals = col.GetValues();
        var stringMap = new Dictionary<string, List<int>>(_source.RowCount / 100, StringComparer.Ordinal);
        List<int>? nullGroup = null;

        for (int r = 0; r < _source.RowCount; r++)
        {
            if (vals[r] is null)
            {
                if (_nullMode == NullGroupingMode.Exclude)
                    continue;
                nullGroup ??= new List<int>();
                nullGroup.Add(r);
                continue;
            }
            var key = vals[r]!;
            if (!stringMap.TryGetValue(key, out var list))
            {
                list = new List<int>();
                stringMap[key] = list;
            }
            list.Add(r);
        }

        var groups = new Dictionary<GroupKey, List<int>>(stringMap.Count + (nullGroup is not null ? 1 : 0));
        foreach (var (key, list) in stringMap)
            groups[new GroupKey(new object?[] { key })] = list;
        if (nullGroup is not null)
            groups[new GroupKey(new object?[] { null })] = nullGroup;
        return groups;
    }

    private Dictionary<GroupKey, List<int>> Build2StringGroups(Column.StringColumn col1, Column.StringColumn col2)
    {
        // Dict-encode both columns → composite int key (code1, code2)
        var dict1 = DictEncoding.Encode(col1);
        var dict2 = DictEncoding.Encode(col2);
        var codes1 = dict1.Codes;
        var codes2 = dict2.Codes;
        // Use (nUniques + 1) to reserve slot 0 for null; remap -1 → 0, valid codes → code+1
        int n1Plus = dict1.Uniques.Length + 1;
        int n2Plus = dict2.Uniques.Length + 1;

        var compositeMap = new Dictionary<long, List<int>>(_source.RowCount / 100);
        for (int r = 0; r < _source.RowCount; r++)
        {
            bool hasNull = codes1[r] < 0 || codes2[r] < 0;
            if (hasNull && _nullMode == NullGroupingMode.Exclude)
                continue;

            // Remap: null (-1) → 0, valid code c → c + 1
            long mapped1 = codes1[r] + 1;
            long mapped2 = codes2[r] + 1;
            long key = mapped1 * n2Plus + mapped2;
            if (!compositeMap.TryGetValue(key, out var list))
            {
                list = new List<int>();
                compositeMap[key] = list;
            }
            list.Add(r);
        }

        var groups = new Dictionary<GroupKey, List<int>>(compositeMap.Count);
        foreach (var (compositeKey, list) in compositeMap)
        {
            int m1 = (int)(compositeKey / n2Plus);
            int m2 = (int)(compositeKey % n2Plus);
            string? v1 = m1 == 0 ? null : dict1.Uniques[m1 - 1];
            string? v2 = m2 == 0 ? null : dict2.Uniques[m2 - 1];
            groups[new GroupKey(new object?[] { v1, v2 })] = list;
        }
        return groups;
    }

    private Dictionary<GroupKey, List<int>> BuildIntGroups(Column.Column<int> col)
    {
        var span = col.Buffer.Span;
        var intMap = new Dictionary<int, List<int>>();
        List<int>? nullGroup = null;

        for (int r = 0; r < _source.RowCount; r++)
        {
            if (col.Nulls.IsNull(r))
            {
                if (_nullMode == NullGroupingMode.Exclude)
                    continue;
                nullGroup ??= new List<int>();
                nullGroup.Add(r);
                continue;
            }
            var key = span[r];
            if (!intMap.TryGetValue(key, out var list))
            {
                list = new List<int>();
                intMap[key] = list;
            }
            list.Add(r);
        }

        var groups = new Dictionary<GroupKey, List<int>>(intMap.Count + (nullGroup is not null ? 1 : 0));
        foreach (var (key, list) in intMap)
            groups[new GroupKey(new object?[] { key })] = list;
        if (nullGroup is not null)
            groups[new GroupKey(new object?[] { null })] = nullGroup;
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

        // Key columns — typed fast path for single string key
        if (_keyColumns.Length == 1 && _source[_keyColumns[0]].DataType == typeof(string))
        {
            var keyVals = new string?[groupKeys.Count];
            for (int g = 0; g < groupKeys.Count; g++)
                keyVals[g] = groupKeys[g][0] as string;
            columns.Add(StringColumn.CreateOwned(_keyColumns[0], keyVals));
        }
        else
        {
            foreach (var keyCol in _keyColumns)
            {
                var srcCol = _source[keyCol];
                int keyIdx = Array.IndexOf(_keyColumns, keyCol);
                if (srcCol.DataType == typeof(string))
                {
                    var keyVals = new string?[groupKeys.Count];
                    for (int g = 0; g < groupKeys.Count; g++)
                        keyVals[g] = groupKeys[g][keyIdx] as string;
                    columns.Add(StringColumn.CreateOwned(keyCol, keyVals));
                }
                else
                {
                    var keyValues = groupKeys.Select(k => k[keyIdx]).ToArray();
                    columns.Add(BuildColumnFromObjects(keyCol, srcCol.DataType, keyValues));
                }
            }
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
                        if (!dColSum.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) sum += span[idx];
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
                        if (!dColMean.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) { sum += span[idx]; count++; }
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

            // Typed fast path: Min on double
            if (func == AggFunc.Min && srcCol is Column.Column<double> dColMin)
            {
                var vals = new double[groupKeys.Count];
                var span = dColMin.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double min = double.MaxValue;
                    bool found = false;
                    foreach (var idx in _groups[groupKeys[g]])
                    {
                        if (!dColMin.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) { if (span[idx] < min) min = span[idx]; found = true; }
                    }
                    vals[g] = found ? min : double.NaN;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            // Typed fast path: Max on double
            if (func == AggFunc.Max && srcCol is Column.Column<double> dColMax)
            {
                var vals = new double[groupKeys.Count];
                var span = dColMax.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    double max = double.MinValue;
                    bool found = false;
                    foreach (var idx in _groups[groupKeys[g]])
                    {
                        if (!dColMax.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) { if (span[idx] > max) max = span[idx]; found = true; }
                    }
                    vals[g] = found ? max : double.NaN;
                }
                columns.Add(new Column.Column<double>(outputName, vals));
                continue;
            }

            // Typed fast path: Std on double
            if (func == AggFunc.Std && srcCol is Column.Column<double> dColStd)
            {
                var vals = new double[groupKeys.Count];
                var span = dColStd.Buffer.Span;
                for (int g = 0; g < groupKeys.Count; g++)
                {
                    var indices = _groups[groupKeys[g]];
                    double sum = 0; int count = 0;
                    foreach (var idx in indices)
                        if (!dColStd.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) { sum += span[idx]; count++; }
                    if (count <= 1) { vals[g] = double.NaN; continue; }
                    double mean = sum / count;
                    double sumSq = 0;
                    foreach (var idx in indices)
                        if (!dColStd.Nulls.IsNull(idx) && !double.IsNaN(span[idx])) { double d = span[idx] - mean; sumSq += d * d; }
                    vals[g] = Math.Sqrt(sumSq / (count - 1));
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
                        if (!IsValueMissing(srcCol, idx)) count++;
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
            AggFunc.Count => indices.Count(i => !IsValueMissing(col, i)),
            AggFunc.First => FindFirstNonMissing(col, indices),
            AggFunc.Last => FindLastNonMissing(col, indices),
            _ => ComputeNumericAggregate(col, indices, func)
        };
    }

    /// <summary>Find the first non-null, non-NaN value in the group.</summary>
    private static object? FindFirstNonMissing(IColumn col, List<int> indices)
    {
        foreach (var idx in indices)
        {
            if (!IsValueMissing(col, idx))
                return col.GetObject(idx);
        }
        return null;
    }

    /// <summary>Find the last non-null, non-NaN value in the group.</summary>
    private static object? FindLastNonMissing(IColumn col, List<int> indices)
    {
        for (int i = indices.Count - 1; i >= 0; i--)
        {
            if (!IsValueMissing(col, indices[i]))
                return col.GetObject(indices[i]);
        }
        return null;
    }

    /// <summary>
    /// Check if a value at the given index is missing: either null in the bitmask,
    /// or NaN for floating-point columns (double/float).
    /// </summary>
    private static bool IsValueMissing(IColumn col, int index)
    {
        if (col.IsNull(index)) return true;
        if (col is Column.Column<double> dc)
            return double.IsNaN(dc.Buffer.Span[index]);
        if (col is Column.Column<float> fc)
            return float.IsNaN(fc.Buffer.Span[index]);
        return false;
    }

    private static object? ComputeNumericAggregate(IColumn col, List<int> indices, AggFunc func)
    {
        var values = new List<double>();
        if (col is Column.Column<int> intCol)
        {
            var span = intCol.Buffer.Span;
            for (int i = 0; i < indices.Count; i++)
            {
                if (!col.IsNull(indices[i]))
                    values.Add((double)span[indices[i]]);
            }
        }
        else if (col is Column.Column<float> floatCol)
        {
            var span = floatCol.Buffer.Span;
            for (int i = 0; i < indices.Count; i++)
            {
                if (!col.IsNull(indices[i]) && !float.IsNaN(span[indices[i]]))
                    values.Add((double)span[indices[i]]);
            }
        }
        else if (col is Column.Column<long> longCol)
        {
            var span = longCol.Buffer.Span;
            for (int i = 0; i < indices.Count; i++)
            {
                if (!col.IsNull(indices[i]))
                    values.Add((double)span[indices[i]]);
            }
        }
        else if (col is Column.Column<double> dblCol)
        {
            var span = dblCol.Buffer.Span;
            for (int i = 0; i < indices.Count; i++)
            {
                if (!col.IsNull(indices[i]) && !double.IsNaN(span[indices[i]]))
                    values.Add(span[indices[i]]);
            }
        }
        else
        {
            foreach (var i in indices)
            {
                if (col.IsNull(i)) continue;
                var obj = col.GetObject(i);
                if (obj is double d && double.IsNaN(d)) continue;
                if (obj is float f && float.IsNaN(f)) continue;
                values.Add(Convert.ToDouble(obj));
            }
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
        if (values.Count <= 1) return double.NaN;
        double mean = values.Average();
        double sumSq = values.Sum(v => (v - mean) * (v - mean));
        return Math.Sqrt(sumSq / (values.Count - 1));
    }

    private static double ComputeVar(List<double> values)
    {
        if (values.Count <= 1) return double.NaN;
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
