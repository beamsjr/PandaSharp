#pragma warning disable CS8714, CS8604 // nullable type as dictionary key — intentional for pivot/reshape operations
using PandaSharp.Column;

namespace PandaSharp.Reshape;

public static class ReshapeExtensions
{
    /// <summary>
    /// Pivot from long to wide format. Unique values in 'columns' become new column names.
    /// </summary>
    public static DataFrame Pivot(this DataFrame df, string index, string columns, string values)
    {
        var indexCol = df[index];
        var colCol = df[columns];
        var valCol = df[values];

        // Get unique index values and pivot column values
        var uniqueIndex = new List<object?>();
        var indexMap = new Dictionary<object?, int>(new ObjectEqualityComparer());
        for (int i = 0; i < df.RowCount; i++)
        {
            var val = indexCol.GetObject(i);
            if (!indexMap.ContainsKey(val))
            {
                indexMap[val] = uniqueIndex.Count;
                uniqueIndex.Add(val);
            }
        }

        var uniqueCols = new List<object?>();
        var colMap = new Dictionary<object?, int>(new ObjectEqualityComparer());
        for (int i = 0; i < df.RowCount; i++)
        {
            var val = colCol.GetObject(i);
            if (!colMap.ContainsKey(val))
            {
                colMap[val] = uniqueCols.Count;
                uniqueCols.Add(val);
            }
        }

        // Build lookup: (indexPos, colPos) → value — O(1) per row via dictionaries
        var lookup = new Dictionary<(int, int), object?>();
        for (int i = 0; i < df.RowCount; i++)
        {
            int idxPos = indexMap[indexCol.GetObject(i)];
            int colPos = colMap[colCol.GetObject(i)];
            lookup[(idxPos, colPos)] = valCol.GetObject(i);
        }

        // Build result
        var resultCols = new List<IColumn>();

        // Index column
        var indexValues = uniqueIndex.ToArray();
        resultCols.Add(BuildColumn(index, indexCol.DataType, indexValues));

        // One column per unique pivot value
        for (int c = 0; c < uniqueCols.Count; c++)
        {
            var colName = uniqueCols[c]?.ToString() ?? "null";
            var vals = new object?[uniqueIndex.Count];
            for (int r = 0; r < uniqueIndex.Count; r++)
                vals[r] = lookup.GetValueOrDefault((r, c));
            resultCols.Add(BuildColumn(colName, valCol.DataType, vals));
        }

        return new DataFrame(resultCols);
    }

    /// <summary>
    /// Pivot table with aggregation. Like Excel pivot tables.
    /// Groups by (index, columns), aggregates values, then pivots columns to wide format.
    /// </summary>
    public static DataFrame PivotTable(this DataFrame df, string index, string columns, string values,
        Func<IEnumerable<double>, double>? aggFunc = null)
    {
        aggFunc ??= vals => vals.Sum();

        var indexCol = df[index];
        var colCol = df[columns];
        var valCol = df[values];

        // Get unique values with O(1) lookup maps
        var uniqueIndex = GetUnique(indexCol);
        var uniqueCols = GetUnique(colCol);
        var indexMap = BuildLookupMap(uniqueIndex);
        var colMap = BuildLookupMap(uniqueCols);

        // Group and aggregate: (indexVal, colVal) → aggregated value
        var groups = new Dictionary<(int, int), List<double>>();
        for (int i = 0; i < df.RowCount; i++)
        {
            if (!indexMap.TryGetValue(indexCol.GetObject(i), out int idxPos)) continue;
            if (!colMap.TryGetValue(colCol.GetObject(i), out int colPos)) continue;

            var key = (idxPos, colPos);
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<double>();
                groups[key] = list;
            }
            if (!valCol.IsNull(i))
                list.Add(Convert.ToDouble(valCol.GetObject(i)));
        }

        // Build result
        var resultCols = new List<IColumn>();
        resultCols.Add(BuildColumn(index, indexCol.DataType, uniqueIndex.ToArray()));

        for (int c = 0; c < uniqueCols.Count; c++)
        {
            var vals = new double?[uniqueIndex.Count];
            for (int r = 0; r < uniqueIndex.Count; r++)
            {
                if (groups.TryGetValue((r, c), out var list) && list.Count > 0)
                    vals[r] = aggFunc(list);
            }
            var colName = uniqueCols[c]?.ToString() ?? "null";
            resultCols.Add(Column<double>.FromNullable(colName, vals));
        }

        return new DataFrame(resultCols);
    }

    private static Dictionary<object, int> BuildLookupMap(List<object?> values)
    {
        var map = new Dictionary<object, int>(values.Count);
        for (int i = 0; i < values.Count; i++)
            if (values[i] is { } v) map[v] = i;
        return map;
    }

    private static List<object?> GetUnique(IColumn col)
    {
        var result = new List<object?>();
        var seen = new HashSet<object?>(new ObjectEqualityComparer());
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i);
            if (seen.Add(val)) result.Add(val);
        }
        return result;
    }

    /// <summary>
    /// Melt (unpivot) from wide to long format.
    /// </summary>
    public static DataFrame Melt(this DataFrame df, string[] idVars, string[]? valueVars = null,
        string varName = "variable", string valueName = "value")
    {
        valueVars ??= df.ColumnNames.Where(c => !idVars.Contains(c)).ToArray();

        var resultIds = new Dictionary<string, List<object?>>();
        foreach (var id in idVars)
            resultIds[id] = new List<object?>();

        var varValues = new List<string?>();
        var valValues = new List<object?>();

        for (int r = 0; r < df.RowCount; r++)
        {
            foreach (var valVar in valueVars)
            {
                foreach (var id in idVars)
                    resultIds[id].Add(df[id].GetObject(r));
                varValues.Add(valVar);
                valValues.Add(df[valVar].GetObject(r));
            }
        }

        var cols = new List<IColumn>();
        foreach (var id in idVars)
            cols.Add(BuildColumn(id, df[id].DataType, resultIds[id].ToArray()));
        cols.Add(new StringColumn(varName, varValues.ToArray()));

        // Determine value type from first value var
        var valType = valueVars.Length > 0 ? df[valueVars[0]].DataType : typeof(string);
        cols.Add(BuildColumn(valueName, valType, valValues.ToArray()));

        return new DataFrame(cols);
    }

    /// <summary>
    /// One-hot encode a categorical column, adding a boolean column per unique value.
    /// </summary>
    public static DataFrame GetDummies(this DataFrame df, string column, string? prefix = null)
    {
        var col = df[column];
        prefix ??= column;

        // Single pass: build category→row indices map
        var categoryIndices = new Dictionary<string, List<int>>();
        var categoryOrder = new List<string>();
        for (int i = 0; i < col.Length; i++)
        {
            var val = col.GetObject(i)?.ToString();
            if (val is null) continue;
            if (!categoryIndices.TryGetValue(val, out var list))
            {
                list = new List<int>();
                categoryIndices[val] = list;
                categoryOrder.Add(val);
            }
            list.Add(i);
        }

        // Start with all existing columns except the target
        var resultCols = new List<IColumn>();
        foreach (var name in df.ColumnNames)
        {
            if (name != column)
                resultCols.Add(df[name]);
        }

        // Build boolean columns from index lists (no re-scanning)
        foreach (var cat in categoryOrder)
        {
            var dummyName = $"{prefix}_{cat}";
            var values = new bool[df.RowCount];
            foreach (var idx in categoryIndices[cat])
                values[idx] = true;
            resultCols.Add(new Column<bool>(dummyName, values));
        }

        return new DataFrame(resultCols);
    }

    private static IColumn BuildColumn(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
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

    private class ObjectEqualityComparer : IEqualityComparer<object?>
    {
        public new bool Equals(object? x, object? y) => object.Equals(x, y);
        public int GetHashCode(object? obj) => obj?.GetHashCode() ?? 0;
    }
}
