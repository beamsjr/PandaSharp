using PandaSharp.Column;

namespace PandaSharp.Joins;

public static class JoinExtensions
{
    /// <summary>
    /// Join a fact table to multiple dimension tables on the same int key.
    /// Much faster than chaining .Join().Join().Join() because it builds
    /// the fact key index once and never copies fact columns between joins.
    /// </summary>
    public static DataFrame JoinMany(this DataFrame fact, string keyColumn, params DataFrame[] dimensions)
    {
        if (fact[keyColumn] is not Column<int> factKey || factKey.NullCount != 0)
            throw new ArgumentException("JoinMany requires a non-null int key column.");

        // Build fact key array once (used by all dimensions, read-only)
        var factKeys = factKey.Buffer.Span.ToArray();
        int nFactRows = fact.RowCount;
        var factColNames = new HashSet<string>(fact.ColumnNames);

        // Validate all dimensions have the right key column upfront
        foreach (var dim in dimensions)
            if (dim[keyColumn] is not Column<int> { NullCount: 0 })
                throw new ArgumentException($"Dimension table must have int key column '{keyColumn}'.");

        // Process all dimensions in parallel — each produces its own column list
        // Uses flat lookup tables instead of materialized row maps (saves 235MB allocation)
        var dimResults = new IColumn[dimensions.Length][];
        Parallel.For(0, dimensions.Length, di =>
        {
            var dim = dimensions[di];
            var dimKey = (Column<int>)dim[keyColumn];
            var dimSpan = dimKey.Buffer.Span;

            // Build dimension lookup: key → row index (small, fits in L1 cache)
            int maxKey = 0;
            for (int i = 0; i < dimSpan.Length; i++)
                if (dimSpan[i] > maxKey) maxKey = dimSpan[i];

            var flatLookup = new int[maxKey + 1];
            Array.Fill(flatLookup, -1);
            for (int i = 0; i < dim.RowCount; i++)
                flatLookup[dimSpan[i]] = i;

            // Map columns using double indirection: source[flatLookup[factKeys[i]]]
            // Avoids allocating a 14.7M-element dimRowMap per dimension
            var cols = new List<IColumn>();
            foreach (var colName in dim.ColumnNames)
            {
                if (colName == keyColumn) continue;
                if (factColNames.Contains(colName)) continue;
                cols.Add(BuildMappedColumnIndirect(dim[colName], factKeys, flatLookup, maxKey, nFactRows));
            }
            dimResults[di] = cols.ToArray();
        });

        // Build result: fact columns + all dimension columns (preserve dimension order)
        var resultCols = new List<IColumn>(fact.ColumnNames.Select(n => fact[n]));
        foreach (var dimCols in dimResults)
            resultCols.AddRange(dimCols);
        return new DataFrame(resultCols);
    }

    /// <summary>
    /// Map a dimension column using double indirection: output[i] = source[lookup[factKeys[i]]].
    /// Avoids materializing a full N-element rowMap per dimension.
    /// </summary>
    private static IColumn BuildMappedColumnIndirect(IColumn source, int[] factKeys, int[] lookup, int maxKey, int outputRows)
    {
        if (source is Column<double> dc)
        {
            var bytes = new byte[outputRows * sizeof(double)];
            var result = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(bytes.AsSpan());
            var srcArr = dc.Buffer.Span.ToArray();
            for (int i = 0; i < outputRows; i++)
            {
                int k = factKeys[i];
                int dimRow = (uint)k <= (uint)maxKey ? lookup[k] : -1;
                result[i] = dimRow >= 0 ? srcArr[dimRow] : 0;
            }
            return Column<double>.WrapResult(source.Name, bytes, outputRows);
        }
        if (source is Column<int> ic)
        {
            var bytes = new byte[outputRows * sizeof(int)];
            var result = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(bytes.AsSpan());
            var srcArr = ic.Buffer.Span.ToArray();
            for (int i = 0; i < outputRows; i++)
            {
                int k = factKeys[i];
                int dimRow = (uint)k <= (uint)maxKey ? lookup[k] : -1;
                result[i] = dimRow >= 0 ? srcArr[dimRow] : 0;
            }
            return Column<int>.WrapResult(source.Name, bytes, outputRows);
        }
        if (source is StringColumn sc)
        {
            var srcVals = sc.GetValues();
            var result = new string?[outputRows];
            for (int i = 0; i < outputRows; i++)
            {
                int k = factKeys[i];
                int row = (uint)k <= (uint)maxKey ? lookup[k] : -1;
                result[i] = row >= 0 ? srcVals[row] : null;
            }
            return StringColumn.CreateOwned(source.Name, result);
        }
        // Fallback: materialize row map
        var rowMap = new int[outputRows];
        for (int i = 0; i < outputRows; i++)
        {
            int k = factKeys[i];
            rowMap[i] = (uint)k <= (uint)maxKey ? lookup[k] : -1;
        }
        return BuildMappedColumn(source, rowMap, outputRows);
    }

    private static IColumn BuildMappedColumn(IColumn source, int[] rowMap, int outputRows)
    {
        if (source is Column<double> dc)
        {
            // Write directly into Arrow byte buffer — avoids 118MB memcpy
            var bytes = new byte[outputRows * sizeof(double)];
            var result = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(bytes.AsSpan());
            var srcArr = dc.Buffer.Span.ToArray(); // tiny dimension array
            for (int i = 0; i < outputRows; i++)
                result[i] = rowMap[i] >= 0 ? srcArr[rowMap[i]] : 0;
            return Column<double>.WrapResult(source.Name, bytes, outputRows);
        }
        if (source is Column<int> ic)
        {
            var bytes = new byte[outputRows * sizeof(int)];
            var result = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, int>(bytes.AsSpan());
            var srcArr = ic.Buffer.Span.ToArray();
            for (int i = 0; i < outputRows; i++)
                result[i] = rowMap[i] >= 0 ? srcArr[rowMap[i]] : 0;
            return Column<int>.WrapResult(source.Name, bytes, outputRows);
        }
        if (source is StringColumn sc)
        {
            var srcVals = sc.GetValues();
            var result = new string?[outputRows];
            for (int i = 0; i < outputRows; i++)
                result[i] = rowMap[i] >= 0 ? srcVals[rowMap[i]] : null;
            return StringColumn.CreateOwned(source.Name, result);
        }
        var objs = new object?[outputRows];
        for (int i = 0; i < outputRows; i++)
            objs[i] = rowMap[i] >= 0 ? source.GetObject(rowMap[i]) : null;
        return BuildColumnFromObjects(source.Name, source.DataType, objs);
    }

    public static DataFrame Join(this DataFrame left, DataFrame right, string on,
        JoinType how = JoinType.Inner, string leftSuffix = "_x", string rightSuffix = "_y")
    {
        return Join(left, right, [on], [on], how, leftSuffix, rightSuffix);
    }

    public static DataFrame Join(this DataFrame left, DataFrame right,
        string leftOn, string rightOn,
        JoinType how = JoinType.Inner, string leftSuffix = "_x", string rightSuffix = "_y")
    {
        return Join(left, right, [leftOn], [rightOn], how, leftSuffix, rightSuffix);
    }

    public static DataFrame Join(this DataFrame left, DataFrame right,
        string[] leftOn, string[] rightOn,
        JoinType how = JoinType.Inner, string leftSuffix = "_x", string rightSuffix = "_y")
    {
        if (leftOn.Length != rightOn.Length)
            throw new ArgumentException(
                $"leftOn has {leftOn.Length} column(s) [{string.Join(", ", leftOn.Select(c => $"'{c}'"))}] but rightOn has {rightOn.Length} column(s) [{string.Join(", ", rightOn.Select(c => $"'{c}'"))}]. They must have the same number of columns.");

        // Validate that join key columns exist
        foreach (var col in leftOn)
        {
            if (!left.ColumnNames.Contains(col))
                throw new ArgumentException(
                    $"Cannot join: column '{col}' not found in left DataFrame. Available columns: [{string.Join(", ", left.ColumnNames.Select(c => $"'{c}'"))}]");
        }
        foreach (var col in rightOn)
        {
            if (!right.ColumnNames.Contains(col))
                throw new ArgumentException(
                    $"Cannot join: column '{col}' not found in right DataFrame. Available columns: [{string.Join(", ", right.ColumnNames.Select(c => $"'{c}'"))}]");
        }

        if (how == JoinType.Cross)
            return CrossJoin(left, right, leftSuffix, rightSuffix);

        // Typed fast path: single int key, inner join — no boxing
        if (leftOn.Length == 1 && how == JoinType.Inner
            && left[leftOn[0]] is Column<int> leftIntCol && leftIntCol.NullCount == 0
            && right[rightOn[0]] is Column<int> rightIntCol && rightIntCol.NullCount == 0)
        {
            return TypedInnerJoin(left, right, leftOn[0], rightOn[0],
                leftIntCol, rightIntCol, leftSuffix, rightSuffix);
        }

        // Typed fast path: single string key, inner join — no boxing
        if (leftOn.Length == 1 && how == JoinType.Inner
            && left[leftOn[0]] is StringColumn leftStrCol
            && right[rightOn[0]] is StringColumn rightStrCol)
        {
            return TypedStringInnerJoin(left, right, leftOn[0], rightOn[0],
                leftStrCol, rightStrCol, leftSuffix, rightSuffix);
        }

        // Generic path
        var rightIndex = BuildHashIndex(right, rightOn);

        var leftIndices = new List<int>();
        var rightIndices = new List<int?>(); // null = no match

        var rightMatched = new HashSet<int>();

        for (int l = 0; l < left.RowCount; l++)
        {
            var key = ExtractKey(left, leftOn, l);
            var matches = rightIndex.TryGetValue(key, out var list) ? list : null;

            if (how == JoinType.Anti)
            {
                if (matches is null || matches.Count == 0)
                {
                    leftIndices.Add(l);
                    rightIndices.Add(null);
                }
                continue;
            }

            if (matches is not null && matches.Count > 0)
            {
                foreach (var r in matches)
                {
                    leftIndices.Add(l);
                    rightIndices.Add(r);
                    rightMatched.Add(r);
                }
            }
            else if (how == JoinType.Left || how == JoinType.Outer)
            {
                leftIndices.Add(l);
                rightIndices.Add(null);
            }
        }

        // For Right/Outer joins, add unmatched right rows
        if (how == JoinType.Right || how == JoinType.Outer)
        {
            for (int r = 0; r < right.RowCount; r++)
            {
                if (!rightMatched.Contains(r))
                {
                    leftIndices.Add(-1); // -1 = null left
                    rightIndices.Add(r);
                }
            }
        }

        return BuildJoinResult(left, right, leftOn, rightOn,
            leftIndices, rightIndices, leftSuffix, rightSuffix, how);
    }

    /// <summary>
    /// Typed inner join for single int key — zero boxing in probe/build phase.
    /// </summary>
    private static DataFrame TypedInnerJoin(DataFrame left, DataFrame right,
        string leftKeyName, string rightKeyName,
        Column<int> leftKey, Column<int> rightKey,
        string leftSuffix, string rightSuffix)
    {
        // Build hash table on right side — typed Dictionary<int, List<int>>
        var rightMap = new Dictionary<int, List<int>>(right.RowCount);
        var rSpan = rightKey.Buffer.Span;
        for (int i = 0; i < right.RowCount; i++)
        {
            int key = rSpan[i];
            if (!rightMap.TryGetValue(key, out var list))
            {
                list = new List<int>();
                rightMap[key] = list;
            }
            list.Add(i);
        }

        // Probe left side
        var leftIndices = new List<int>(left.RowCount);
        var rightIndices = new List<int>(left.RowCount);
        var lSpan = leftKey.Buffer.Span;
        for (int l = 0; l < left.RowCount; l++)
        {
            if (rightMap.TryGetValue(lSpan[l], out var matches))
            {
                foreach (var r in matches)
                {
                    leftIndices.Add(l);
                    rightIndices.Add(r);
                }
            }
        }

        // Build result via TakeRows — fully typed, no boxing
        var leftIdx = leftIndices.ToArray();
        var rightIdx = rightIndices.ToArray();
        var rightNonKey = right.ColumnNames.Where(c => c != rightKeyName).ToHashSet();
        var leftNames = new HashSet<string>(left.ColumnNames);
        var overlap = leftNames.Where(n => rightNonKey.Contains(n)).ToHashSet();

        var columns = new List<IColumn>();
        foreach (var colName in left.ColumnNames)
        {
            string outputName = overlap.Contains(colName) ? colName + leftSuffix : colName;
            columns.Add(left[colName].TakeRows(leftIdx).RenameOrKeep(outputName));
        }
        foreach (var colName in right.ColumnNames)
        {
            if (colName == rightKeyName) continue;
            string outputName = overlap.Contains(colName) ? colName + rightSuffix : colName;
            columns.Add(right[colName].TakeRows(rightIdx).RenameOrKeep(outputName));
        }

        return new DataFrame(columns);
    }

    /// <summary>Typed string inner join — no boxing in build/probe.</summary>
    private static DataFrame TypedStringInnerJoin(DataFrame left, DataFrame right,
        string leftKeyName, string rightKeyName,
        StringColumn leftKey, StringColumn rightKey,
        string leftSuffix, string rightSuffix)
    {
        // Build hash table on right side
        var rightVals = rightKey.GetValues();
        var rightMap = new Dictionary<string, List<int>>(right.RowCount / 2, StringComparer.Ordinal);
        for (int i = 0; i < right.RowCount; i++)
        {
            var key = rightVals[i] ?? "";
            if (!rightMap.TryGetValue(key, out var list))
            {
                list = new List<int>();
                rightMap[key] = list;
            }
            list.Add(i);
        }

        // Probe left side
        var leftVals = leftKey.GetValues();
        var leftIndices = new List<int>(left.RowCount);
        var rightIndices = new List<int>(left.RowCount);
        for (int l = 0; l < left.RowCount; l++)
        {
            if (rightMap.TryGetValue(leftVals[l] ?? "", out var matches))
            {
                foreach (var r in matches)
                {
                    leftIndices.Add(l);
                    rightIndices.Add(r);
                }
            }
        }

        // Build result
        var leftIdx = leftIndices.ToArray();
        var rightIdx = rightIndices.ToArray();
        var rightNonKey = right.ColumnNames.Where(c => c != rightKeyName).ToHashSet();
        var leftNames = new HashSet<string>(left.ColumnNames);
        var overlap = leftNames.Where(n => rightNonKey.Contains(n)).ToHashSet();

        var columns = new List<IColumn>();
        foreach (var colName in left.ColumnNames)
        {
            string outputName = overlap.Contains(colName) ? colName + leftSuffix : colName;
            columns.Add(left[colName].TakeRows(leftIdx).RenameOrKeep(outputName));
        }
        foreach (var colName in right.ColumnNames)
        {
            if (colName == rightKeyName) continue;
            string outputName = overlap.Contains(colName) ? colName + rightSuffix : colName;
            columns.Add(right[colName].TakeRows(rightIdx).RenameOrKeep(outputName));
        }

        return new DataFrame(columns);
    }

    private static Dictionary<JoinKey, List<int>> BuildHashIndex(DataFrame df, string[] keyColumns)
    {
        var index = new Dictionary<JoinKey, List<int>>();
        for (int i = 0; i < df.RowCount; i++)
        {
            var key = ExtractKey(df, keyColumns, i);
            if (!index.TryGetValue(key, out var list))
            {
                list = new List<int>();
                index[key] = list;
            }
            list.Add(i);
        }
        return index;
    }

    private static JoinKey ExtractKey(DataFrame df, string[] keyColumns, int row)
    {
        var values = new object?[keyColumns.Length];
        for (int k = 0; k < keyColumns.Length; k++)
            values[k] = df[keyColumns[k]].GetObject(row);
        return new JoinKey(values);
    }

    private static DataFrame BuildJoinResult(
        DataFrame left, DataFrame right,
        string[] leftKeyCols, string[] rightKeyCols,
        List<int> leftIndices, List<int?> rightIndices,
        string leftSuffix, string rightSuffix, JoinType how)
    {
        int resultRows = leftIndices.Count;
        var columns = new List<IColumn>();

        // Determine overlapping column names (excluding join keys)
        var rightNonKey = right.ColumnNames.Where(c => !rightKeyCols.Contains(c)).ToHashSet();
        var leftNonKey = left.ColumnNames.Where(c => !leftKeyCols.Contains(c)).ToHashSet();
        var overlap = leftNonKey.Intersect(rightNonKey).ToHashSet();

        // Check if we have any null indices (right/outer joins produce -1 on left side)
        bool hasNullLeft = leftIndices.Any(i => i < 0);
        bool hasNullRight = rightIndices.Any(i => !i.HasValue);

        // Pre-compute index arrays once (avoid repeated ToArray per column)
        int[]? leftIdx = !hasNullLeft ? leftIndices.ToArray() : null;
        int[]? rightIdx = !hasNullRight ? rightIndices.Select(i => i!.Value).ToArray() : null;

        // Add left columns — typed fast path when no null indices
        foreach (var colName in left.ColumnNames)
        {
            var col = left[colName];
            string outputName = overlap.Contains(colName) ? colName + leftSuffix : colName;

            if (leftIdx is not null)
            {
                columns.Add(col.TakeRows(leftIdx).RenameOrKeep(outputName));
            }
            else
            {
                var values = new object?[resultRows];
                for (int i = 0; i < resultRows; i++)
                    values[i] = leftIndices[i] >= 0 ? col.GetObject(leftIndices[i]) : null;
                columns.Add(BuildColumnFromObjects(outputName, col.DataType, values));
            }
        }

        // Add right columns — typed fast path when no null indices
        foreach (var colName in right.ColumnNames)
        {
            if (rightKeyCols.Contains(colName)) continue;

            var col = right[colName];
            string outputName = overlap.Contains(colName) ? colName + rightSuffix : colName;

            if (rightIdx is not null)
            {
                columns.Add(col.TakeRows(rightIdx).RenameOrKeep(outputName));
            }
            else
            {
                var values = new object?[resultRows];
                for (int i = 0; i < resultRows; i++)
                    values[i] = rightIndices[i] is { } ri ? col.GetObject(ri) : null;
                columns.Add(BuildColumnFromObjects(outputName, col.DataType, values));
            }
        }

        return new DataFrame(columns);
    }

    private static DataFrame CrossJoin(DataFrame left, DataFrame right, string leftSuffix, string rightSuffix)
    {
        int resultRows = left.RowCount * right.RowCount;
        var columns = new List<IColumn>();

        var rightNames = right.ColumnNames.ToHashSet();
        var leftNames = left.ColumnNames.ToHashSet();
        var overlap = leftNames.Intersect(rightNames).ToHashSet();

        foreach (var colName in left.ColumnNames)
        {
            var col = left[colName];
            string outputName = overlap.Contains(colName) ? colName + leftSuffix : colName;
            var values = new object?[resultRows];
            int idx = 0;
            for (int l = 0; l < left.RowCount; l++)
                for (int r = 0; r < right.RowCount; r++)
                    values[idx++] = col.GetObject(l);
            columns.Add(BuildColumnFromObjects(outputName, col.DataType, values));
        }

        foreach (var colName in right.ColumnNames)
        {
            var col = right[colName];
            string outputName = overlap.Contains(colName) ? colName + rightSuffix : colName;
            var values = new object?[resultRows];
            int idx = 0;
            for (int l = 0; l < left.RowCount; l++)
                for (int r = 0; r < right.RowCount; r++)
                    values[idx++] = col.GetObject(r);
            columns.Add(BuildColumnFromObjects(outputName, col.DataType, values));
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

    private readonly struct JoinKey : IEquatable<JoinKey>
    {
        private readonly object?[] _values;
        private readonly int _hash;

        public JoinKey(object?[] values)
        {
            _values = values;
            var h = new HashCode();
            foreach (var v in values) h.Add(v);
            _hash = h.ToHashCode();
        }

        public bool Equals(JoinKey other)
        {
            if (_values.Length != other._values.Length) return false;
            for (int i = 0; i < _values.Length; i++)
                if (!Equals(_values[i], other._values[i])) return false;
            return true;
        }

        public override bool Equals(object? obj) => obj is JoinKey other && Equals(other);
        public override int GetHashCode() => _hash;
    }
}
