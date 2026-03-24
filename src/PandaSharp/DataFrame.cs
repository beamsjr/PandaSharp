using System.Collections;
using PandaSharp.Column;
using PandaSharp.Display;
using PandaSharp.Index;
using PandaSharp.Indexing;

using PandaSharp.Concat;

namespace PandaSharp;

/// <summary>
/// High-performance, immutable columnar DataFrame.
/// Core data structure of PandaSharp — holds an ordered collection of typed columns.
/// </summary>
public class DataFrame : IEnumerable<DataFrameRow>
{
    /// <summary>
    /// Create a DataFrame from a dictionary of column name → values.
    /// Usage: DataFrame.FromDictionary(new() { ["Name"] = new[] {"Alice", "Bob"}, ["Age"] = new[] {25, 30} })
    /// </summary>
    public static DataFrame FromDictionary(Dictionary<string, Array> data)
    {
        var columns = new List<Column.IColumn>();
        foreach (var (name, values) in data)
        {
            columns.Add(BuildColumnFromArray(name, values));
        }
        return new DataFrame(columns);
    }

    private static Column.IColumn BuildColumnFromArray(string name, Array values)
    {
        return values switch
        {
            int[] v => new Column.Column<int>(name, v),
            long[] v => new Column.Column<long>(name, v),
            double[] v => new Column.Column<double>(name, v),
            float[] v => new Column.Column<float>(name, v),
            bool[] v => new Column.Column<bool>(name, v),
            DateTime[] v => new Column.Column<DateTime>(name, v),
            string?[] v => new Column.StringColumn(name, v),
            _ => throw new ArgumentException($"Unsupported array type: {values.GetType().Name} for column '{name}'")
        };
    }

    /// <summary>
    /// Concatenate DataFrames along rows (default) or columns.
    /// </summary>
    public static DataFrame Concat(params DataFrame[] frames) => ConcatExtensions.Concat(frames);
    /// <summary>
    /// Concatenate DataFrames along specified axis (0=rows, 1=columns).
    /// </summary>
    public static DataFrame Concat(int axis, params DataFrame[] frames) => ConcatExtensions.Concat(axis, frames);

    private readonly List<IColumn> _columns;
    private readonly Dictionary<string, int> _columnIndex;
    private readonly IIndex _index;

    /// <summary>Number of rows in the DataFrame.</summary>
    public int RowCount { get; }
    /// <summary>Number of columns in the DataFrame.</summary>
    public int ColumnCount => _columns.Count;
    /// <summary>Ordered list of column names.</summary>
    public IReadOnlyList<string> ColumnNames { get; }

    /// <summary>Create a DataFrame from columns.</summary>
    public DataFrame(params IColumn[] columns)
        : this((IEnumerable<IColumn>)columns) { }

    public DataFrame(IEnumerable<IColumn> columns)
    {
        _columns = new List<IColumn>(columns);
        _columnIndex = new Dictionary<string, int>();
        var names = new List<string>();

        for (int i = 0; i < _columns.Count; i++)
        {
            var col = _columns[i];
            if (_columnIndex.ContainsKey(col.Name))
                throw new ArgumentException($"Duplicate column name: '{col.Name}'.");
            _columnIndex[col.Name] = i;
            names.Add(col.Name);
        }

        ColumnNames = names.AsReadOnly();

        if (_columns.Count > 0)
        {
            RowCount = _columns[0].Length;
            for (int i = 1; i < _columns.Count; i++)
            {
                if (_columns[i].Length != RowCount)
                    throw new ArgumentException(
                        $"Column '{_columns[i].Name}' has {_columns[i].Length} rows, expected {RowCount}.");
            }
        }

        _index = new RangeIndex(RowCount);
    }

    // -- Column access --

    /// <summary>Access a column by name.</summary>
    public IColumn this[string columnName]
    {
        get
        {
            if (!_columnIndex.TryGetValue(columnName, out int idx))
                throw new KeyNotFoundException($"Column '{columnName}' not found.");
            return _columns[idx];
        }
    }

    /// <summary>Access a typed column by name.</summary>
    public Column<T> GetColumn<T>(string name) where T : struct => (Column<T>)this[name];

    /// <summary>Access a string column by name.</summary>
    public StringColumn GetStringColumn(string name) => (StringColumn)this[name];

    // -- Row access --

    /// <summary>Access a row by position index.</summary>
    public DataFrameRow this[int rowIndex]
    {
        get
        {
            if ((uint)rowIndex >= (uint)RowCount)
                throw new IndexOutOfRangeException($"Row index {rowIndex} is out of range for DataFrame with {RowCount} rows.");
            return new DataFrameRow(_columns, _columnIndex, rowIndex);
        }
    }

    // -- Slicing --

    /// <summary>Return the first n rows (default 5).</summary>
    public DataFrame Head(int count = 5)
    {
        count = Math.Min(count, RowCount);
        var cols = new List<Column.IColumn>(_columns.Count);
        for (int i = 0; i < _columns.Count; i++)
            cols.Add(_columns[i].Slice(0, count));
        return new DataFrame(cols);
    }

    /// <summary>Return the last n rows (default 5).</summary>
    public DataFrame Tail(int count = 5)
    {
        count = Math.Min(count, RowCount);
        int offset = RowCount - count;
        var cols = new List<Column.IColumn>(_columns.Count);
        for (int i = 0; i < _columns.Count; i++)
            cols.Add(_columns[i].Slice(offset, count));
        return new DataFrame(cols);
    }

    /// <summary>Select a subset of columns by name.</summary>
    public DataFrame Select(params string[] columns) =>
        new(columns.Select(name => this[name].Clone()));

    // -- Filter --

    /// <summary>Filter rows using a boolean mask.</summary>
    public DataFrame Filter(ReadOnlySpan<bool> mask)
    {
        if (mask.Length != RowCount)
            throw new ArgumentException("Mask length must match row count.");

        // Single pass: count and build index array simultaneously
        // Pre-allocate to RowCount (worst case all true), then slice
        var indices = new int[RowCount];
        int count = 0;
        for (int i = 0; i < mask.Length; i++)
        {
            indices[count] = i;  // always write (branchless-friendly)
            count += mask[i] ? 1 : 0;  // conditional increment
        }

        if (count == 0)
        {
            var emptyCols = new List<Column.IColumn>(_columns.Count);
            foreach (var col in _columns)
                emptyCols.Add(col.Slice(0, 0));
            return new DataFrame(emptyCols);
        }

        if (count == RowCount)
            return this;

        var idxSpan = indices.AsSpan(0, count);
        var filtered = new List<Column.IColumn>(_columns.Count);
        foreach (var col in _columns)
            filtered.Add(col.TakeRows(idxSpan));
        return new DataFrame(filtered);
    }

    public DataFrame Filter(bool[] mask) => Filter(mask.AsSpan());

    public DataFrame Filter(Func<DataFrameRow, bool> predicate)
    {
        var mask = new bool[RowCount];
        for (int i = 0; i < RowCount; i++)
            mask[i] = predicate(this[i]);
        return Filter(mask.AsSpan());
    }

    /// <summary>
    /// Filter rows where a column satisfies a predicate.
    /// Usage: df.Where("Age", val => (int)val! > 30)
    /// </summary>
    public DataFrame Where(string column, Func<object?, bool> predicate)
    {
        var col = this[column];
        var mask = new bool[RowCount];
        for (int i = 0; i < RowCount; i++)
            mask[i] = predicate(col.GetObject(i));
        return Filter(mask);
    }

    // -- Sort --

    /// <summary>Sort rows by a single column.</summary>
    public DataFrame Sort(string column, bool ascending = true)
    {
        var col = this[column];
        var indices = new int[RowCount];
        for (int i = 0; i < RowCount; i++) indices[i] = i;

        // Typed fast paths: use Array.Sort(keys, items) for cache-friendly sorting
        if (col is Column.Column<double> dc && dc.NullCount == 0)
            SortByKeys(indices, dc.Buffer.Span, ascending);
        else if (col is Column.Column<int> ic && ic.NullCount == 0)
            SortByKeys(indices, ic.Buffer.Span, ascending);
        else if (col is Column.Column<long> lc && lc.NullCount == 0)
            SortByKeys(indices, lc.Buffer.Span, ascending);
        else if (col is Column.Column<float> fc && fc.NullCount == 0)
            SortByKeys(indices, fc.Buffer.Span, ascending);
        else
        {
            // Fallback: boxing path for nullable/string/mixed columns
            Array.Sort(indices, (a, b) =>
            {
                var va = col.GetObject(a);
                var vb = col.GetObject(b);
                if (va is null && vb is null) return 0;
                if (va is null) return ascending ? 1 : -1;
                if (vb is null) return ascending ? -1 : 1;
                int cmp = Comparer<object>.Default.Compare(va, vb);
                return ascending ? cmp : -cmp;
            });
        }

        return new DataFrame(_columns.Select(c => c.TakeRows(indices)));
    }

    /// <summary>
    /// Sort indices by comparing values using a typed struct comparer.
    /// Handles ascending and descending correctly without a separate reverse pass.
    /// </summary>
    private static void SortByKeys<T>(int[] indices, ReadOnlySpan<T> values, bool ascending)
        where T : struct, IComparable<T>
    {
        Array.Sort(indices, new SpanComparer<T>(values.ToArray(), ascending));
    }

    private static void SortTyped<T>(int[] indices, ReadOnlySpan<T> values, bool ascending)
        where T : struct, IComparable<T>
    {
        var vals = values.ToArray();
        Array.Sort(indices, new SpanComparer<T>(vals, ascending));
    }

    private readonly struct SpanComparer<T> : IComparer<int> where T : struct, IComparable<T>
    {
        private readonly T[] _values;
        private readonly bool _ascending;
        public SpanComparer(T[] values, bool ascending) { _values = values; _ascending = ascending; }
        public int Compare(int a, int b) => _ascending
            ? _values[a].CompareTo(_values[b])
            : _values[b].CompareTo(_values[a]);
    }

    private sealed class BoxingComparer : IComparer<int>
    {
        private readonly Column.IColumn _col;
        private readonly bool _ascending;
        public BoxingComparer(Column.IColumn col, bool ascending) { _col = col; _ascending = ascending; }
        public int Compare(int a, int b)
        {
            var va = _col.GetObject(a);
            var vb = _col.GetObject(b);
            if (va is null && vb is null) return 0;
            if (va is null) return _ascending ? 1 : -1;
            if (vb is null) return _ascending ? -1 : 1;
            int cmp = Comparer<object>.Default.Compare(va, vb);
            return _ascending ? cmp : -cmp;
        }
    }

    private readonly struct ChainedComparer : IComparer<int>
    {
        private readonly IComparer<int>[] _comparers;
        public ChainedComparer(IComparer<int>[] comparers) => _comparers = comparers;
        public int Compare(int a, int b)
        {
            for (int k = 0; k < _comparers.Length; k++)
            {
                int cmp = _comparers[k].Compare(a, b);
                if (cmp != 0) return cmp;
            }
            return 0;
        }
    }

    /// <summary>
    /// Sort by multiple columns. First key is primary sort, second is tiebreaker, etc.
    /// </summary>
    public DataFrame Sort(params (string Column, bool Ascending)[] keys)
    {
        if (keys.Length == 0) return this;
        var cols = keys.Select(k => this[k.Column]).ToArray();
        var indices = Enumerable.Range(0, RowCount).ToArray();

        // Build typed struct comparers per key column — avoids delegate allocation and boxing
        var comparers = new IComparer<int>[keys.Length];
        for (int k = 0; k < keys.Length; k++)
        {
            var col = cols[k];
            bool asc = keys[k].Ascending;

            if (col is Column.Column<double> dc && dc.NullCount == 0)
                comparers[k] = new SpanComparer<double>(dc.Buffer.Span.ToArray(), asc);
            else if (col is Column.Column<int> ic && ic.NullCount == 0)
                comparers[k] = new SpanComparer<int>(ic.Buffer.Span.ToArray(), asc);
            else if (col is Column.Column<long> lc && lc.NullCount == 0)
                comparers[k] = new SpanComparer<long>(lc.Buffer.Span.ToArray(), asc);
            else if (col is Column.Column<float> fc && fc.NullCount == 0)
                comparers[k] = new SpanComparer<float>(fc.Buffer.Span.ToArray(), asc);
            else
                comparers[k] = new BoxingComparer(col, asc);
        }

        Array.Sort(indices, new ChainedComparer(comparers));

        return new DataFrame(_columns.Select(c => c.TakeRows(indices)));
    }

    /// <summary>
    /// Sort by column (pandas-style alias for Sort).
    /// </summary>
    public DataFrame SortValues(string column, bool ascending = true) => Sort(column, ascending);

    /// <summary>
    /// Sort by multiple columns (pandas-style alias).
    /// </summary>
    public DataFrame SortValues(params (string Column, bool Ascending)[] keys) => Sort(keys);

    /// <summary>
    /// Return the n rows with the largest values in the specified column.
    /// </summary>
    public DataFrame Nlargest(int n, string column) =>
        Sort(column, ascending: false).Head(n);

    /// <summary>
    /// Return the n rows with the smallest values in the specified column.
    /// </summary>
    public DataFrame Nsmallest(int n, string column) =>
        Sort(column, ascending: true).Head(n);

    // -- Apply --

    /// <summary>
    /// Apply a function to each row, producing a new column of results.
    /// Usage: df.Apply(row => (double)(int)row["Age"]! * 2, "DoubleAge")
    /// </summary>
    public DataFrame Apply<T>(Func<DataFrameRow, T> func, string columnName) where T : struct
    {
        var values = new T?[RowCount];
        for (int i = 0; i < RowCount; i++)
        {
            try { values[i] = func(this[i]); }
            catch (InvalidOperationException) { values[i] = null; }
            catch (NullReferenceException) { values[i] = null; }
        }
        return AddColumn(Column<T>.FromNullable(columnName, values));
    }

    /// <summary>
    /// Apply a function to each row, producing a new string column.
    /// </summary>
    public DataFrame Apply(Func<DataFrameRow, string?> func, string columnName)
    {
        var values = new string?[RowCount];
        for (int i = 0; i < RowCount; i++)
        {
            try { values[i] = func(this[i]); }
            catch (InvalidOperationException) { values[i] = null; }
            catch (NullReferenceException) { values[i] = null; }
        }
        return AddColumn(new Column.StringColumn(columnName, values));
    }

    // -- Copy and Equality --

    /// <summary>
    /// Create a deep copy of this DataFrame. All column data is copied.
    /// </summary>
    public DataFrame Copy() => new(_columns.Select(c => c.Clone()));

    /// <summary>
    /// Check if two DataFrames have identical schema and values.
    /// </summary>
    public bool ContentEquals(DataFrame other)
    {
        if (RowCount != other.RowCount || ColumnCount != other.ColumnCount) return false;
        if (!ColumnNames.SequenceEqual(other.ColumnNames)) return false;
        for (int c = 0; c < ColumnCount; c++)
        {
            var name = ColumnNames[c];
            if (this[name].DataType != other[name].DataType) return false;
            for (int r = 0; r < RowCount; r++)
            {
                if (!Equals(this[name].GetObject(r), other[name].GetObject(r)))
                    return false;
            }
        }
        return true;
    }

    /// <summary>
    /// Combine two DataFrames element-wise using a function on matching numeric columns.
    /// DataFrames must have the same shape and column names.
    /// Usage: df1.Combine(df2, (a, b) => a + b)
    /// </summary>
    public DataFrame Combine(DataFrame other, Func<double?, double?, double?> combiner)
    {
        if (RowCount != other.RowCount)
            throw new ArgumentException("DataFrames must have the same number of rows.");

        var cols = new List<Column.IColumn>();
        foreach (var name in ColumnNames)
        {
            if (!other.ColumnNames.Contains(name))
            {
                cols.Add(this[name]);
                continue;
            }

            var leftCol = this[name];
            var rightCol = other[name];

            if (IsNumericType(leftCol.DataType) && IsNumericType(rightCol.DataType))
            {
                var result = new double?[RowCount];
                for (int r = 0; r < RowCount; r++)
                {
                    double? lv = leftCol.IsNull(r) ? null : Convert.ToDouble(leftCol.GetObject(r));
                    double? rv = rightCol.IsNull(r) ? null : Convert.ToDouble(rightCol.GetObject(r));
                    result[r] = combiner(lv, rv);
                }
                cols.Add(Column.Column<double>.FromNullable(name, result));
            }
            else
            {
                cols.Add(leftCol); // non-numeric: keep left
            }
        }
        return new DataFrame(cols);
    }

    // -- Mutation (returns new DataFrame) --

    /// <summary>
    /// Update values in a column using a transform function. Returns a new DataFrame.
    /// Usage: df.UpdateColumn("Age", col => col.Map(v => v + 1)) — must return same-length column.
    /// </summary>
    public DataFrame UpdateColumn(string name, Func<Column.IColumn, Column.IColumn> transform)
    {
        return new DataFrame(_columns.Select(c =>
            c.Name == name ? transform(c).Clone(name) : c));
    }

    /// <summary>
    /// Replace a column entirely with a new one (same length required).
    /// </summary>
    public DataFrame ReplaceColumn(string name, Column.IColumn newColumn)
    {
        if (newColumn.Length != RowCount)
            throw new ArgumentException($"Column length {newColumn.Length} doesn't match row count {RowCount}.");
        var renamed = newColumn.Name == name ? newColumn : newColumn.Clone(name);
        return new DataFrame(_columns.Select(c => c.Name == name ? renamed : c));
    }

    /// <summary>
    /// Fluent column assignment (like pandas .assign()).
    /// If a column with the same name exists, it is replaced. Otherwise, it is added.
    /// Usage: df.Assign("Total", col) or chained: df.Assign("A", colA).Assign("B", colB)
    /// </summary>
    public DataFrame Assign(string name, IColumn column)
    {
        var renamed = column.Name == name ? column : column.Clone(name);
        if (ColumnNames.Contains(name))
        {
            // Replace existing
            return new DataFrame(_columns.Select(c => c.Name == name ? renamed : c));
        }
        return AddColumn(renamed);
    }

    public DataFrame AddColumn(IColumn column)
    {
        if (column.Length != RowCount && RowCount > 0)
            throw new ArgumentException($"Column length {column.Length} doesn't match row count {RowCount}.");
        var cols = _columns.Append(column);
        return new DataFrame(cols);
    }

    /// <summary>
    /// Cast a column to a different type, returning a new DataFrame.
    /// Usage: df.CastColumn("age", typeof(double)) or df.CastColumn&lt;int, double&gt;("age")
    /// </summary>
    public DataFrame CastColumn(string columnName, Type targetType)
    {
        var col = this[columnName];
        IColumn castCol;

        if (targetType == typeof(double))
            castCol = CastToDouble(col);
        else if (targetType == typeof(int))
            castCol = CastToInt(col);
        else if (targetType == typeof(long))
            castCol = CastToLong(col);
        else if (targetType == typeof(string))
            castCol = CastToString(col);
        else
            throw new NotSupportedException($"Cannot cast to {targetType.Name}");

        return new DataFrame(_columns.Select(c => c.Name == columnName ? castCol : c));
    }

    private static IColumn CastToDouble(IColumn col)
    {
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.IsNull(i) ? null : Convert.ToDouble(col.GetObject(i));
        return Column.Column<double>.FromNullable(col.Name, result);
    }

    private static IColumn CastToInt(IColumn col)
    {
        var result = new int?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.IsNull(i) ? null : Convert.ToInt32(col.GetObject(i));
        return Column.Column<int>.FromNullable(col.Name, result);
    }

    private static IColumn CastToLong(IColumn col)
    {
        var result = new long?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.IsNull(i) ? null : Convert.ToInt64(col.GetObject(i));
        return Column.Column<long>.FromNullable(col.Name, result);
    }

    private static IColumn CastToString(IColumn col)
    {
        var result = new string?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.GetObject(i)?.ToString();
        return new Column.StringColumn(col.Name, result);
    }

    public DataFrame DropColumn(string name)
    {
        return new DataFrame(_columns.Where(c => c.Name != name));
    }

    public DataFrame RenameColumn(string oldName, string newName)
    {
        return new DataFrame(_columns.Select(c =>
            c.Name == oldName ? c.Clone(newName) : c));
    }

    /// <summary>
    /// Rename multiple columns using a dictionary.
    /// Usage: df.RenameColumns(new() { ["OldName"] = "NewName", ["Age"] = "Years" })
    /// </summary>
    public DataFrame RenameColumns(Dictionary<string, string> mapping)
    {
        return new DataFrame(_columns.Select(c =>
            mapping.TryGetValue(c.Name, out var newName) ? c.Clone(newName) : c));
    }

    /// <summary>
    /// Reorder columns by name. Unmentioned columns are dropped.
    /// </summary>
    public DataFrame ReorderColumns(params string[] columnOrder)
    {
        return new DataFrame(columnOrder.Select(name => this[name]));
    }

    /// <summary>
    /// Drop columns by name.
    /// </summary>
    public DataFrame DropColumns(params string[] names)
    {
        var dropSet = new HashSet<string>(names);
        return new DataFrame(_columns.Where(c => !dropSet.Contains(c.Name)));
    }

    /// <summary>
    /// Remove duplicate rows. Compares all columns by default, or a subset.
    /// </summary>
    public DataFrame DropDuplicates(params string[] subset)
    {
        var checkCols = subset.Length > 0
            ? subset.Select(n => this[n]).ToArray()
            : _columns.ToArray();

        // Bucketed dedup: Dictionary<hash, List<rowIndex>> for O(1) amortized collision handling
        var buckets = new Dictionary<int, List<int>>();
        var keepMask = new bool[RowCount];

        for (int r = 0; r < RowCount; r++)
        {
            var hash = new HashCode();
            foreach (var col in checkCols)
                hash.Add(col.GetObject(r));
            int h = hash.ToHashCode();

            if (!buckets.TryGetValue(h, out var bucket))
            {
                bucket = new List<int> { r };
                buckets[h] = bucket;
                keepMask[r] = true;
                continue;
            }

            // Only compare within the same hash bucket
            bool isDuplicate = false;
            foreach (var prev in bucket)
            {
                bool allEqual = true;
                foreach (var col in checkCols)
                {
                    if (!Equals(col.GetObject(r), col.GetObject(prev)))
                    { allEqual = false; break; }
                }
                if (allEqual) { isDuplicate = true; break; }
            }

            keepMask[r] = !isDuplicate;
            if (!isDuplicate) bucket.Add(r);
        }

        return Filter(keepMask);
    }

    /// <summary>
    /// Return (RowCount, ColumnCount) tuple.
    /// </summary>
    public (int Rows, int Columns) Shape => (RowCount, ColumnCount);

    /// <summary>
    /// Estimated total memory usage in bytes across all columns.
    /// </summary>
    public long Memory()
    {
        long total = 0;
        foreach (var col in _columns)
        {
            if (col is CategoricalColumn cat)
                total += cat.EstimatedBytes;
            else if (col is StringColumn sc)
            {
                for (int i = 0; i < sc.Length; i++)
                    total += (sc[i]?.Length ?? 0) * 2 + IntPtr.Size;
            }
            else
            {
                int elemSize = col.DataType == typeof(int) ? 4
                    : col.DataType == typeof(long) ? 8
                    : col.DataType == typeof(double) ? 8
                    : col.DataType == typeof(float) ? 4
                    : col.DataType == typeof(bool) ? 1
                    : col.DataType == typeof(DateTime) ? 8
                    : 8;
                total += (long)elemSize * col.Length + (col.Length + 7) / 8;
            }
        }
        return total;
    }

    /// <summary>
    /// Quick summary string: shape, column types, null counts, memory.
    /// Great for REPL exploration.
    /// </summary>
    public string Summary()
    {
        var lines = new List<string>
        {
            $"DataFrame: {RowCount} rows × {ColumnCount} columns ({Memory():N0} bytes)",
            ""
        };
        foreach (var name in ColumnNames)
        {
            var col = this[name];
            var typeName = col.DataType == typeof(int) ? "int32" : col.DataType == typeof(long) ? "int64"
                : col.DataType == typeof(double) ? "float64" : col.DataType == typeof(float) ? "float32"
                : col.DataType == typeof(bool) ? "bool" : col.DataType == typeof(DateTime) ? "datetime"
                : col.DataType == typeof(string) ? "string" : col.DataType.Name;
            var nullInfo = col.NullCount > 0 ? $" ({col.NullCount} nulls)" : "";
            lines.Add($"  {name,-20} {typeName,-10} {col.Length - col.NullCount} non-null{nullInfo}");
        }
        return string.Join(Environment.NewLine, lines);
    }

    /// <summary>
    /// Select columns that match specified data types.
    /// Usage: df.SelectDtypes(include: typeof(int), typeof(double))
    /// </summary>
    public DataFrame SelectDtypes(params Type[] include)
    {
        var set = new HashSet<Type>(include);
        var cols = _columns.Where(c => set.Contains(c.DataType)).ToList();
        return new DataFrame(cols);
    }

    /// <summary>
    /// Select columns excluding specified data types.
    /// </summary>
    public DataFrame ExcludeDtypes(params Type[] exclude)
    {
        var set = new HashSet<Type>(exclude);
        var cols = _columns.Where(c => !set.Contains(c.DataType)).ToList();
        return new DataFrame(cols);
    }

    /// <summary>
    /// Apply a function to every element in the DataFrame (element-wise).
    /// Returns a new DataFrame with all string columns (since types may change).
    /// </summary>
    public DataFrame ApplyMap(Func<object?, string?> func)
    {
        var cols = new List<Column.IColumn>();
        foreach (var col in _columns)
        {
            var values = new string?[RowCount];
            for (int r = 0; r < RowCount; r++)
                values[r] = func(col.GetObject(r));
            cols.Add(new Column.StringColumn(col.Name, values));
        }
        return new DataFrame(cols);
    }

    /// <summary>
    /// For each row, return the column name containing the minimum numeric value.
    /// </summary>
    public Column.StringColumn Idxmin()
    {
        var numCols = _columns.Where(c => IsNumericType(c.DataType)).ToList();
        var result = new string?[RowCount];
        for (int r = 0; r < RowCount; r++)
        {
            string? bestCol = null;
            double bestVal = double.MaxValue;
            foreach (var col in numCols)
            {
                if (col.IsNull(r)) continue;
                double v = Convert.ToDouble(col.GetObject(r));
                if (v < bestVal) { bestVal = v; bestCol = col.Name; }
            }
            result[r] = bestCol;
        }
        return new Column.StringColumn("idxmin", result);
    }

    /// <summary>
    /// For each row, return the column name containing the maximum numeric value.
    /// </summary>
    public Column.StringColumn Idxmax()
    {
        var numCols = _columns.Where(c => IsNumericType(c.DataType)).ToList();
        var result = new string?[RowCount];
        for (int r = 0; r < RowCount; r++)
        {
            string? bestCol = null;
            double bestVal = double.MinValue;
            foreach (var col in numCols)
            {
                if (col.IsNull(r)) continue;
                double v = Convert.ToDouble(col.GetObject(r));
                if (v > bestVal) { bestVal = v; bestCol = col.Name; }
            }
            result[r] = bestCol;
        }
        return new Column.StringColumn("idxmax", result);
    }

    /// <summary>
    /// Iterate over columns as (name, column) pairs.
    /// </summary>
    public IEnumerable<(string Name, Column.IColumn Column)> Itercolumns()
    {
        foreach (var name in ColumnNames)
            yield return (name, this[name]);
    }

    /// <summary>
    /// Return a DataFrame containing only numeric columns (int, long, float, double).
    /// </summary>
    public DataFrame NumericOnly() => SelectDtypes(typeof(int), typeof(long), typeof(float), typeof(double));

    /// <summary>Shorthand for Transpose().</summary>
    public DataFrame T => Transpose();

    /// <summary>
    /// Describe all columns including string stats (count, unique, top, freq).
    /// Returns a DataFrame with mixed stats.
    /// </summary>
    public DataFrame DescribeAll()
    {
        var statNames = new List<string> { "count", "unique", "top", "freq", "mean", "std", "min", "25%", "50%", "75%", "max" };
        var columns = new List<Column.IColumn>();
        columns.Add(new Column.StringColumn("stat", statNames.ToArray()));

        foreach (var name in ColumnNames)
        {
            var col = this[name];
            var vals = new string?[statNames.Count];

            int nonNull = col.Length - col.NullCount;
            vals[0] = nonNull.ToString(); // count

            if (col is Column.StringColumn sc || col is Column.CategoricalColumn)
            {
                // String stats
                var unique = new HashSet<string>();
                var freq = new Dictionary<string, int>();
                for (int i = 0; i < col.Length; i++)
                {
                    var v = col.GetObject(i)?.ToString();
                    if (v is null) continue;
                    unique.Add(v);
                    freq[v] = freq.GetValueOrDefault(v) + 1;
                }
                vals[1] = unique.Count.ToString();
                if (freq.Count > 0)
                {
                    var top = freq.MaxBy(kv => kv.Value);
                    vals[2] = top.Key;
                    vals[3] = top.Value.ToString();
                }
            }
            else if (IsNumericType(col.DataType))
            {
                // Numeric stats
                var values = new List<double>();
                for (int i = 0; i < col.Length; i++)
                    if (!col.IsNull(i)) values.Add(Convert.ToDouble(col.GetObject(i)));

                if (values.Count > 0)
                {
                    values.Sort();
                    vals[4] = values.Average().ToString("G6");
                    if (values.Count > 1)
                    {
                        double mean = values.Average();
                        double std = Math.Sqrt(values.Sum(v => (v - mean) * (v - mean)) / (values.Count - 1));
                        vals[5] = std.ToString("G6");
                    }
                    vals[6] = values[0].ToString("G6");
                    vals[7] = Percentile(values, 0.25).ToString("G6");
                    vals[8] = Percentile(values, 0.50).ToString("G6");
                    vals[9] = Percentile(values, 0.75).ToString("G6");
                    vals[10] = values[^1].ToString("G6");
                }
            }

            columns.Add(new Column.StringColumn(name, vals));
        }

        return new DataFrame(columns);
    }

    private static double Percentile(List<double> sorted, double p)
    {
        double pos = p * (sorted.Count - 1);
        int lower = (int)Math.Floor(pos);
        int upper = (int)Math.Ceiling(pos);
        if (lower == upper) return sorted[lower];
        return sorted[lower] * (1 - (pos - lower)) + sorted[upper] * (pos - lower);
    }

    /// <summary>
    /// Returns a dictionary of column name → data type.
    /// </summary>
    public IReadOnlyDictionary<string, Type> Dtypes
    {
        get
        {
            var dict = new Dictionary<string, Type>();
            foreach (var name in ColumnNames)
                dict[name] = this[name].DataType;
            return dict;
        }
    }

    /// <summary>
    /// True if the DataFrame has zero rows or zero columns.
    /// </summary>
    public bool IsEmpty => RowCount == 0 || ColumnCount == 0;

    /// <summary>
    /// Aggregate each numeric column with a function, returning a single-row DataFrame.
    /// Usage: df.Agg(values => values.Sum()) or df.Agg(values => values.Average())
    /// </summary>
    public DataFrame Agg(Func<IEnumerable<double>, double> func)
    {
        var cols = new List<Column.IColumn>();
        foreach (var name in ColumnNames)
        {
            var col = this[name];
            if (!IsNumericType(col.DataType)) continue;

            var values = new List<double>();
            for (int i = 0; i < RowCount; i++)
            {
                if (!col.IsNull(i))
                    values.Add(Convert.ToDouble(col.GetObject(i)));
            }
            cols.Add(new Column.Column<double>(name, [values.Count > 0 ? func(values) : double.NaN]));
        }
        return new DataFrame(cols);
    }

    /// <summary>
    /// Apply a column transform to all numeric columns.
    /// </summary>
    public DataFrame ApplyColumns(Func<Column.IColumn, Column.IColumn> transform)
    {
        return new DataFrame(_columns.Select(c => transform(c)));
    }

    /// <summary>
    /// Clip all numeric columns to [lower, upper].
    /// </summary>
    public DataFrame Clip(double lower, double upper)
    {
        var cols = new List<Column.IColumn>();
        foreach (var col in _columns)
        {
            if (col is Column.Column<double> dc)
                cols.Add(Column.ColumnExtensions.Clip(dc, lower, upper));
            else if (col is Column.Column<int> ic)
                cols.Add(Column.ColumnExtensions.Clip(ic, (int)lower, (int)upper));
            else
                cols.Add(col);
        }
        return new DataFrame(cols);
    }

    private static bool IsNumericType(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);

    /// <summary>
    /// Iterate rows as (index, row) pairs.
    /// Usage: foreach (var (idx, row) in df.Iterrows()) { ... }
    /// </summary>
    public IEnumerable<(int Index, DataFrameRow Row)> Iterrows()
    {
        for (int i = 0; i < RowCount; i++)
            yield return (i, this[i]);
    }

    /// <summary>
    /// Iterate rows as object[] tuples (faster than DataFrameRow for bulk access).
    /// </summary>
    public IEnumerable<object?[]> Itertuples()
    {
        for (int r = 0; r < RowCount; r++)
        {
            var tuple = new object?[ColumnCount];
            for (int c = 0; c < ColumnCount; c++)
                tuple[c] = _columns[c].GetObject(r);
            yield return tuple;
        }
    }

    /// <summary>
    /// Filter rows using a query string.
    /// Supports: "Column > value", "Column == value", "Column != value",
    /// "Column >= value", "Column &lt; value", "Column &lt;= value".
    /// Compound: "Age > 30 and Salary &lt; 100000", "Age &lt; 25 or Age > 40"
    /// Usage: df.Query("Age > 30")
    /// </summary>
    public DataFrame Query(string query)
    {
        // Handle compound AND/OR
        var andParts = query.Split(new[] { " and ", " AND ", " && " }, StringSplitOptions.TrimEntries);
        if (andParts.Length > 1)
        {
            var result = this;
            foreach (var part in andParts)
                result = result.Query(part);
            return result;
        }

        var orParts = query.Split(new[] { " or ", " OR ", " || " }, StringSplitOptions.TrimEntries);
        if (orParts.Length > 1)
        {
            var masks = orParts.Select(part =>
            {
                var (cn, o, vs) = ParseQuery(part);
                return EvaluateQueryMask(cn, o, vs);
            }).ToArray();

            var combined = masks[0];
            for (int i = 1; i < masks.Length; i++)
                combined = Column.MaskExtensions.Or(combined, masks[i]);
            return Filter(combined);
        }

        var (colName, op, valueStr) = ParseQuery(query);
        var col = this[colName];

        var mask = new bool[RowCount];
        for (int r = 0; r < RowCount; r++)
        {
            if (col.IsNull(r)) continue;
            var val = col.GetObject(r);
            int cmp = CompareToString(val!, valueStr, col.DataType);
            mask[r] = op switch
            {
                ">" => cmp > 0,
                ">=" => cmp >= 0,
                "<" => cmp < 0,
                "<=" => cmp <= 0,
                "==" => cmp == 0,
                "!=" => cmp != 0,
                _ => false
            };
        }
        return Filter(mask);
    }

    private bool[] EvaluateQueryMask(string colName, string op, string valueStr)
    {
        var col = this[colName];
        var mask = new bool[RowCount];
        for (int r = 0; r < RowCount; r++)
        {
            if (col.IsNull(r)) continue;
            int cmp = CompareToString(col.GetObject(r)!, valueStr, col.DataType);
            mask[r] = op switch
            {
                ">" => cmp > 0, ">=" => cmp >= 0, "<" => cmp < 0,
                "<=" => cmp <= 0, "==" => cmp == 0, "!=" => cmp != 0,
                _ => false
            };
        }
        return mask;
    }

    private static (string Column, string Op, string Value) ParseQuery(string query)
    {
        string[] ops = [">=", "<=", "!=", "==", ">", "<"];
        foreach (var op in ops)
        {
            int idx = query.IndexOf(op, StringComparison.Ordinal);
            if (idx > 0)
            {
                string col = query[..idx].Trim();
                string val = query[(idx + op.Length)..].Trim().Trim('\'', '"');
                return (col, op, val);
            }
        }
        throw new ArgumentException($"Cannot parse query: '{query}'. Expected format: 'Column > value'");
    }

    private static int CompareToString(object val, string valueStr, Type colType)
    {
        if (colType == typeof(int) && int.TryParse(valueStr, out int iv))
            return ((int)val).CompareTo(iv);
        if (colType == typeof(long) && long.TryParse(valueStr, out long lv))
            return ((long)val).CompareTo(lv);
        if (colType == typeof(double) && double.TryParse(valueStr, System.Globalization.CultureInfo.InvariantCulture, out double dv))
            return ((double)val).CompareTo(dv);
        if (colType == typeof(float) && float.TryParse(valueStr, System.Globalization.CultureInfo.InvariantCulture, out float fv))
            return ((float)val).CompareTo(fv);
        if (colType == typeof(string))
            return string.Compare((string)val, valueStr, StringComparison.Ordinal);
        return val.ToString()!.CompareTo(valueStr);
    }

    /// <summary>
    /// Transpose the DataFrame: rows become columns and columns become rows.
    /// All values are converted to strings since columns may have mixed types.
    /// First column of result contains original column names.
    /// </summary>
    public DataFrame Transpose()
    {
        var cols = new List<Column.IColumn>();

        // Column names become first column
        cols.Add(new Column.StringColumn("column", ColumnNames.ToArray()));

        // Each original row becomes a column
        for (int r = 0; r < RowCount; r++)
        {
            var values = new string?[ColumnCount];
            for (int c = 0; c < ColumnCount; c++)
                values[c] = _columns[c].GetObject(r)?.ToString();
            cols.Add(new Column.StringColumn($"row_{r}", values));
        }

        return new DataFrame(cols);
    }

    /// <summary>
    /// Pipe a DataFrame through a custom transformation function.
    /// Enables fluent chaining: df.Pipe(Normalize).Pipe(AddFeatures).Pipe(FilterOutliers)
    /// </summary>
    public DataFrame Pipe(Func<DataFrame, DataFrame> transform) => transform(this);

    /// <summary>
    /// Pipe with an additional argument.
    /// Usage: df.Pipe(ScaleColumn, "Salary", 1000.0)
    /// </summary>
    public DataFrame Pipe<TArg>(Func<DataFrame, TArg, DataFrame> transform, TArg arg) => transform(this, arg);

    // -- Index management --

    /// <summary>
    /// Promote a column to the index (removes it from columns).
    private IColumn? _indexColumn;
    private string? _indexColumnName;

    /// <summary>The name of the current index column, or null if using default range index.</summary>
    public string? IndexName => _indexColumnName;

    // -- Indexing accessors --

    /// <summary>Label-based indexing (like pandas .loc).</summary>
    public LocAccessor Loc => new(this);

    /// <summary>Position-based indexing (like pandas .iloc).</summary>
    public ILocAccessor ILoc => new(this);

    /// <summary>Fast scalar access by row index and column name. Like pandas df.at[row, col].</summary>
    public AtAccessor At => new(this);

    /// <summary>Fast scalar access by integer position. Like pandas df.iat[row, col].</summary>
    public IAtAccessor IAt => new(this);

    /// <summary>
    /// Cross-section: select rows where a column matches a value, then drop that column.
    /// Like pandas df.xs(key, level=column).
    /// </summary>
    public DataFrame Xs(string column, object value)
    {
        var col = this[column];
        var mask = new bool[RowCount];
        for (int i = 0; i < RowCount; i++)
            mask[i] = Equals(col.GetObject(i), value);
        return Filter(mask).DropColumn(column);
    }

    /// <summary>
    /// Cross-section on the index column (if set via SetIndex).
    /// </summary>
    public DataFrame Xs(object value)
    {
        if (_indexColumn is null || _indexColumnName is null)
            throw new InvalidOperationException("No index set. Use SetIndex() first or Xs(column, value).");

        var mask = new bool[RowCount];
        for (int i = 0; i < RowCount; i++)
            mask[i] = Equals(_indexColumn.GetObject(i), value);
        return Filter(mask);
    }

    /// <summary>
    /// Set multiple columns as a MultiIndex, removing them from visible columns.
    /// </summary>
    public DataFrame SetIndex(params string[] columnNames)
    {
        if (columnNames.Length == 1)
        {
            // Single column — use existing behavior
            var indexCol = this[columnNames[0]];
            var otherCols = _columns.Where(c => c.Name != columnNames[0]);
            var result = new DataFrame(otherCols);
            result._indexColumnName = columnNames[0];
            result._indexColumn = indexCol;
            return result;
        }

        // Multi-column → MultiIndex
        var indexCols = columnNames.Select(n => this[n]).ToArray();
        var multiIndex = new Index.MultiIndex(indexCols, columnNames);
        var remaining = _columns.Where(c => !columnNames.Contains(c.Name));
        var df = new DataFrame(remaining);
        df._multiIndex = multiIndex;
        return df;
    }

    /// <summary>
    /// Reset the index, restoring index columns as regular columns.
    /// </summary>
    public DataFrame ResetIndex()
    {
        if (_multiIndex is not null)
        {
            var cols = new List<Column.IColumn>();
            for (int l = 0; l < _multiIndex.NLevels; l++)
                cols.Add(_multiIndex.GetLevelValues(l));
            cols.AddRange(_columns);
            return new DataFrame(cols);
        }

        if (_indexColumn is null || _indexColumnName is null) return this;
        var result = new List<Column.IColumn> { _indexColumn.Clone(_indexColumnName) };
        result.AddRange(_columns);
        return new DataFrame(result);
    }

    private Index.MultiIndex? _multiIndex;

    /// <summary>The MultiIndex if set, or null.</summary>
    public Index.MultiIndex? MultiIndex => _multiIndex;

    // -- Slicing --

    /// <summary>
    /// Return a slice of rows from offset with given length.
    /// </summary>
    public DataFrame Slice(int offset, int length)
    {
        if (offset < 0 || length < 0 || offset + length > RowCount)
            throw new ArgumentOutOfRangeException();
        return new DataFrame(_columns.Select(c => c.Slice(offset, length)));
    }

    /// <summary>
    /// Return n random rows.
    /// </summary>
    public DataFrame Sample(int n, int? seed = null)
    {
        if (n > RowCount) n = RowCount;
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var indices = Enumerable.Range(0, RowCount).OrderBy(_ => rng.Next()).Take(n).ToArray();
        Array.Sort(indices);
        return new DataFrame(_columns.Select(c => c.TakeRows(indices)));
    }

    /// <summary>
    /// Return a fraction of rows randomly sampled.
    /// Usage: df.SampleFrac(0.5) returns ~50% of rows.
    /// </summary>
    public DataFrame SampleFrac(double fraction, int? seed = null)
    {
        if (fraction < 0 || fraction > 1)
            throw new ArgumentOutOfRangeException(nameof(fraction), "Must be between 0 and 1.");
        return Sample((int)Math.Round(RowCount * fraction), seed);
    }

    /// <summary>
    /// Return a boolean mask indicating duplicate rows.
    /// First occurrence is NOT marked as duplicate (keep='first').
    /// </summary>
    public bool[] Duplicated(params string[] subset)
    {
        var checkCols = subset.Length > 0
            ? subset.Select(n => this[n]).ToArray()
            : _columns.ToArray();

        var seen = new List<int>(); // hashes of seen rows
        var seenRows = new List<int>(); // indices of kept rows for collision check
        var mask = new bool[RowCount];

        for (int r = 0; r < RowCount; r++)
        {
            var hash = new HashCode();
            foreach (var col in checkCols) hash.Add(col.GetObject(r));
            int h = hash.ToHashCode();

            bool isDup = false;
            if (seen.Contains(h))
            {
                foreach (var prev in seenRows)
                {
                    bool allEqual = true;
                    foreach (var col in checkCols)
                    {
                        if (!Equals(col.GetObject(r), col.GetObject(prev))) { allEqual = false; break; }
                    }
                    if (allEqual) { isDup = true; break; }
                }
            }

            mask[r] = isDup;
            if (!isDup) { seen.Add(h); seenRows.Add(r); }
        }

        return mask;
    }

    /// <summary>
    /// Export rows as an array of object arrays.
    /// Usage: var records = df.ToRecords(); // records[0] = { "Alice", 25, 90.0 }
    /// </summary>
    public object?[][] ToRecords()
    {
        var records = new object?[RowCount][];
        for (int r = 0; r < RowCount; r++)
        {
            records[r] = new object?[ColumnCount];
            for (int c = 0; c < ColumnCount; c++)
                records[r][c] = _columns[c].GetObject(r);
        }
        return records;
    }

    // -- Display --

    // -- Conversion --

    /// <summary>
    /// Convert to a dictionary with various orient modes (like pandas to_dict).
    /// "list": {colName: [val1, val2, ...]}
    /// "records": [{col1: val1, col2: val2}, ...]
    /// "dict": {colName: {0: val1, 1: val2, ...}}
    /// "split": {columns: [...], data: [[...], ...], index: [...]}
    /// </summary>
    public Dictionary<string, object?> ToDictionary(string orient = "list")
    {
        return orient switch
        {
            "list" => ToDictList(),
            "dict" => ToDictDict(),
            "split" => ToDictSplit(),
            _ => throw new ArgumentException($"Unknown orient: '{orient}'. Supported: list, dict, split")
        };
    }

    /// <summary>Convert to a list of dictionaries (one per row). Like pandas to_dict('records').</summary>
    public List<Dictionary<string, object?>> ToRecordDicts()
    {
        var result = new List<Dictionary<string, object?>>(RowCount);
        for (int r = 0; r < RowCount; r++)
        {
            var row = new Dictionary<string, object?>(ColumnCount);
            for (int c = 0; c < ColumnCount; c++)
                row[ColumnNames[c]] = _columns[c].GetObject(r);
            result.Add(row);
        }
        return result;
    }

    private Dictionary<string, object?> ToDictList()
    {
        var result = new Dictionary<string, object?>(ColumnCount);
        foreach (var name in ColumnNames)
        {
            var col = this[name];
            var values = new object?[RowCount];
            for (int r = 0; r < RowCount; r++)
                values[r] = col.GetObject(r);
            result[name] = values;
        }
        return result;
    }

    private Dictionary<string, object?> ToDictDict()
    {
        var result = new Dictionary<string, object?>(ColumnCount);
        foreach (var name in ColumnNames)
        {
            var col = this[name];
            var dict = new Dictionary<int, object?>(RowCount);
            for (int r = 0; r < RowCount; r++)
                dict[r] = col.GetObject(r);
            result[name] = dict;
        }
        return result;
    }

    private Dictionary<string, object?> ToDictSplit()
    {
        var data = new object?[RowCount][];
        for (int r = 0; r < RowCount; r++)
        {
            data[r] = new object?[ColumnCount];
            for (int c = 0; c < ColumnCount; c++)
                data[r][c] = _columns[c].GetObject(r);
        }
        return new Dictionary<string, object?>
        {
            ["columns"] = ColumnNames.ToArray(),
            ["data"] = data,
            ["index"] = Enumerable.Range(0, RowCount).ToArray()
        };
    }

    // -- Display --

    public override string ToString() => ConsoleFormatter.Format(this);

    public string ToHtml(int maxRows = 50) => HtmlFormatter.Format(this, maxRows);

    public string ToMarkdown(int maxRows = 50) => MarkdownFormatter.Format(this, maxRows);

    /// <summary>
    /// Format as a LaTeX table. Useful for academic papers and documentation.
    /// </summary>
    public string ToLatex(int maxRows = 50)
    {
        var sb = new System.Text.StringBuilder();
        var cols = string.Join(" ", Enumerable.Repeat("r", ColumnCount));
        sb.AppendLine($"\\begin{{tabular}}{{{cols}}}");
        sb.AppendLine("\\toprule");
        sb.AppendLine(string.Join(" & ", ColumnNames) + " \\\\");
        sb.AppendLine("\\midrule");

        int rows = Math.Min(RowCount, maxRows);
        for (int r = 0; r < rows; r++)
        {
            var values = ColumnNames.Select(name =>
            {
                var val = this[name].GetObject(r);
                return val?.ToString() ?? "";
            });
            sb.AppendLine(string.Join(" & ", values) + " \\\\");
        }

        if (RowCount > maxRows)
            sb.AppendLine($"\\multicolumn{{{ColumnCount}}}{{c}}{{... {RowCount - maxRows} more rows ...}} \\\\");

        sb.AppendLine("\\bottomrule");
        sb.AppendLine("\\end{tabular}");
        return sb.ToString();
    }

    // -- IEnumerable --

    public IEnumerator<DataFrameRow> GetEnumerator()
    {
        for (int i = 0; i < RowCount; i++)
            yield return this[i];
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
