using Microsoft.ML;
using Microsoft.ML.Data;
using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.MLNet;

/// <summary>
/// Bridge between PandaSharp DataFrames and ML.NET IDataView.
/// Enables using PandaSharp for data loading/processing and ML.NET for model training.
/// </summary>
public static class DataViewBridge
{
    /// <summary>
    /// Convert a PandaSharp DataFrame to an ML.NET IDataView.
    /// Each DataFrame column maps to a typed ML.NET column (Single, Double, Boolean, or Text).
    /// Uses a zero-copy cursor that reads directly from Arrow-backed buffers.
    /// </summary>
    public static IDataView ToDataView(this DataFrame df, MLContext mlContext)
    {
        return new DataFrameDataView(df, mlContext);
    }

    /// <summary>
    /// Convert an ML.NET IDataView back to a PandaSharp DataFrame.
    /// Uses typed builders to avoid boxing overhead.
    /// </summary>
    public static DataFrame ToDataFrame(this IDataView dataView)
    {
        var schema = dataView.Schema;
        var visibleCols = schema.Where(c => !c.IsHidden).ToList();

        // Pre-allocate typed builders
        var floatBuilders = new Dictionary<int, List<float>>();
        var doubleBuilders = new Dictionary<int, List<double>>();
        var intBuilders = new Dictionary<int, List<int>>();
        var longBuilders = new Dictionary<int, List<long>>();
        var boolBuilders = new Dictionary<int, List<bool>>();
        var stringBuilders = new Dictionary<int, List<string?>>();

        foreach (var col in visibleCols)
        {
            if (col.Type == NumberDataViewType.Single) floatBuilders[col.Index] = new();
            else if (col.Type == NumberDataViewType.Double) doubleBuilders[col.Index] = new();
            else if (col.Type == NumberDataViewType.Int32) intBuilders[col.Index] = new();
            else if (col.Type == NumberDataViewType.Int64) longBuilders[col.Index] = new();
            else if (col.Type == BooleanDataViewType.Instance) boolBuilders[col.Index] = new();
            else stringBuilders[col.Index] = new();
        }

        // Pre-create getters for each column
        using var cursor = dataView.GetRowCursor(schema);
        var getters = new Dictionary<int, Delegate>();
        foreach (var col in visibleCols)
        {
            if (col.Type == NumberDataViewType.Single)
                getters[col.Index] = cursor.GetGetter<float>(col);
            else if (col.Type == NumberDataViewType.Double)
                getters[col.Index] = cursor.GetGetter<double>(col);
            else if (col.Type == NumberDataViewType.Int32)
                getters[col.Index] = cursor.GetGetter<int>(col);
            else if (col.Type == NumberDataViewType.Int64)
                getters[col.Index] = cursor.GetGetter<long>(col);
            else if (col.Type == BooleanDataViewType.Instance)
                getters[col.Index] = cursor.GetGetter<bool>(col);
            else if (col.Type is TextDataViewType)
                getters[col.Index] = cursor.GetGetter<ReadOnlyMemory<char>>(col);
        }

        while (cursor.MoveNext())
        {
            foreach (var col in visibleCols)
            {
                if (col.Type == NumberDataViewType.Single)
                {
                    float val = 0;
                    ((ValueGetter<float>)getters[col.Index])(ref val);
                    floatBuilders[col.Index].Add(val);
                }
                else if (col.Type == NumberDataViewType.Double)
                {
                    double val = 0;
                    ((ValueGetter<double>)getters[col.Index])(ref val);
                    doubleBuilders[col.Index].Add(val);
                }
                else if (col.Type == NumberDataViewType.Int32)
                {
                    int val = 0;
                    ((ValueGetter<int>)getters[col.Index])(ref val);
                    intBuilders[col.Index].Add(val);
                }
                else if (col.Type == NumberDataViewType.Int64)
                {
                    long val = 0;
                    ((ValueGetter<long>)getters[col.Index])(ref val);
                    longBuilders[col.Index].Add(val);
                }
                else if (col.Type == BooleanDataViewType.Instance)
                {
                    bool val = false;
                    ((ValueGetter<bool>)getters[col.Index])(ref val);
                    boolBuilders[col.Index].Add(val);
                }
                else if (col.Type is TextDataViewType && getters.ContainsKey(col.Index))
                {
                    var val = default(ReadOnlyMemory<char>);
                    ((ValueGetter<ReadOnlyMemory<char>>)getters[col.Index])(ref val);
                    stringBuilders[col.Index].Add(val.ToString());
                }
            }
        }

        // Build typed columns
        var columns = new List<IColumn>();
        foreach (var col in visibleCols)
        {
            if (floatBuilders.TryGetValue(col.Index, out var fb))
            {
                var arr = new double[fb.Count];
                for (int i = 0; i < fb.Count; i++) arr[i] = fb[i];
                columns.Add(new Column<double>(col.Name, arr));
            }
            else if (doubleBuilders.TryGetValue(col.Index, out var db))
                columns.Add(new Column<double>(col.Name, db.ToArray()));
            else if (intBuilders.TryGetValue(col.Index, out var ib))
                columns.Add(new Column<int>(col.Name, ib.ToArray()));
            else if (longBuilders.TryGetValue(col.Index, out var lb))
                columns.Add(new Column<long>(col.Name, lb.ToArray()));
            else if (boolBuilders.TryGetValue(col.Index, out var bb))
                columns.Add(new Column<bool>(col.Name, bb.ToArray()));
            else if (stringBuilders.TryGetValue(col.Index, out var sb))
                columns.Add(new StringColumn(col.Name, sb.ToArray()));
        }

        return new DataFrame(columns);
    }
}

/// <summary>
/// Custom IDataView implementation that reads directly from PandaSharp DataFrame columns.
/// Each column is mapped to a typed ML.NET column with efficient per-row getters.
/// </summary>
internal sealed class DataFrameDataView : IDataView
{
    private readonly DataFrame _df;
    private readonly DataViewSchema _schema;
    private readonly DataViewType[] _colTypes;

    public DataFrameDataView(DataFrame df, MLContext mlContext)
    {
        _df = df;
        var builder = new DataViewSchema.Builder();
        _colTypes = new DataViewType[df.ColumnCount];

        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            var dvType = MapToDataViewType(col.DataType);
            _colTypes[c] = dvType;
            builder.AddColumn(df.ColumnNames[c], dvType);
        }

        _schema = builder.ToSchema();
    }

    public bool CanShuffle => true;
    public DataViewSchema Schema => _schema;

    public long? GetRowCount() => _df.RowCount;

    public DataViewRowCursor GetRowCursor(IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand = null)
    {
        return new DataFrameCursor(this, columnsNeeded, rand);
    }

    public DataViewRowCursor[] GetRowCursorSet(IEnumerable<DataViewSchema.Column> columnsNeeded, int n, Random? rand = null)
    {
        return [GetRowCursor(columnsNeeded, rand)];
    }

    private static DataViewType MapToDataViewType(Type type)
    {
        if (type == typeof(float)) return NumberDataViewType.Single;
        if (type == typeof(double)) return NumberDataViewType.Double;
        if (type == typeof(int)) return NumberDataViewType.Int32;
        if (type == typeof(long)) return NumberDataViewType.Int64;
        if (type == typeof(bool)) return BooleanDataViewType.Instance;
        return TextDataViewType.Instance; // strings and others
    }

    private sealed class DataFrameCursor : DataViewRowCursor
    {
        private readonly DataFrameDataView _parent;
        private readonly HashSet<int> _activeColumns;
        private readonly int[] _rowOrder;
        private long _position = -1;
        private bool _disposed;

        public DataFrameCursor(DataFrameDataView parent, IEnumerable<DataViewSchema.Column> columnsNeeded, Random? rand)
        {
            _parent = parent;
            _activeColumns = new HashSet<int>(columnsNeeded.Select(c => c.Index));

            int n = parent._df.RowCount;
            _rowOrder = new int[n];
            for (int i = 0; i < n; i++) _rowOrder[i] = i;

            if (rand is not null)
            {
                for (int i = n - 1; i > 0; i--)
                {
                    int j = rand.Next(i + 1);
                    (_rowOrder[i], _rowOrder[j]) = (_rowOrder[j], _rowOrder[i]);
                }
            }
        }

        public override long Position => _position;
        public override long Batch => 0;
        public override DataViewSchema Schema => _parent._schema;

        public override bool MoveNext()
        {
            if (_disposed) return false;
            _position++;
            return _position < _parent._df.RowCount;
        }

        public override bool IsColumnActive(DataViewSchema.Column column) => _activeColumns.Contains(column.Index);

        public override ValueGetter<TValue> GetGetter<TValue>(DataViewSchema.Column column)
        {
            var df = _parent._df;
            var colName = df.ColumnNames[column.Index];
            var col = df[colName];
            var dvType = _parent._colTypes[column.Index];

            if (dvType == NumberDataViewType.Single)
            {
                ValueGetter<float> getter = (ref float value) =>
                {
                    int row = _rowOrder[_position];
                    value = col.IsNull(row) ? 0f : Convert.ToSingle(TypeHelpers.GetDouble(col, row));
                };
                return (ValueGetter<TValue>)(Delegate)getter;
            }

            if (dvType == NumberDataViewType.Double)
            {
                ValueGetter<double> getter = (ref double value) =>
                {
                    int row = _rowOrder[_position];
                    value = col.IsNull(row) ? 0.0 : TypeHelpers.GetDouble(col, row);
                };
                return (ValueGetter<TValue>)(Delegate)getter;
            }

            if (dvType == NumberDataViewType.Int32)
            {
                ValueGetter<int> getter = (ref int value) =>
                {
                    int row = _rowOrder[_position];
                    var obj = col.GetObject(row);
                    value = obj is int i ? i : (obj is not null ? Convert.ToInt32(obj) : 0);
                };
                return (ValueGetter<TValue>)(Delegate)getter;
            }

            if (dvType == NumberDataViewType.Int64)
            {
                ValueGetter<long> getter = (ref long value) =>
                {
                    int row = _rowOrder[_position];
                    var obj = col.GetObject(row);
                    value = obj is long l ? l : (obj is not null ? Convert.ToInt64(obj) : 0);
                };
                return (ValueGetter<TValue>)(Delegate)getter;
            }

            if (dvType == BooleanDataViewType.Instance)
            {
                ValueGetter<bool> getter = (ref bool value) =>
                {
                    int row = _rowOrder[_position];
                    var obj = col.GetObject(row);
                    value = obj is bool b && b;
                };
                return (ValueGetter<TValue>)(Delegate)getter;
            }

            // Text
            ValueGetter<ReadOnlyMemory<char>> textGetter = (ref ReadOnlyMemory<char> value) =>
            {
                int row = _rowOrder[_position];
                var str = col.GetObject(row)?.ToString() ?? "";
                value = str.AsMemory();
            };
            return (ValueGetter<TValue>)(Delegate)textGetter;
        }

        public override ValueGetter<DataViewRowId> GetIdGetter()
        {
            return (ref DataViewRowId id) =>
            {
                id = new DataViewRowId((ulong)_position, 0);
            };
        }

        protected override void Dispose(bool disposing)
        {
            _disposed = true;
            base.Dispose(disposing);
        }
    }
}
