using Cortex;
using Cortex.Column;

namespace Cortex.ML.Transformers;

/// <summary>Scale using median and IQR — robust to outliers.</summary>
public class RobustScaler : ITransformer
{
    private readonly string[] _columns;
    private Dictionary<string, (double Median, double IQR)>? _params;

    public string Name => "RobustScaler";

    public RobustScaler(params string[] columns) => _columns = columns;

    public ITransformer Fit(DataFrame df)
    {
        _params = new Dictionary<string, (double, double)>();
        var cols = _columns.Length > 0 ? _columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        foreach (var name in cols)
        {
            var values = GetSortedValues(df[name]);
            if (values.Length == 0) { _params[name] = (0, 1); continue; }
            double median = Percentile(values, 0.5);
            double q1 = Percentile(values, 0.25);
            double q3 = Percentile(values, 0.75);
            double iqr = q3 - q1;
            _params[name] = (median, iqr == 0 ? 1.0 : iqr);
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_params is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, (median, iqr)) in _params)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            var values = new double[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
                values[i] = col.IsNull(i) ? double.NaN : (TypeHelpers.GetDouble(col, i) - median) / iqr;
            result = result.Assign(name, new Column<double>(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    private static double[] GetSortedValues(IColumn col)
    {
        var list = new List<double>();
        for (int i = 0; i < col.Length; i++)
            if (!col.IsNull(i)) list.Add(TypeHelpers.GetDouble(col, i));
        list.Sort();
        return list.ToArray();
    }

    private static double Percentile(double[] sorted, double p)
    {
        double pos = p * (sorted.Length - 1);
        int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
        return lo == hi ? sorted[lo] : sorted[lo] * (1 - (pos - lo)) + sorted[hi] * (pos - lo);
    }

    private static bool IsNumeric(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
