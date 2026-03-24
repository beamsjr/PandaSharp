using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Z-score normalization: (x - mean) / std per column.</summary>
public class StandardScaler : ITransformer
{
    private readonly string[] _columns;
    private Dictionary<string, (double Mean, double Std)>? _params;

    public string Name => "StandardScaler";

    public StandardScaler(params string[] columns) => _columns = columns;

    public ITransformer Fit(DataFrame df)
    {
        _params = new Dictionary<string, (double, double)>();
        var cols = _columns.Length > 0 ? _columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        foreach (var name in cols)
        {
            var col = df[name];
            double mean = 0, m2 = 0;
            int count = 0;
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                double val = TypeHelpers.GetDouble(col, i);
                count++;
                double delta = val - mean;
                mean += delta / count;
                m2 += delta * (val - mean);
            }
            double std = count > 1 ? Math.Sqrt(m2 / (count - 1)) : 1.0;
            _params[name] = (mean, std == 0 ? 1.0 : std);
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_params is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, (mean, std)) in _params)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            var values = new double[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
                values[i] = col.IsNull(i) ? double.NaN : (TypeHelpers.GetDouble(col, i) - mean) / std;
            result = result.Assign(name, new Column<double>(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    private static bool IsNumeric(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
