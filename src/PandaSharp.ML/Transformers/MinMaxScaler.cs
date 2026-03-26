using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Scale to [0, 1] range: (x - min) / (max - min).</summary>
public class MinMaxScaler : ITransformer
{
    private readonly string[] _columns;
    private Dictionary<string, (double Min, double Max)>? _params;

    public string Name => "MinMaxScaler";

    public MinMaxScaler(params string[] columns) => _columns = columns;

    public ITransformer Fit(DataFrame df)
    {
        _params = new Dictionary<string, (double, double)>();
        var cols = _columns.Length > 0 ? _columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        foreach (var name in cols)
        {
            double min = double.MaxValue, max = double.MinValue;
            var col = df[name];
            bool hasValues = false;
            for (int i = 0; i < col.Length; i++)
            {
                if (col.IsNull(i)) continue;
                hasValues = true;
                double val = TypeHelpers.GetDouble(col, i);
                if (val < min) min = val;
                if (val > max) max = val;
            }
            // Handle all-null columns: treat as zero range so Transform produces zeros
            if (!hasValues)
            {
                _params[name] = (0, 1);
            }
            else
            {
                _params[name] = (min, max == min ? min + 1 : max);
            }
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_params is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, (min, max)) in _params)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            double range = max - min;
            var values = new double[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
                values[i] = col.IsNull(i) ? double.NaN : (TypeHelpers.GetDouble(col, i) - min) / range;
            result = result.Assign(name, new Column<double>(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
    private static bool IsNumeric(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
