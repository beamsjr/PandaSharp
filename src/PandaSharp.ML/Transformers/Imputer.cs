using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

public enum ImputeStrategy { Mean, Median, Mode, Constant }

/// <summary>Fill missing values using a learned strategy.</summary>
public class Imputer : ITransformer
{
    private readonly ImputeStrategy _strategy;
    private readonly double _constantValue;
    private readonly string[] _columns;
    private Dictionary<string, double>? _fillValues;

    public string Name => "Imputer";

    public Imputer(ImputeStrategy strategy = ImputeStrategy.Mean, double constantValue = 0, params string[] columns)
    {
        _strategy = strategy;
        _constantValue = constantValue;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _fillValues = new Dictionary<string, double>();
        var cols = _columns.Length > 0 ? _columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();

        foreach (var name in cols)
        {
            var col = df[name];
            var values = new List<double>();
            for (int i = 0; i < col.Length; i++)
                if (!col.IsNull(i)) values.Add(TypeHelpers.GetDouble(col, i));

            if (values.Count == 0) { _fillValues[name] = _constantValue; continue; }

            _fillValues[name] = _strategy switch
            {
                ImputeStrategy.Mean => values.Average(),
                ImputeStrategy.Median => ComputeMedian(values),
                ImputeStrategy.Mode => ComputeMode(values),
                ImputeStrategy.Constant => _constantValue,
                _ => _constantValue
            };
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_fillValues is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, fillVal) in _fillValues)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            if (col.NullCount == 0) continue;

            var values = new double[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
                values[i] = col.IsNull(i) ? fillVal : TypeHelpers.GetDouble(col, i);
            result = result.Assign(name, new Column<double>(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    private static double ComputeMedian(List<double> values)
    {
        var sorted = new List<double>(values);
        sorted.Sort();
        values = sorted;
        int mid = values.Count / 2;
        return values.Count % 2 == 0 ? (values[mid - 1] + values[mid]) / 2 : values[mid];
    }

    private static double ComputeMode(List<double> values)
    {
        return values.GroupBy(v => v).OrderByDescending(g => g.Count()).First().Key;
    }

    private static bool IsNumeric(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
