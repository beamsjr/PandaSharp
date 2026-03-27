using Cortex;
using Cortex.Column;

namespace Cortex.ML.Transformers;

public enum BinStrategy { Uniform, Quantile }

/// <summary>Bin continuous values into discrete categories.</summary>
public class Discretizer : ITransformer
{
    private readonly int _nBins;
    private readonly BinStrategy _strategy;
    private readonly string[] _columns;
    private Dictionary<string, double[]>? _binEdges;

    public string Name => "Discretizer";

    public Discretizer(int nBins = 5, BinStrategy strategy = BinStrategy.Uniform, params string[] columns)
    {
        _nBins = nBins;
        _strategy = strategy;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _binEdges = new Dictionary<string, double[]>();
        var cols = _columns.Length > 0 ? _columns : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();

        foreach (var name in cols)
        {
            var values = new List<double>();
            var col = df[name];
            for (int i = 0; i < col.Length; i++)
                if (!col.IsNull(i)) values.Add(TypeHelpers.GetDouble(col, i));
            values.Sort();

            if (values.Count == 0) { _binEdges[name] = []; continue; }

            var edges = new double[_nBins + 1];
            if (_strategy == BinStrategy.Uniform)
            {
                double min = values[0], max = values[^1];
                for (int b = 0; b <= _nBins; b++)
                    edges[b] = min + (max - min) * b / _nBins;
            }
            else // Quantile
            {
                for (int b = 0; b <= _nBins; b++)
                {
                    double p = (double)b / _nBins;
                    double pos = p * (values.Count - 1);
                    int lo = (int)Math.Floor(pos), hi = (int)Math.Ceiling(pos);
                    edges[b] = lo == hi ? values[lo] : values[lo] + (pos - lo) * (values[hi] - values[lo]);
                }
            }
            _binEdges[name] = edges;
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_binEdges is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, edges) in _binEdges)
        {
            if (!df.ColumnNames.Contains(name) || edges.Length == 0) continue;
            var col = df[name];
            var values = new int?[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
            {
                if (col.IsNull(i)) continue;
                double val = TypeHelpers.GetDouble(col, i);
                int bin = Array.BinarySearch(edges, val);
                if (bin < 0) bin = ~bin - 1;
                values[i] = Math.Clamp(bin, 0, _nBins - 1);
            }
            result = result.Assign(name, Column<int>.FromNullable(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
    private static bool IsNumeric(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
