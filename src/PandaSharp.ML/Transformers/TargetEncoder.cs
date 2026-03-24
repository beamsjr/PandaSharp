using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>
/// Encode categories by target mean with additive smoothing.
/// Smoothing blends category mean toward global mean based on category count.
/// </summary>
public class TargetEncoder : ITransformer
{
    private readonly string[] _columns;
    private readonly string _target;
    private readonly double _smoothing;
    private Dictionary<string, Dictionary<string, double>>? _encodings;
    private double _globalMean;

    public string Name => "TargetEncoder";

    public TargetEncoder(string target, double smoothing = 10.0, params string[] columns)
    {
        _target = target;
        _smoothing = smoothing;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _encodings = new Dictionary<string, Dictionary<string, double>>();
        var targetCol = df[_target];

        // Compute global mean
        double globalSum = 0; int globalCount = 0;
        for (int i = 0; i < targetCol.Length; i++)
        {
            if (!targetCol.IsNull(i))
            {
                globalSum += TypeHelpers.GetDouble(targetCol, i);
                globalCount++;
            }
        }
        _globalMean = globalCount > 0 ? globalSum / globalCount : 0;

        foreach (var colName in _columns)
        {
            var col = df[colName];
            var catSums = new Dictionary<string, double>();
            var catCounts = new Dictionary<string, int>();

            for (int i = 0; i < df.RowCount; i++)
            {
                var cat = col.GetObject(i)?.ToString();
                if (cat is null || targetCol.IsNull(i)) continue;
                double val = TypeHelpers.GetDouble(targetCol, i);
                catSums[cat] = catSums.GetValueOrDefault(cat) + val;
                catCounts[cat] = catCounts.GetValueOrDefault(cat) + 1;
            }

            var encoding = new Dictionary<string, double>();
            foreach (var (cat, sum) in catSums)
            {
                int count = catCounts[cat];
                double catMean = sum / count;
                // Smoothed encoding: blend category mean toward global mean
                encoding[cat] = (count * catMean + _smoothing * _globalMean) / (count + _smoothing);
            }
            _encodings[colName] = encoding;
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_encodings is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (colName, encoding) in _encodings)
        {
            if (!df.ColumnNames.Contains(colName)) continue;
            var col = df[colName];
            var values = new double[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
            {
                var cat = col.GetObject(i)?.ToString();
                values[i] = cat is not null && encoding.TryGetValue(cat, out double v) ? v : _globalMean;
            }
            result = result.Assign(colName, new Column<double>(colName, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
