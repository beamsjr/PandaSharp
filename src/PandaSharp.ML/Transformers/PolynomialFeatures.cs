using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Generate polynomial and interaction features from numeric columns.</summary>
public class PolynomialFeatures : ITransformer
{
    private readonly int _degree;
    private readonly string[] _columns;
    private string[]? _inputColumns;

    public string Name => "PolynomialFeatures";

    public PolynomialFeatures(int degree = 2, params string[] columns)
    {
        _degree = degree;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _inputColumns = _columns.Length > 0 ? _columns
            : df.ColumnNames.Where(n => IsNumeric(df[n].DataType)).ToArray();
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_inputColumns is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;

        // Add polynomial terms (x², x³, ...)
        foreach (var col in _inputColumns)
        {
            if (!df.ColumnNames.Contains(col)) continue;
            var srcCol = df[col];
            for (int d = 2; d <= _degree; d++)
            {
                var values = new double[df.RowCount];
                for (int i = 0; i < df.RowCount; i++)
                {
                    if (srcCol.IsNull(i)) { values[i] = double.NaN; continue; }
                    double v = TypeHelpers.GetDouble(srcCol, i);
                    values[i] = Math.Pow(v, d);
                }
                result = result.AddColumn(new Column<double>($"{col}^{d}", values));
            }
        }

        // Add interaction terms (x1*x2) — reads from original df, not result.
        // This is correct: interactions use original feature values, not polynomial terms.
        if (_degree >= 2)
        {
            for (int i = 0; i < _inputColumns.Length; i++)
            {
                for (int j = i + 1; j < _inputColumns.Length; j++)
                {
                    var colA = df[_inputColumns[i]];
                    var colB = df[_inputColumns[j]];
                    var values = new double[df.RowCount];
                    for (int r = 0; r < df.RowCount; r++)
                    {
                        if (colA.IsNull(r) || colB.IsNull(r)) { values[r] = double.NaN; continue; }
                        values[r] = TypeHelpers.GetDouble(colA, r) * TypeHelpers.GetDouble(colB, r);
                    }
                    result = result.AddColumn(new Column<double>($"{_inputColumns[i]}*{_inputColumns[j]}", values));
                }
            }
        }

        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
    private static bool IsNumeric(Type t) => t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
