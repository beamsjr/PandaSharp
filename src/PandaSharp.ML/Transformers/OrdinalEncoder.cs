using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Encode categories with explicit ordering.</summary>
public class OrdinalEncoder : ITransformer
{
    private readonly Dictionary<string, string[]> _orderings;

    public string Name => "OrdinalEncoder";

    /// <summary>Provide explicit category orderings per column.</summary>
    public OrdinalEncoder(Dictionary<string, string[]> orderings) => _orderings = orderings;

    public ITransformer Fit(DataFrame df) => this; // ordering is provided, no fitting needed

    public DataFrame Transform(DataFrame df)
    {
        var result = df;
        foreach (var (colName, ordering) in _orderings)
        {
            if (!df.ColumnNames.Contains(colName)) continue;
            var map = new Dictionary<string, int>();
            for (int i = 0; i < ordering.Length; i++) map[ordering[i]] = i;

            var col = df[colName];
            var values = new int?[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
            {
                var val = col.GetObject(i)?.ToString();
                values[i] = val is not null && map.TryGetValue(val, out int code) ? code : null;
            }
            result = result.Assign(colName, Column<int>.FromNullable(colName, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Transform(df);
}
