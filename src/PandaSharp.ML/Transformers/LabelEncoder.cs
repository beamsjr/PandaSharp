using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Map string categories to integer codes.</summary>
public class LabelEncoder : ITransformer
{
    private readonly string[] _columns;
    private Dictionary<string, Dictionary<string, int>>? _mappings;

    public string Name => "LabelEncoder";

    public LabelEncoder(params string[] columns) => _columns = columns;

    public ITransformer Fit(DataFrame df)
    {
        _mappings = new Dictionary<string, Dictionary<string, int>>();
        foreach (var name in _columns)
        {
            var col = df[name];
            var mapping = new Dictionary<string, int>();
            for (int i = 0; i < col.Length; i++)
            {
                var val = col.GetObject(i)?.ToString();
                if (val is not null && !mapping.ContainsKey(val))
                    mapping[val] = mapping.Count;
            }
            _mappings[name] = mapping;
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_mappings is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, mapping) in _mappings)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            var values = new int?[df.RowCount];
            for (int i = 0; i < df.RowCount; i++)
            {
                var val = col.GetObject(i)?.ToString();
                values[i] = val is not null && mapping.TryGetValue(val, out int code) ? code : null;
            }
            result = result.Assign(name, Column<int>.FromNullable(name, values));
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    /// <summary>Get the mapping for a column.</summary>
    public IReadOnlyDictionary<string, int>? GetMapping(string column) =>
        _mappings?.GetValueOrDefault(column);
}
