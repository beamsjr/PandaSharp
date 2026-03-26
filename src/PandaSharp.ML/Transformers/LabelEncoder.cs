using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

/// <summary>Map string categories to integer codes.</summary>
public class LabelEncoder : ITransformer
{
    private readonly string[] _columns;
    private readonly UnknownCategoryHandling _handleUnknown;
    private Dictionary<string, Dictionary<string, int>>? _mappings;

    public string Name => "LabelEncoder";

    /// <summary>Gets or sets how unseen categories are handled during Transform.</summary>
    public UnknownCategoryHandling HandleUnknown => _handleUnknown;

    public LabelEncoder(params string[] columns)
        : this(UnknownCategoryHandling.Ignore, columns) { }

    /// <summary>
    /// Create a LabelEncoder with configurable unknown-category handling.
    /// </summary>
    /// <param name="handleUnknown">Strategy for unseen categories during Transform.</param>
    /// <param name="columns">Columns to encode.</param>
    public LabelEncoder(UnknownCategoryHandling handleUnknown, params string[] columns)
    {
        _handleUnknown = handleUnknown;
        _columns = columns;
    }

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
            int indicatorCode = mapping.Count; // code for unknown indicator

            for (int i = 0; i < df.RowCount; i++)
            {
                var val = col.GetObject(i)?.ToString();
                if (val is not null && mapping.TryGetValue(val, out int code))
                {
                    values[i] = code;
                }
                else if (val is not null)
                {
                    // Unseen category
                    switch (_handleUnknown)
                    {
                        case UnknownCategoryHandling.Error:
                            throw new InvalidOperationException(
                                $"Unseen category '{val}' in column '{name}'. " +
                                "Use UnknownCategoryHandling.Ignore or .Indicator to handle unseen categories.");
                        case UnknownCategoryHandling.Indicator:
                            values[i] = indicatorCode;
                            break;
                        case UnknownCategoryHandling.Ignore:
                        default:
                            values[i] = null;
                            break;
                    }
                }
                else
                {
                    values[i] = null;
                }
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
