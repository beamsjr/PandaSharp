using Cortex;
using Cortex.Column;

namespace Cortex.ML.Transformers;

/// <summary>Strategy for handling unseen categories during Transform.</summary>
public enum UnknownCategoryHandling
{
    /// <summary>Produce all-zero rows (default, original behavior). Unseen and null are indistinguishable.</summary>
    Ignore,
    /// <summary>Throw an exception if an unseen category is encountered.</summary>
    Error,
    /// <summary>Add an extra indicator column "{col}_unknown" that is 1 for unseen categories.</summary>
    Indicator,
}

/// <summary>Expand categories to binary columns. Drops the original column.</summary>
public class OneHotEncoder : ITransformer
{
    private readonly string[] _columns;
    private readonly UnknownCategoryHandling _handleUnknown;
    private Dictionary<string, string[]>? _vocabularies;

    public string Name => "OneHotEncoder";

    public OneHotEncoder(params string[] columns)
        : this(UnknownCategoryHandling.Ignore, columns) { }

    public OneHotEncoder(UnknownCategoryHandling handleUnknown, params string[] columns)
    {
        _handleUnknown = handleUnknown;
        _columns = columns;
    }

    public ITransformer Fit(DataFrame df)
    {
        _vocabularies = new Dictionary<string, string[]>();
        foreach (var name in _columns)
        {
            var col = df[name];
            var unique = new List<string>();
            var seen = new HashSet<string>();
            for (int i = 0; i < col.Length; i++)
            {
                var val = col.GetObject(i)?.ToString();
                if (val is not null && seen.Add(val)) unique.Add(val);
            }
            _vocabularies[name] = unique.ToArray();
        }
        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_vocabularies is null) throw new InvalidOperationException("Call Fit() first.");
        var result = df;
        foreach (var (name, vocab) in _vocabularies)
        {
            if (!df.ColumnNames.Contains(name)) continue;
            var col = df[name];
            var vocabSet = new HashSet<string>(vocab);

            // Build category → row indices map (single pass)
            var catIndices = new Dictionary<string, List<int>>();
            foreach (var cat in vocab) catIndices[cat] = new List<int>();
            var unknownRows = new List<int>();

            for (int i = 0; i < df.RowCount; i++)
            {
                var val = col.GetObject(i)?.ToString();
                if (val is not null && catIndices.TryGetValue(val, out var list))
                {
                    list.Add(i);
                }
                else if (val is not null && !vocabSet.Contains(val))
                {
                    if (_handleUnknown == UnknownCategoryHandling.Error)
                        throw new InvalidOperationException(
                            $"Unseen category '{val}' in column '{name}'. " +
                            "Use UnknownCategoryHandling.Ignore or .Indicator to handle unseen categories.");
                    unknownRows.Add(i);
                }
            }

            // Drop original, add binary columns
            result = result.DropColumn(name);
            foreach (var cat in vocab)
            {
                var values = new int[df.RowCount]; // 0-initialized
                foreach (var idx in catIndices[cat])
                    values[idx] = 1;
                result = result.AddColumn(new Column<int>($"{name}_{cat}", values));
            }

            // Add indicator column for unseen categories
            if (_handleUnknown == UnknownCategoryHandling.Indicator)
            {
                var unknownValues = new int[df.RowCount];
                foreach (var idx in unknownRows)
                    unknownValues[idx] = 1;
                result = result.AddColumn(new Column<int>($"{name}_unknown", unknownValues));
            }
        }
        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);
}
