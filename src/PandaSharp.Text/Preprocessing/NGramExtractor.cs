using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;

namespace PandaSharp.Text.Preprocessing;

/// <summary>
/// Extracts n-grams from text data. Implements the <see cref="ITransformer"/> pattern.
/// Supports configurable n-gram ranges (e.g., unigrams, bigrams, trigrams, or combinations).
/// </summary>
public sealed class NGramExtractor : ITransformer
{
    private readonly int _minN;
    private readonly int _maxN;
    private string _inputColumn = "text";
    private Dictionary<string, int>? _vocabulary;

    /// <inheritdoc />
    public string Name => "NGramExtractor";

    /// <summary>
    /// Creates an n-gram extractor for a specific n-gram size.
    /// </summary>
    /// <param name="n">N-gram size (1=unigram, 2=bigram, 3=trigram).</param>
    public NGramExtractor(int n) : this(n, n) { }

    /// <summary>
    /// Creates an n-gram extractor for a range of sizes.
    /// </summary>
    /// <param name="minN">Minimum n-gram size (inclusive).</param>
    /// <param name="maxN">Maximum n-gram size (inclusive).</param>
    public NGramExtractor(int minN, int maxN)
    {
        if (minN < 1) throw new ArgumentOutOfRangeException(nameof(minN), "Minimum n must be at least 1.");
        if (maxN < minN) throw new ArgumentOutOfRangeException(nameof(maxN), "Maximum n must be >= minimum n.");
        _minN = minN;
        _maxN = maxN;
    }

    /// <summary>
    /// Set the input column name to process.
    /// </summary>
    /// <param name="column">Column name containing text.</param>
    /// <returns>This instance for chaining.</returns>
    public NGramExtractor WithColumn(string column)
    {
        ArgumentNullException.ThrowIfNull(column);
        _inputColumn = column;
        return this;
    }

    /// <summary>
    /// Extract n-grams from a text string.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Array of n-gram strings.</returns>
    public string[] Extract(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var ngrams = new List<string>();

        for (int n = _minN; n <= _maxN; n++)
        {
            for (int i = 0; i <= words.Length - n; i++)
            {
                var ngram = string.Join(' ', words, i, n);
                ngrams.Add(ngram);
            }
        }

        return ngrams.ToArray();
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df)
    {
        var col = (StringColumn)df[_inputColumn];
        int len = col.Length;
        var vocab = new Dictionary<string, int>(StringComparer.Ordinal);

        for (int i = 0; i < len; i++)
        {
            var val = col[i];
            if (val is null) continue;
            var ngrams = Extract(val);
            foreach (var ng in ngrams)
            {
                if (!vocab.ContainsKey(ng))
                    vocab[ng] = vocab.Count;
            }
        }

        _vocabulary = vocab;
        return this;
    }

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        if (_vocabulary is null)
            throw new InvalidOperationException("Call Fit() before Transform(), or use FitTransform().");

        var col = (StringColumn)df[_inputColumn];
        int len = col.Length;
        int vocabCount = _vocabulary.Count;

        // Build per-column arrays directly (sparse-friendly: only touch non-zero entries)
        var colArrays = new int[vocabCount][];
        for (int v = 0; v < vocabCount; v++)
            colArrays[v] = new int[len];

        for (int i = 0; i < len; i++)
        {
            var val = col[i];
            if (val is null) continue;
            var ngrams = Extract(val);
            foreach (var ng in ngrams)
            {
                if (_vocabulary.TryGetValue(ng, out int idx))
                    colArrays[idx][i]++;
            }
        }

        // Add n-gram count columns to DataFrame
        var columns = new List<IColumn>();
        foreach (var existingCol in df.ColumnNames)
            columns.Add(df[existingCol]);

        foreach (var (ngram, idx) in _vocabulary.OrderBy(kv => kv.Value))
        {
            columns.Add(new Column<int>($"ngram_{ngram}", colArrays[idx]));
        }

        return new DataFrame(columns);
    }
}
