using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Transformers;

public enum VectorizerMode { Count, TfIdf }

/// <summary>
/// Vectorize text columns into numeric features using bag-of-words (count or TF-IDF).
/// </summary>
public class TextVectorizer : ITransformer
{
    private readonly string _column;
    private readonly VectorizerMode _mode;
    private readonly int _maxFeatures;
    private string[]? _vocabulary;
    private double[]? _idfWeights;

    public string Name => "TextVectorizer";

    public TextVectorizer(string column, VectorizerMode mode = VectorizerMode.TfIdf, int maxFeatures = 1000)
    {
        _column = column;
        _mode = mode;
        _maxFeatures = maxFeatures;
    }

    public ITransformer Fit(DataFrame df)
    {
        var col = df[_column];
        int nDocs = 0;

        // Build vocabulary: word → document frequency
        var wordDocFreq = new Dictionary<string, int>();
        var wordFreq = new Dictionary<string, int>();

        for (int i = 0; i < col.Length; i++)
        {
            var text = col.GetObject(i)?.ToString();
            if (text is null) continue;
            nDocs++;

            var wordsInDoc = new HashSet<string>();
            foreach (var word in Tokenize(text))
            {
                wordFreq[word] = wordFreq.GetValueOrDefault(word) + 1;
                wordsInDoc.Add(word);
            }
            foreach (var word in wordsInDoc)
                wordDocFreq[word] = wordDocFreq.GetValueOrDefault(word) + 1;
        }

        // Select top features by frequency
        _vocabulary = wordFreq
            .OrderByDescending(kv => kv.Value)
            .Take(_maxFeatures)
            .Select(kv => kv.Key)
            .OrderBy(w => w) // alphabetical for consistency
            .ToArray();

        // Compute IDF weights: log(N / df)
        if (_mode == VectorizerMode.TfIdf)
        {
            _idfWeights = new double[_vocabulary.Length];
            for (int v = 0; v < _vocabulary.Length; v++)
            {
                int docFreq = wordDocFreq.GetValueOrDefault(_vocabulary[v], 0);
                _idfWeights[v] = Math.Log((double)(nDocs + 1) / (docFreq + 1)) + 1; // smoothed IDF
            }
        }

        return this;
    }

    public DataFrame Transform(DataFrame df)
    {
        if (_vocabulary is null) throw new InvalidOperationException("Call Fit() first.");

        var col = df[_column];
        var vocabIndex = new Dictionary<string, int>();
        for (int v = 0; v < _vocabulary.Length; v++)
            vocabIndex[_vocabulary[v]] = v;

        // Build feature matrix
        var features = new double[df.RowCount * _vocabulary.Length];

        for (int r = 0; r < df.RowCount; r++)
        {
            var text = col.GetObject(r)?.ToString();
            if (text is null) continue;

            // Count term frequencies
            var termCounts = new Dictionary<string, int>();
            foreach (var word in Tokenize(text))
                termCounts[word] = termCounts.GetValueOrDefault(word) + 1;

            foreach (var (word, count) in termCounts)
            {
                if (vocabIndex.TryGetValue(word, out int idx))
                {
                    double value = count;
                    if (_mode == VectorizerMode.TfIdf)
                        value *= _idfWeights![idx];
                    features[r * _vocabulary.Length + idx] = value;
                }
            }
        }

        // Build output DataFrame: drop original column, add feature columns
        var result = df.DropColumn(_column);
        for (int v = 0; v < _vocabulary.Length; v++)
        {
            var colData = new double[df.RowCount];
            for (int r = 0; r < df.RowCount; r++)
                colData[r] = features[r * _vocabulary.Length + v];
            result = result.AddColumn(new Column<double>($"{_column}_{_vocabulary[v]}", colData));
        }

        return result;
    }

    public DataFrame FitTransform(DataFrame df) => Fit(df).Transform(df);

    public IReadOnlyList<string>? Vocabulary => _vocabulary;

    private static IEnumerable<string> Tokenize(string text)
    {
        // Simple whitespace + punctuation tokenizer, lowercased
        var sb = new System.Text.StringBuilder();
        foreach (char c in text)
        {
            if (char.IsLetterOrDigit(c))
                sb.Append(char.ToLowerInvariant(c));
            else if (sb.Length > 0)
            {
                yield return sb.ToString();
                sb.Clear();
            }
        }
        if (sb.Length > 0) yield return sb.ToString();
    }
}
