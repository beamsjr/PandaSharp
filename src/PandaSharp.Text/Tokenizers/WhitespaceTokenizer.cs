namespace PandaSharp.Text.Tokenizers;

/// <summary>
/// Tokenizer that splits text on whitespace boundaries.
/// Builds a vocabulary from a training corpus and maps tokens to integer IDs.
/// </summary>
public sealed class WhitespaceTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _reverseVocab = new();
    private readonly bool _lowercase;
    private int _nextId;

    /// <summary>Unknown token string.</summary>
    public const string UnkToken = "[UNK]";

    /// <summary>Current vocabulary size.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Creates a new whitespace tokenizer.
    /// </summary>
    /// <param name="lowercase">Whether to lowercase all tokens before lookup.</param>
    public WhitespaceTokenizer(bool lowercase = false)
    {
        _lowercase = lowercase;
        AddToken(UnkToken);
    }

    /// <summary>
    /// Build vocabulary from a training corpus.
    /// </summary>
    /// <param name="corpus">Array of text documents.</param>
    public void Train(string[] corpus)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        for (int i = 0; i < corpus.Length; i++)
        {
            var text = corpus[i];
            if (_lowercase) text = text.ToLowerInvariant();
            foreach (var token in text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries))
            {
                if (!_vocab.ContainsKey(token))
                    AddToken(token);
            }
        }
    }

    /// <inheritdoc />
    public TokenizerResult Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_lowercase) text = text.ToLowerInvariant();
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var ids = new int[words.Length];
        var mask = new int[words.Length];

        int unkId = _vocab.GetValueOrDefault(UnkToken, 0);
        for (int i = 0; i < words.Length; i++)
        {
            ids[i] = _vocab.TryGetValue(words[i], out int id) ? id : unkId;
            mask[i] = 1;
        }

        return new TokenizerResult(ids, mask);
    }

    /// <inheritdoc />
    public string Decode(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var tokens = new string[ids.Length];
        for (int i = 0; i < ids.Length; i++)
        {
            tokens[i] = _reverseVocab.TryGetValue(ids[i], out var token) ? token : UnkToken;
        }
        return string.Join(' ', tokens);
    }

    private void AddToken(string token)
    {
        int id = _nextId++;
        _vocab[token] = id;
        _reverseVocab[id] = token;
    }
}
