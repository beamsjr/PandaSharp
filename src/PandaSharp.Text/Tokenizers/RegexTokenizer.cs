using System.Text.RegularExpressions;

namespace PandaSharp.Text.Tokenizers;

/// <summary>
/// Tokenizer that splits text using a configurable regular expression pattern.
/// Default pattern matches word characters (<c>\w+</c>), which handles punctuation separation.
/// </summary>
public sealed class RegexTokenizer : ITokenizer
{
    private static readonly Regex DefaultPattern = new(@"\w+", RegexOptions.Compiled);
    private readonly Regex _pattern;
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _reverseVocab = new();
    private readonly bool _lowercase;
    private int _nextId;

    /// <summary>Unknown token string.</summary>
    public const string UnkToken = "[UNK]";

    /// <summary>Current vocabulary size.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Creates a new regex tokenizer.
    /// </summary>
    /// <param name="pattern">Regex pattern for token extraction. Defaults to <c>\w+</c>.</param>
    /// <param name="lowercase">Whether to lowercase all tokens before lookup.</param>
    public RegexTokenizer(string? pattern = null, bool lowercase = false)
    {
        _pattern = pattern is not null ? new Regex(pattern, RegexOptions.Compiled) : DefaultPattern;
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
            foreach (Match match in _pattern.Matches(text))
            {
                var token = match.Value;
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
        var matches = _pattern.Matches(text);
        var ids = new int[matches.Count];
        var mask = new int[matches.Count];
        var alignment = new (int, int)[matches.Count];

        int unkId = _vocab[UnkToken];
        for (int i = 0; i < matches.Count; i++)
        {
            var token = matches[i].Value;
            ids[i] = _vocab.TryGetValue(token, out int id) ? id : unkId;
            mask[i] = 1;
            alignment[i] = (i, i);
        }

        return new TokenizerResult(ids, mask, alignment);
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
