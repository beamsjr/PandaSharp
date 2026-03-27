namespace Cortex.Text.Tokenizers;

/// <summary>
/// WordPiece tokenizer compatible with BERT-style models.
/// Supports HuggingFace vocab.txt format and <c>##</c> continuation tokens.
/// </summary>
public sealed class WordPieceTokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _reverseVocab = new();
    private int _maxTokenLength = 200;

    /// <summary>Special token: classification token prepended to input.</summary>
    public const string ClsToken = "[CLS]";
    /// <summary>Special token: separator between sentences.</summary>
    public const string SepToken = "[SEP]";
    /// <summary>Special token: padding.</summary>
    public const string PadToken = "[PAD]";
    /// <summary>Special token: unknown token.</summary>
    public const string UnkToken = "[UNK]";
    /// <summary>Special token: mask for MLM pre-training.</summary>
    public const string MaskToken = "[MASK]";
    /// <summary>Continuation prefix for subword tokens.</summary>
    public const string ContinuationPrefix = "##";

    /// <summary>Current vocabulary size.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Creates a new WordPiece tokenizer. Call <see cref="LoadVocab"/> to load a vocabulary.
    /// </summary>
    public WordPieceTokenizer() { }

    /// <summary>
    /// Load vocabulary from a HuggingFace-format vocab.txt file.
    /// Each line contains one token. Line number is the token ID.
    /// </summary>
    /// <param name="path">Path to vocab.txt file.</param>
    public void LoadVocab(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        _vocab.Clear();
        _reverseVocab.Clear();
        int id = 0;
        foreach (var line in File.ReadLines(path))
        {
            var token = line.TrimEnd();
            if (token.Length > 0)
            {
                _vocab[token] = id;
                _reverseVocab[id] = token;
                if (token.Length > _maxTokenLength)
                    _maxTokenLength = token.Length;
            }
            id++;
        }
    }

    /// <summary>
    /// Add a vocabulary entry manually (useful for building vocab programmatically).
    /// </summary>
    /// <param name="token">Token string.</param>
    /// <param name="id">Token ID.</param>
    public void AddToken(string token, int id)
    {
        _vocab[token] = id;
        _reverseVocab[id] = token;
    }

    /// <inheritdoc />
    public TokenizerResult Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_vocab.Count == 0)
            throw new InvalidOperationException("Vocabulary is empty. Call LoadVocab() or AddToken() first.");
        var tokenIds = new List<int>();
        var alignment = new List<(int, int)>();
        int tokenIdx = 0;

        int clsId = GetIdOrUnk(ClsToken);
        int sepId = GetIdOrUnk(SepToken);
        int unkId = GetIdOrUnk(UnkToken);

        // Prepend [CLS]
        tokenIds.Add(clsId);
        alignment.Add((tokenIdx++, -1));

        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        for (int w = 0; w < words.Length; w++)
        {
            var word = words[w].ToLowerInvariant();
            var subTokens = TokenizeWord(word, unkId);
            foreach (var (token, id) in subTokens)
            {
                tokenIds.Add(id);
                alignment.Add((tokenIdx++, w));
            }
        }

        // Append [SEP]
        tokenIds.Add(sepId);
        alignment.Add((tokenIdx, -1));

        var ids = tokenIds.ToArray();
        var mask = new int[ids.Length];
        Array.Fill(mask, 1);

        return new TokenizerResult(ids, mask, alignment.ToArray());
    }

    /// <inheritdoc />
    public string Decode(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var sb = new System.Text.StringBuilder();
        bool first = true;
        for (int i = 0; i < ids.Length; i++)
        {
            var token = _reverseVocab.TryGetValue(ids[i], out var t) ? t : UnkToken;

            // Skip special tokens
            if (token is ClsToken or SepToken or PadToken or MaskToken)
                continue;

            if (token.StartsWith(ContinuationPrefix, StringComparison.Ordinal))
            {
                sb.Append(token.AsSpan(ContinuationPrefix.Length));
            }
            else
            {
                if (!first) sb.Append(' ');
                sb.Append(token);
            }
            first = false;
        }
        return sb.ToString();
    }

    private List<(string Token, int Id)> TokenizeWord(string word, int unkId)
    {
        var result = new List<(string, int)>();
        int start = 0;

        while (start < word.Length)
        {
            int end = word.Length;
            string? bestToken = null;
            int bestId = unkId;
            bool found = false;

            while (end > start)
            {
                var substr = start == 0
                    ? word[start..end]
                    : ContinuationPrefix + word[start..end];

                if (_vocab.TryGetValue(substr, out int id))
                {
                    bestToken = substr;
                    bestId = id;
                    found = true;
                    break;
                }
                end--;
            }

            if (!found)
            {
                // Try single character before falling back to [UNK]
                var charStr = start == 0
                    ? word[start].ToString()
                    : ContinuationPrefix + word[start];
                if (_vocab.TryGetValue(charStr, out int charId))
                {
                    bestToken = charStr;
                    bestId = charId;
                    end = start + 1;
                }
                else
                {
                    // Character not in vocab — emit [UNK] for entire remaining word
                    result.Add((UnkToken, unkId));
                    return result;
                }
            }

            result.Add((bestToken!, bestId));
            start = end;
        }

        return result;
    }

    private int GetIdOrUnk(string token)
    {
        return _vocab.TryGetValue(token, out int id) ? id : 0;
    }
}
