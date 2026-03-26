namespace PandaSharp.Text.Tokenizers;

/// <summary>
/// Byte-Pair Encoding (BPE) tokenizer.
/// Iteratively merges the most frequent adjacent token pairs to build a subword vocabulary.
/// </summary>
public sealed class BPETokenizer : ITokenizer
{
    private readonly Dictionary<string, int> _vocab = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _reverseVocab = new();
    private readonly List<(string A, string B)> _merges = new();
    private int _nextId;

    /// <summary>Unknown token string.</summary>
    public const string UnkToken = "[UNK]";

    /// <summary>End-of-word marker.</summary>
    public const string EndOfWord = "</w>";

    /// <summary>Current vocabulary size.</summary>
    public int VocabSize => _vocab.Count;

    /// <summary>
    /// Creates a new BPE tokenizer with default special tokens.
    /// </summary>
    public BPETokenizer()
    {
        AddToken(UnkToken);
    }

    /// <summary>
    /// Train the BPE vocabulary from a corpus.
    /// </summary>
    /// <param name="corpus">Array of text documents.</param>
    /// <param name="vocabSize">Target vocabulary size.</param>
    public void Train(string[] corpus, int vocabSize)
    {
        ArgumentNullException.ThrowIfNull(corpus);
        if (vocabSize < 1) throw new ArgumentOutOfRangeException(nameof(vocabSize), "vocabSize must be >= 1.");
        // Build initial character-level vocabulary and word frequencies
        var wordFreqs = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int d = 0; d < corpus.Length; d++)
        {
            var words = corpus[d].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            foreach (var word in words)
            {
                var key = string.Join(' ', word.Select(c => c.ToString())) + " " + EndOfWord;
                wordFreqs.TryGetValue(key, out int count);
                wordFreqs[key] = count + 1;
            }
        }

        // Seed vocab with all individual characters
        foreach (var entry in wordFreqs)
        {
            foreach (var token in entry.Key.Split(' '))
            {
                if (!_vocab.ContainsKey(token))
                    AddToken(token);
            }
        }

        // Iteratively merge most frequent pairs
        while (_vocab.Count < vocabSize)
        {
            var pairFreqs = new Dictionary<(string, string), int>();
            foreach (var (word, freq) in wordFreqs)
            {
                var symbols = word.Split(' ');
                for (int i = 0; i < symbols.Length - 1; i++)
                {
                    var pair = (symbols[i], symbols[i + 1]);
                    pairFreqs.TryGetValue(pair, out int pf);
                    pairFreqs[pair] = pf + freq;
                }
            }

            if (pairFreqs.Count == 0) break;

            var bestPair = pairFreqs.MaxBy(kv => kv.Value).Key;
            _merges.Add(bestPair);

            var merged = bestPair.Item1 + bestPair.Item2;
            if (!_vocab.ContainsKey(merged))
                AddToken(merged);

            // Apply merge to all words
            var newWordFreqs = new Dictionary<string, int>(StringComparer.Ordinal);
            foreach (var (word, freq) in wordFreqs)
            {
                var newWord = ApplyMerge(word, bestPair.Item1, bestPair.Item2);
                newWordFreqs[newWord] = freq;
            }
            wordFreqs = newWordFreqs;
        }
    }

    /// <inheritdoc />
    public TokenizerResult Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (_merges.Count == 0 && _vocab.Count <= 1)
            throw new InvalidOperationException("Tokenizer has not been trained. Call Train() or LoadVocab() first.");
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var allIds = new List<int>();
        var alignment = new List<(int, int)>();
        int tokenIdx = 0;

        int unkId = _vocab[UnkToken];
        for (int w = 0; w < words.Length; w++)
        {
            var symbols = words[w].Select(c => c.ToString()).ToList();
            symbols.Add(EndOfWord);

            // Apply merges in order
            foreach (var (a, b) in _merges)
            {
                int i = 0;
                while (i < symbols.Count - 1)
                {
                    if (symbols[i] == a && symbols[i + 1] == b)
                    {
                        symbols[i] = a + b;
                        symbols.RemoveAt(i + 1);
                    }
                    else
                    {
                        i++;
                    }
                }
            }

            foreach (var sym in symbols)
            {
                allIds.Add(_vocab.TryGetValue(sym, out int id) ? id : unkId);
                alignment.Add((tokenIdx, w));
                tokenIdx++;
            }
        }

        var ids = allIds.ToArray();
        var mask = new int[ids.Length];
        Array.Fill(mask, 1);

        return new TokenizerResult(ids, mask, alignment.ToArray());
    }

    /// <inheritdoc />
    public string Decode(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < ids.Length; i++)
        {
            var token = _reverseVocab.TryGetValue(ids[i], out var t) ? t : UnkToken;
            if (token == EndOfWord)
            {
                sb.Append(' ');
            }
            else
            {
                sb.Append(token);
            }
        }
        return sb.ToString().Trim();
    }

    /// <summary>
    /// Save the vocabulary and merge rules to a file.
    /// </summary>
    /// <param name="path">File path to write.</param>
    public void SaveVocab(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var writer = new StreamWriter(path);
        writer.WriteLine($"#version: pandasharp-bpe-v1");

        // Write vocab
        writer.WriteLine($"#vocab {_vocab.Count}");
        foreach (var (token, id) in _vocab.OrderBy(kv => kv.Value))
        {
            writer.WriteLine($"{id}\t{token}");
        }

        // Write merges
        writer.WriteLine($"#merges {_merges.Count}");
        foreach (var (a, b) in _merges)
        {
            writer.WriteLine($"{a} {b}");
        }
    }

    /// <summary>
    /// Load vocabulary and merge rules from a file.
    /// </summary>
    /// <param name="path">File path to read.</param>
    public void LoadVocab(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        _vocab.Clear();
        _reverseVocab.Clear();
        _merges.Clear();
        _nextId = 0;

        using var reader = new StreamReader(path);
        string? line;
        var section = "";

        int vocabCount = 0;
        int mergesCount = 0;

        while ((line = reader.ReadLine()) is not null)
        {
            if (line.StartsWith("#version:")) continue;

            if (line.StartsWith("#vocab "))
            {
                section = "vocab";
                vocabCount = int.Parse(line.AsSpan(7));
                continue;
            }

            if (line.StartsWith("#merges "))
            {
                section = "merges";
                mergesCount = int.Parse(line.AsSpan(8));
                continue;
            }

            if (section == "vocab")
            {
                var tabIdx = line.IndexOf('\t');
                if (tabIdx < 0) continue;
                var id = int.Parse(line.AsSpan(0, tabIdx));
                var token = line[(tabIdx + 1)..];
                _vocab[token] = id;
                _reverseVocab[id] = token;
                if (id >= _nextId) _nextId = id + 1;
            }
            else if (section == "merges")
            {
                var spaceIdx = line.IndexOf(' ');
                if (spaceIdx < 0) continue;
                _merges.Add((line[..spaceIdx], line[(spaceIdx + 1)..]));
            }
        }
    }

    private static string ApplyMerge(string word, string a, string b)
    {
        var symbols = word.Split(' ');
        var result = new List<string>(symbols.Length);
        int i = 0;
        while (i < symbols.Length)
        {
            if (i < symbols.Length - 1 && symbols[i] == a && symbols[i + 1] == b)
            {
                result.Add(a + b);
                i += 2;
            }
            else
            {
                result.Add(symbols[i]);
                i++;
            }
        }
        return string.Join(' ', result);
    }

    private void AddToken(string token)
    {
        int id = _nextId++;
        _vocab[token] = id;
        _reverseVocab[id] = token;
    }
}
