namespace PandaSharp.Text.Tokenizers;

/// <summary>
/// Stub SentencePiece tokenizer for loading pre-trained <c>.model</c> files.
/// Implements basic unigram model scoring. Full native binding support is planned
/// for a future release via SentencePiece C++ library interop.
/// </summary>
public sealed class SentencePieceTokenizer : ITokenizer
{
    private readonly Dictionary<string, (int Id, float Score)> _pieces = new(StringComparer.Ordinal);
    private readonly Dictionary<int, string> _reversePieces = new();
    private bool _loaded;

    /// <summary>Sentence boundary marker used by SentencePiece.</summary>
    public const string SentenceBoundary = "\u2581"; // ▁

    /// <summary>Unknown token string.</summary>
    public const string UnkToken = "<unk>";

    /// <summary>Current vocabulary size.</summary>
    public int VocabSize => _pieces.Count;

    /// <summary>
    /// Creates a new SentencePiece tokenizer.
    /// Call <see cref="LoadModel"/> to load a pre-trained model file.
    /// </summary>
    public SentencePieceTokenizer() { }

    /// <summary>
    /// Load a pre-trained SentencePiece vocabulary from a text-format vocab file.
    /// Each line: <c>piece\tscore</c>.
    /// </summary>
    /// <remarks>
    /// Loading binary <c>.model</c> protobuf files requires native SentencePiece bindings,
    /// which are not yet available. Use <c>spm_export_vocab</c> to export a text vocabulary.
    /// </remarks>
    /// <param name="path">Path to the vocabulary file (text format).</param>
    public void LoadModel(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        _pieces.Clear();
        _reversePieces.Clear();

        int id = 0;
        foreach (var line in File.ReadLines(path))
        {
            var tabIdx = line.IndexOf('\t');
            if (tabIdx < 0) continue;

            var piece = line[..tabIdx];
            var score = float.Parse(line.AsSpan(tabIdx + 1),
                System.Globalization.CultureInfo.InvariantCulture);

            _pieces[piece] = (id, score);
            _reversePieces[id] = piece;
            id++;
        }

        _loaded = true;
    }

    /// <inheritdoc />
    public TokenizerResult Encode(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (!_loaded)
            throw new InvalidOperationException(
                "No model loaded. Call LoadModel() with a text-format vocabulary file first. " +
                "For binary .model files, export with spm_export_vocab first.");

        // Basic greedy left-to-right longest-match encoding
        var normalized = SentenceBoundary + text.Replace(" ", SentenceBoundary);
        var ids = new List<int>();

        int unkId = _pieces.TryGetValue(UnkToken, out var unkEntry) ? unkEntry.Id : 0;
        int pos = 0;

        while (pos < normalized.Length)
        {
            int bestLen = 0;
            int bestId = unkId;

            // Try longest match first
            int maxLen = Math.Min(normalized.Length - pos, 64);
            for (int len = maxLen; len >= 1; len--)
            {
                var candidate = normalized.Substring(pos, len);
                if (_pieces.TryGetValue(candidate, out var entry))
                {
                    bestLen = len;
                    bestId = entry.Id;
                    break;
                }
            }

            if (bestLen == 0)
            {
                // Single character fallback
                ids.Add(unkId);
                pos++;
            }
            else
            {
                ids.Add(bestId);
                pos += bestLen;
            }
        }

        var idsArr = ids.ToArray();
        var mask = new int[idsArr.Length];
        Array.Fill(mask, 1);

        return new TokenizerResult(idsArr, mask);
    }

    /// <inheritdoc />
    public string Decode(int[] ids)
    {
        ArgumentNullException.ThrowIfNull(ids);
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < ids.Length; i++)
        {
            var piece = _reversePieces.TryGetValue(ids[i], out var p) ? p : UnkToken;
            sb.Append(piece);
        }
        return sb.ToString().Replace(SentenceBoundary, " ").Trim();
    }
}
