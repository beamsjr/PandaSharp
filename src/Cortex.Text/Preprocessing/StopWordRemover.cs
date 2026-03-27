using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.Text.Preprocessing;

/// <summary>
/// Removes stop words from text columns. Implements the <see cref="ITransformer"/> pattern.
/// Ships with built-in word lists for English, Spanish, French, and German.
/// </summary>
public sealed class StopWordRemover : ITransformer
{
    private readonly HashSet<string> _stopWords;
    private string _inputColumn = "text";

    /// <inheritdoc />
    public string Name => "StopWordRemover";

    /// <summary>
    /// Creates a stop word remover with a built-in language list and optional custom words.
    /// </summary>
    /// <param name="language">Language for built-in stop words: "english", "spanish", "french", "german".</param>
    /// <param name="custom">Additional custom stop words to include.</param>
    public StopWordRemover(string language = "english", string[]? custom = null)
    {
        _stopWords = new HashSet<string>(GetStopWords(language), StringComparer.OrdinalIgnoreCase);
        if (custom is not null)
        {
            foreach (var w in custom)
                _stopWords.Add(w);
        }
    }

    /// <summary>
    /// Set the input column name to process.
    /// </summary>
    /// <param name="column">Column name containing text.</param>
    /// <returns>This instance for chaining.</returns>
    public StopWordRemover WithColumn(string column)
    {
        ArgumentNullException.ThrowIfNull(column);
        _inputColumn = column;
        return this;
    }

    /// <inheritdoc />
    public ITransformer Fit(DataFrame df) => this; // No fitting needed

    /// <inheritdoc />
    public DataFrame Transform(DataFrame df)
    {
        var col = (StringColumn)df[_inputColumn];
        int len = col.Length;
        var result = new string?[len];

        for (int i = 0; i < len; i++)
        {
            var val = col[i];
            if (val is null) { result[i] = null; continue; }
            var words = val.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            var filtered = new List<string>(words.Length);
            foreach (var word in words)
            {
                if (!_stopWords.Contains(word))
                    filtered.Add(word);
            }
            result[i] = string.Join(' ', filtered);
        }

        var newCol = new StringColumn(_inputColumn, result);
        return df.ReplaceColumn(_inputColumn, newCol);
    }

    /// <summary>
    /// Remove stop words from a single text string.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Text with stop words removed.</returns>
    public string RemoveStopWords(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        var words = text.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
        var filtered = new List<string>(words.Length);
        foreach (var word in words)
        {
            if (!_stopWords.Contains(word))
                filtered.Add(word);
        }
        return string.Join(' ', filtered);
    }

    private static string[] GetStopWords(string language) => language.ToLowerInvariant() switch
    {
        "english" => EnglishStopWords,
        "spanish" => SpanishStopWords,
        "french" => FrenchStopWords,
        "german" => GermanStopWords,
        _ => throw new ArgumentException($"Unsupported language: {language}. Supported: english, spanish, french, german.")
    };

    private static readonly string[] EnglishStopWords =
    [
        "a", "about", "above", "after", "again", "against", "ain", "all", "am", "an",
        "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been",
        "before", "being", "below", "between", "both", "but", "by", "can", "couldn",
        "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't",
        "doing", "don", "don't", "down", "during", "each", "few", "for", "from",
        "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven",
        "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself",
        "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's",
        "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more",
        "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor",
        "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our",
        "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't",
        "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some",
        "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them",
        "themselves", "then", "there", "these", "they", "this", "those", "through",
        "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't",
        "we", "were", "weren", "weren't", "what", "when", "where", "which", "while",
        "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't",
        "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
        "yourselves"
    ];

    private static readonly string[] SpanishStopWords =
    [
        "a", "al", "algo", "algunas", "algunos", "ante", "antes", "como", "con",
        "contra", "cual", "cuando", "de", "del", "desde", "donde", "durante", "e",
        "el", "ella", "ellas", "ellos", "en", "entre", "era", "esa", "esas", "ese",
        "eso", "esos", "esta", "estaba", "estado", "estar", "estas", "este", "esto",
        "estos", "fue", "ha", "hasta", "hay", "la", "las", "le", "les", "lo", "los",
        "me", "mi", "muy", "nada", "ni", "no", "nos", "nosotros", "nuestro", "o",
        "otra", "otro", "para", "pero", "por", "que", "quien", "se", "ser", "si",
        "sin", "sino", "sobre", "somos", "son", "su", "sus", "te", "ti", "tiene",
        "todo", "tu", "tus", "u", "un", "una", "uno", "unos", "usted", "ustedes",
        "y", "ya", "yo"
    ];

    private static readonly string[] FrenchStopWords =
    [
        "a", "ai", "au", "aux", "avec", "c", "ce", "ces", "dans", "de", "des", "du",
        "elle", "en", "est", "et", "eu", "il", "je", "j", "la", "le", "les", "leur",
        "lui", "l", "m", "ma", "mais", "me", "mes", "mon", "n", "ne", "ni", "nos",
        "notre", "nous", "on", "ont", "ou", "par", "pas", "pour", "qu", "que", "qui",
        "s", "sa", "se", "ses", "si", "son", "sont", "sur", "t", "ta", "te", "tes",
        "ton", "tu", "un", "une", "vos", "votre", "vous", "y"
    ];

    private static readonly string[] GermanStopWords =
    [
        "aber", "alle", "allem", "allen", "aller", "als", "also", "am", "an", "ander",
        "andere", "anderem", "anderen", "anderer", "anderes", "auch", "auf", "aus",
        "bei", "bin", "bis", "bist", "da", "damit", "dann", "das", "dass", "dazu",
        "dein", "deine", "deinem", "deinen", "deiner", "dem", "den", "denn", "der",
        "des", "die", "dies", "diese", "dieselbe", "dieselben", "diesem", "diesen",
        "dieser", "dieses", "doch", "dort", "du", "durch", "ein", "eine", "einem",
        "einen", "einer", "er", "es", "etwas", "euch", "euer", "eure", "eurem",
        "euren", "eurer", "fur", "gegen", "hab", "habe", "haben", "hat", "hatte",
        "ich", "ihm", "ihn", "ihnen", "ihr", "ihre", "ihrem", "ihren", "ihrer", "im",
        "in", "indem", "ins", "ist", "jede", "jedem", "jeden", "jeder", "jedes",
        "jene", "jenem", "jenen", "jener", "jenes", "kann", "kein", "keine", "keinem",
        "keinen", "keiner", "man", "manche", "manchem", "manchen", "mancher", "manches",
        "mein", "meine", "meinem", "meinen", "meiner", "mit", "muss", "nach", "nicht",
        "nichts", "noch", "nun", "nur", "ob", "oder", "ohne", "sehr", "sein", "seine",
        "seinem", "seinen", "seiner", "sich", "sie", "sind", "so", "solche", "solchem",
        "solchen", "solcher", "soll", "sollte", "sondern", "sonst", "uber", "um", "und",
        "uns", "unser", "unsere", "unserem", "unseren", "unserer", "unter", "viel",
        "vom", "von", "vor", "wahrend", "war", "warum", "was", "weil", "welche",
        "welchem", "welchen", "welcher", "welches", "wenn", "wer", "werde", "wie",
        "wieder", "will", "wir", "wird", "wo", "wollen", "wurde", "zu", "zum", "zur",
        "zwar", "zwischen"
    ];
}
