namespace Cortex.Text.Preprocessing;

/// <summary>
/// Dictionary-based English lemmatizer.
/// Combines a built-in dictionary of irregular forms with rule-based suffix stripping
/// for regular verb, noun, and adjective/adverb forms.
/// </summary>
public sealed class Lemmatizer
{
    private readonly Dictionary<string, string> _irregulars;

    /// <summary>
    /// Creates a new lemmatizer with built-in English irregular forms.
    /// </summary>
    public Lemmatizer()
    {
        _irregulars = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            // be
            ["am"] = "be", ["is"] = "be", ["are"] = "be",
            ["was"] = "be", ["were"] = "be", ["been"] = "be", ["being"] = "be",

            // have
            ["has"] = "have", ["had"] = "have", ["having"] = "have",

            // do
            ["does"] = "do", ["did"] = "do", ["done"] = "do", ["doing"] = "do",

            // go
            ["went"] = "go", ["gone"] = "go", ["goes"] = "go", ["going"] = "go",

            // common irregular verbs
            ["ran"] = "run", ["running"] = "run",
            ["ate"] = "eat", ["eaten"] = "eat", ["eating"] = "eat",
            ["took"] = "take", ["taken"] = "take", ["taking"] = "take",
            ["saw"] = "see", ["seen"] = "see", ["seeing"] = "see",
            ["gave"] = "give", ["given"] = "give", ["giving"] = "give",
            ["came"] = "come", ["coming"] = "come",
            ["made"] = "make", ["making"] = "make",
            ["said"] = "say", ["saying"] = "say",
            ["told"] = "tell", ["telling"] = "tell",
            ["thought"] = "think", ["thinking"] = "think",
            ["knew"] = "know", ["known"] = "know", ["knowing"] = "know",
            ["got"] = "get", ["gotten"] = "get", ["getting"] = "get",
            ["found"] = "find", ["finding"] = "find",
            ["left"] = "leave", ["leaving"] = "leave",
            ["felt"] = "feel", ["feeling"] = "feel",
            ["kept"] = "keep", ["keeping"] = "keep",
            ["began"] = "begin", ["begun"] = "begin", ["beginning"] = "begin",
            ["wrote"] = "write", ["written"] = "write", ["writing"] = "write",
            ["brought"] = "bring", ["bringing"] = "bring",
            ["stood"] = "stand", ["standing"] = "stand",
            ["sat"] = "sit", ["sitting"] = "sit",
            ["broke"] = "break", ["broken"] = "break", ["breaking"] = "break",
            ["spoke"] = "speak", ["spoken"] = "speak", ["speaking"] = "speak",
            ["chose"] = "choose", ["chosen"] = "choose", ["choosing"] = "choose",
            ["fell"] = "fall", ["fallen"] = "fall", ["falling"] = "fall",
            ["held"] = "hold", ["holding"] = "hold",
            ["caught"] = "catch", ["catching"] = "catch",
            ["built"] = "build", ["building"] = "build",
            ["sent"] = "send", ["sending"] = "send",
            ["spent"] = "spend", ["spending"] = "spend",
            ["lost"] = "lose", ["losing"] = "lose",
            ["met"] = "meet", ["meeting"] = "meet",
            ["led"] = "lead", ["leading"] = "lead",
            ["read"] = "read",
            ["put"] = "put", ["putting"] = "put",
            ["cut"] = "cut", ["cutting"] = "cut",
            ["set"] = "set", ["setting"] = "set",
            ["let"] = "let", ["letting"] = "let",
            ["hit"] = "hit", ["hitting"] = "hit",
            ["shut"] = "shut", ["shutting"] = "shut",
            ["hurt"] = "hurt", ["hurting"] = "hurt",

            // irregular nouns
            ["men"] = "man", ["women"] = "woman", ["children"] = "child",
            ["mice"] = "mouse", ["teeth"] = "tooth", ["feet"] = "foot",
            ["geese"] = "goose", ["oxen"] = "ox", ["people"] = "person",
            ["lives"] = "life", ["wives"] = "wife", ["knives"] = "knife",
            ["leaves"] = "leaf", ["halves"] = "half", ["selves"] = "self",
            ["shelves"] = "shelf", ["wolves"] = "wolf", ["loaves"] = "loaf",
            ["thieves"] = "thief",

            // adjectives/adverbs
            ["better"] = "good", ["best"] = "good",
            ["worse"] = "bad", ["worst"] = "bad",
            ["more"] = "much", ["most"] = "much",
            ["less"] = "little", ["least"] = "little",
            ["further"] = "far", ["furthest"] = "far",
            ["farther"] = "far", ["farthest"] = "far",
        };
    }

    /// <summary>
    /// Lemmatize a single word.
    /// Checks the irregular forms dictionary first, then applies rule-based suffix stripping.
    /// </summary>
    /// <param name="word">Input word.</param>
    /// <returns>Lemmatized form.</returns>
    public string Lemmatize(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        if (string.IsNullOrWhiteSpace(word)) return word;

        var lower = word.ToLowerInvariant();

        // Check irregular forms first
        if (_irregulars.TryGetValue(lower, out var irregular))
            return irregular;

        // Rule-based suffix stripping

        // -ly → remove (e.g., quickly → quick)
        if (lower.EndsWith("ly", StringComparison.Ordinal) && lower.Length > 4)
        {
            var stem = lower[..^2];
            if (stem.EndsWith("ful", StringComparison.Ordinal))
                return stem[..^3]; // beautifully → beauti → beautiful? no, just return stem
            if (stem.EndsWith("i", StringComparison.Ordinal) && stem.Length > 1)
                return stem[..^1] + "y"; // happily → happi → happy
            return stem;
        }

        // -ing → various rules
        if (lower.EndsWith("ing", StringComparison.Ordinal) && lower.Length > 5)
        {
            var stem = lower[..^3];
            // Double consonant: running → run
            if (stem.Length >= 2 && stem[^1] == stem[^2] && stem[^1] is not ('l' or 's'))
                return stem[..^1];
            // Consonant + e removed: making → mak → make
            if (stem.Length >= 2 && !IsVowel(stem[^1]))
                return stem + "e";
            return stem;
        }

        // -ed → various rules
        if (lower.EndsWith("ed", StringComparison.Ordinal) && lower.Length > 4)
        {
            if (lower.EndsWith("ied", StringComparison.Ordinal))
                return lower[..^3] + "y"; // carried → carry
            var stem = lower[..^2];
            // Double consonant: stopped → stop
            if (stem.Length >= 2 && stem[^1] == stem[^2])
                return stem[..^1];
            // Consonant + e: baked → bake
            if (stem.Length >= 2 && lower[^3] != 'e' && !IsVowel(stem[^1]))
                return stem + "e";
            return stem;
        }

        // -s plurals
        if (lower.EndsWith('s') && !lower.EndsWith("ss", StringComparison.Ordinal) && lower.Length > 3)
        {
            if (lower.EndsWith("ies", StringComparison.Ordinal))
                return lower[..^3] + "y"; // berries → berry
            if (lower.EndsWith("ves", StringComparison.Ordinal))
                return lower[..^3] + "f"; // wolves → wolf
            if (lower.EndsWith("ses", StringComparison.Ordinal) || lower.EndsWith("xes", StringComparison.Ordinal) ||
                lower.EndsWith("zes", StringComparison.Ordinal) || lower.EndsWith("ches", StringComparison.Ordinal) ||
                lower.EndsWith("shes", StringComparison.Ordinal))
                return lower[..^2]; // boxes → box, watches → watch
            return lower[..^1]; // cats → cat
        }

        return lower;
    }

    /// <summary>
    /// Add a custom irregular form mapping.
    /// </summary>
    /// <param name="form">Inflected form.</param>
    /// <param name="lemma">Base (lemma) form.</param>
    public void AddIrregular(string form, string lemma)
    {
        ArgumentNullException.ThrowIfNull(form);
        ArgumentNullException.ThrowIfNull(lemma);
        _irregulars[form] = lemma;
    }

    private static bool IsVowel(char c) => c is 'a' or 'e' or 'i' or 'o' or 'u';
}
