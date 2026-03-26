using PandaSharp.Column;

namespace PandaSharp.Text.Preprocessing;

/// <summary>
/// Porter stemmer implementation for English text.
/// Implements Porter's 5-step algorithm for suffix stripping.
/// </summary>
public sealed class Stemmer
{
    /// <summary>
    /// Stem a single word using the Porter stemming algorithm.
    /// </summary>
    /// <param name="word">Input word.</param>
    /// <returns>Stemmed form of the word.</returns>
    public string Stem(string word)
    {
        ArgumentNullException.ThrowIfNull(word);
        if (word.Length <= 2) return word;

        var w = word.ToLowerInvariant();

        w = Step1a(w);
        w = Step1b(w);
        w = Step1c(w);
        w = Step2(w);
        w = Step3(w);
        w = Step4(w);
        w = Step5a(w);
        w = Step5b(w);

        return w;
    }

    /// <summary>
    /// Stem all values in a <see cref="StringColumn"/>.
    /// </summary>
    /// <param name="col">Input string column.</param>
    /// <returns>New column with stemmed values.</returns>
    public StringColumn StemColumn(StringColumn col)
    {
        int len = col.Length;
        var result = new string?[len];
        for (int i = 0; i < len; i++)
        {
            var val = col[i];
            if (val is null) { result[i] = null; continue; }
            var words = val.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            for (int j = 0; j < words.Length; j++)
                words[j] = Stem(words[j]);
            result[i] = string.Join(' ', words);
        }
        return new StringColumn(col.Name, result);
    }

    // -- Measure: count VC sequences in the stem --

    private static int Measure(ReadOnlySpan<char> stem)
    {
        int n = 0;
        int i = 0;
        // Skip leading consonants
        while (i < stem.Length && !IsVowel(stem, i)) i++;
        while (i < stem.Length)
        {
            // Skip vowels
            while (i < stem.Length && IsVowel(stem, i)) i++;
            if (i >= stem.Length) break;
            n++;
            // Skip consonants
            while (i < stem.Length && !IsVowel(stem, i)) i++;
        }
        return n;
    }

    private static bool IsVowel(ReadOnlySpan<char> w, int i)
    {
        return w[i] switch
        {
            'a' or 'e' or 'i' or 'o' or 'u' => true,
            'y' => i > 0 && !IsVowel(w, i - 1),
            _ => false
        };
    }

    private static bool ContainsVowel(ReadOnlySpan<char> stem)
    {
        for (int i = 0; i < stem.Length; i++)
            if (IsVowel(stem, i)) return true;
        return false;
    }

    private static bool EndsWithDouble(ReadOnlySpan<char> w)
    {
        if (w.Length < 2) return false;
        return w[^1] == w[^2] && !IsVowel(w, w.Length - 1);
    }

    private static bool CvcPattern(ReadOnlySpan<char> w)
    {
        if (w.Length < 3) return false;
        int last = w.Length - 1;
        if (IsVowel(w, last) || !IsVowel(w, last - 1) || IsVowel(w, last - 2))
            return false;
        return w[last] is not ('w' or 'x' or 'y');
    }

    private static string ReplaceSuffix(string w, string suffix, string replacement)
    {
        return w[..^suffix.Length] + replacement;
    }

    // -- Step 1a: plurals --

    private static string Step1a(string w)
    {
        if (w.EndsWith("sses", StringComparison.Ordinal)) return ReplaceSuffix(w, "sses", "ss");
        if (w.EndsWith("ies", StringComparison.Ordinal)) return ReplaceSuffix(w, "ies", "i");
        if (w.EndsWith("ss", StringComparison.Ordinal)) return w;
        if (w.EndsWith('s')) return w[..^1];
        return w;
    }

    // -- Step 1b: -ed, -ing --

    private static string Step1b(string w)
    {
        if (w.EndsWith("eed", StringComparison.Ordinal))
        {
            var stem = w.AsSpan(0, w.Length - 3);
            if (Measure(stem) > 0) return ReplaceSuffix(w, "eed", "ee");
            return w;
        }

        bool modified = false;
        if (w.EndsWith("ed", StringComparison.Ordinal))
        {
            var stem = w.AsSpan(0, w.Length - 2);
            if (ContainsVowel(stem)) { w = w[..^2]; modified = true; }
        }
        else if (w.EndsWith("ing", StringComparison.Ordinal))
        {
            var stem = w.AsSpan(0, w.Length - 3);
            if (ContainsVowel(stem)) { w = w[..^3]; modified = true; }
        }

        if (modified)
        {
            if (w.EndsWith("at", StringComparison.Ordinal) ||
                w.EndsWith("bl", StringComparison.Ordinal) ||
                w.EndsWith("iz", StringComparison.Ordinal))
                return w + "e";

            if (EndsWithDouble(w.AsSpan()) && w[^1] is not ('l' or 's' or 'z'))
                return w[..^1];

            if (Measure(w.AsSpan()) == 1 && CvcPattern(w.AsSpan()))
                return w + "e";
        }

        return w;
    }

    // -- Step 1c: y → i --

    private static string Step1c(string w)
    {
        if (w.EndsWith('y') && ContainsVowel(w.AsSpan(0, w.Length - 1)))
            return w[..^1] + "i";
        return w;
    }

    // -- Step 2: double-suffix removal --

    private static readonly (string Suffix, string Replacement)[] Step2Rules =
    [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"), ("anci", "ance"),
        ("izer", "ize"), ("abli", "able"), ("alli", "al"), ("entli", "ent"),
        ("eli", "e"), ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"), ("fulness", "ful"),
        ("ousness", "ous"), ("aliti", "al"), ("iviti", "ive"), ("biliti", "ble"),
        ("logi", "log")
    ];

    private static string Step2(string w)
    {
        foreach (var (suffix, replacement) in Step2Rules)
        {
            if (w.EndsWith(suffix, StringComparison.Ordinal))
            {
                var stem = w.AsSpan(0, w.Length - suffix.Length);
                if (Measure(stem) > 0) return ReplaceSuffix(w, suffix, replacement);
                return w;
            }
        }
        return w;
    }

    // -- Step 3 --

    private static readonly (string Suffix, string Replacement)[] Step3Rules =
    [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""), ("ness", "")
    ];

    private static string Step3(string w)
    {
        foreach (var (suffix, replacement) in Step3Rules)
        {
            if (w.EndsWith(suffix, StringComparison.Ordinal))
            {
                var stem = w.AsSpan(0, w.Length - suffix.Length);
                if (Measure(stem) > 0) return ReplaceSuffix(w, suffix, replacement);
                return w;
            }
        }
        return w;
    }

    // -- Step 4 --

    private static readonly string[] Step4Suffixes =
    [
        "al", "ance", "ence", "er", "ic", "able", "ible", "ant",
        "ement", "ment", "ent", "ion", "ou", "ism", "ate", "iti",
        "ous", "ive", "ize"
    ];

    private static string Step4(string w)
    {
        foreach (var suffix in Step4Suffixes)
        {
            if (w.EndsWith(suffix, StringComparison.Ordinal))
            {
                var stem = w.AsSpan(0, w.Length - suffix.Length);
                if (suffix == "ion" && stem.Length > 0 && (stem[^1] == 's' || stem[^1] == 't'))
                {
                    if (Measure(stem) > 1) return w[..^suffix.Length];
                }
                else if (Measure(stem) > 1)
                {
                    return w[..^suffix.Length];
                }
                return w;
            }
        }
        return w;
    }

    // -- Step 5a --

    private static string Step5a(string w)
    {
        if (w.EndsWith('e'))
        {
            var stem = w.AsSpan(0, w.Length - 1);
            if (Measure(stem) > 1) return w[..^1];
            if (Measure(stem) == 1 && !CvcPattern(stem)) return w[..^1];
        }
        return w;
    }

    // -- Step 5b --

    private static string Step5b(string w)
    {
        if (Measure(w.AsSpan()) > 1 && EndsWithDouble(w.AsSpan()) && w[^1] == 'l')
            return w[..^1];
        return w;
    }
}
