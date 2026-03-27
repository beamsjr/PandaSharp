using System.Text.RegularExpressions;

namespace Cortex.Text.Preprocessing;

/// <summary>
/// Rule-based sentence splitter. Splits text on sentence-ending punctuation
/// (.!?) followed by whitespace and an uppercase letter, while handling
/// common abbreviations.
/// </summary>
public sealed class SentenceSplitter
{
    private static readonly Regex SentenceBoundary = new(
        @"(?<=[.!?])\s+(?=[A-Z])",
        RegexOptions.Compiled);

    private static readonly HashSet<string> Abbreviations = new(StringComparer.OrdinalIgnoreCase)
    {
        "Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "Sr.", "Jr.", "St.",
        "Gen.", "Gov.", "Sgt.", "Cpl.", "Pvt.", "Capt.", "Lt.", "Col.",
        "Rev.", "Hon.", "Pres.",
        "Inc.", "Corp.", "Ltd.", "Co.", "Bros.",
        "vs.", "etc.", "approx.", "dept.", "est.",
        "Jan.", "Feb.", "Mar.", "Apr.", "Jun.", "Jul.", "Aug.",
        "Sep.", "Sept.", "Oct.", "Nov.", "Dec.",
        "Ave.", "Blvd.", "Rd.", "Ct.",
        "Fig.", "Vol.", "No.", "Eq.",
        "i.e.", "e.g.", "cf.", "al.",
    };

    /// <summary>
    /// Split text into individual sentences using rule-based heuristics.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Array of sentence strings.</returns>
    public string[] Split(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (string.IsNullOrWhiteSpace(text))
            return [];

        // First pass: protect abbreviations by replacing their periods with placeholders
        var protected_ = text;
        var abbrevPositions = new List<int>();

        // Find abbreviation positions and protect them
        foreach (var abbr in Abbreviations)
        {
            int idx = 0;
            while ((idx = protected_.IndexOf(abbr, idx, StringComparison.OrdinalIgnoreCase)) >= 0)
            {
                // Mark the period position in the abbreviation
                int periodPos = idx + abbr.Length - 1;
                abbrevPositions.Add(periodPos);
                idx += abbr.Length;
            }
        }

        // Sort and deduplicate
        abbrevPositions.Sort();

        // Replace abbreviation periods with a placeholder
        const char placeholder = '\x01';
        var chars = protected_.ToCharArray();
        foreach (var pos in abbrevPositions)
        {
            if (pos < chars.Length && chars[pos] == '.')
                chars[pos] = placeholder;
        }

        var processed = new string(chars);

        // Split on sentence boundaries
        var rawSentences = SentenceBoundary.Split(processed);

        // Restore placeholders and trim
        var result = new List<string>(rawSentences.Length);
        for (int i = 0; i < rawSentences.Length; i++)
        {
            var sentence = rawSentences[i].Replace(placeholder, '.').Trim();
            if (sentence.Length > 0)
                result.Add(sentence);
        }

        return result.ToArray();
    }
}
