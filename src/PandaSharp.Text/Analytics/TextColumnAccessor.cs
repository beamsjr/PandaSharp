using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Text.Preprocessing;

namespace PandaSharp.Text.Analytics;

/// <summary>
/// Provides text analysis extension methods for <see cref="StringColumn"/>.
/// Wraps a string column and exposes computed text metrics.
/// </summary>
public sealed class TextColumnAccessor
{
    private readonly StringColumn _column;

    /// <summary>
    /// Creates a text column accessor wrapping the given string column.
    /// </summary>
    /// <param name="column">String column to analyze.</param>
    public TextColumnAccessor(StringColumn column)
    {
        _column = column ?? throw new ArgumentNullException(nameof(column));
    }

    /// <summary>
    /// Count the number of whitespace-delimited tokens in each row.
    /// </summary>
    /// <returns>Integer column with token counts.</returns>
    public Column<int> TokenCount()
    {
        int len = _column.Length;
        var counts = new int[len];
        for (int i = 0; i < len; i++)
        {
            var val = _column[i];
            if (val is null) continue;
            counts[i] = CountTokens(val.AsSpan());
        }
        return new Column<int>($"{_column.Name}_token_count", counts);
    }

    /// <summary>
    /// Compute word frequency distribution across all rows.
    /// </summary>
    /// <returns>DataFrame with columns: word, count, sorted by count descending.</returns>
    public DataFrame WordFrequency()
    {
        var freq = new Dictionary<string, int>(StringComparer.Ordinal);
        int len = _column.Length;

        for (int i = 0; i < len; i++)
        {
            var val = _column[i];
            if (val is null) continue;
            var words = val.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            foreach (var word in words)
            {
                freq.TryGetValue(word, out int count);
                freq[word] = count + 1;
            }
        }

        var sorted = freq.OrderByDescending(kv => kv.Value).ToArray();
        var wordCol = new string?[sorted.Length];
        var countCol = new int[sorted.Length];
        for (int i = 0; i < sorted.Length; i++)
        {
            wordCol[i] = sorted[i].Key;
            countCol[i] = sorted[i].Value;
        }

        return new DataFrame(
            new StringColumn("word", wordCol),
            new Column<int>("count", countCol));
    }

    /// <summary>
    /// Count the number of sentences in each row using the <see cref="SentenceSplitter"/>.
    /// </summary>
    /// <returns>Integer column with sentence counts.</returns>
    public Column<int> SentenceCount()
    {
        var splitter = new SentenceSplitter();
        int len = _column.Length;
        var counts = new int[len];
        for (int i = 0; i < len; i++)
        {
            var val = _column[i];
            if (val is null) continue;
            counts[i] = splitter.Split(val).Length;
        }
        return new Column<int>($"{_column.Name}_sentence_count", counts);
    }

    /// <summary>
    /// Compute the average word length for each row.
    /// </summary>
    /// <returns>Double column with average word lengths.</returns>
    public Column<double> AverageWordLength()
    {
        int len = _column.Length;
        var avgs = new double[len];
        for (int i = 0; i < len; i++)
        {
            var val = _column[i];
            if (val is null) continue;
            var words = val.Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
            if (words.Length == 0) continue;
            long totalLen = 0;
            foreach (var w in words) totalLen += w.Length;
            avgs[i] = (double)totalLen / words.Length;
        }
        return new Column<double>($"{_column.Name}_avg_word_length", avgs);
    }

    /// <summary>
    /// Count the number of characters in each row (including whitespace).
    /// </summary>
    /// <returns>Integer column with character counts.</returns>
    public Column<int> CharCount()
    {
        int len = _column.Length;
        var counts = new int[len];
        for (int i = 0; i < len; i++)
        {
            counts[i] = _column[i]?.Length ?? 0;
        }
        return new Column<int>($"{_column.Name}_char_count", counts);
    }

    private static int CountTokens(ReadOnlySpan<char> text)
    {
        int count = 0;
        bool inWord = false;
        for (int i = 0; i < text.Length; i++)
        {
            if (char.IsWhiteSpace(text[i]))
            {
                inWord = false;
            }
            else if (!inWord)
            {
                inWord = true;
                count++;
            }
        }
        return count;
    }
}
