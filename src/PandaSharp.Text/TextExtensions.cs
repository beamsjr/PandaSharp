using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Text.Analytics;
using PandaSharp.Text.Tokenizers;

namespace PandaSharp.Text;

/// <summary>
/// Extension methods for integrating text analysis into PandaSharp DataFrames and columns.
/// </summary>
public static class TextExtensions
{
    /// <summary>
    /// Get a <see cref="TextColumnAccessor"/> for the named string column.
    /// Provides text analysis methods like token count, word frequency, etc.
    /// </summary>
    /// <param name="df">Source DataFrame.</param>
    /// <param name="columnName">Name of the string column.</param>
    /// <returns>Text column accessor wrapping the specified column.</returns>
    public static TextColumnAccessor GetTextColumn(this DataFrame df, string columnName)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(columnName);
        var col = df[columnName];
        if (col is not StringColumn strCol)
            throw new InvalidOperationException($"Column '{columnName}' is not a StringColumn (type: {col.DataType.Name}).");
        return new TextColumnAccessor(strCol);
    }

    /// <summary>
    /// Tokenize a string column using the provided tokenizer and add token ID columns.
    /// </summary>
    /// <param name="column">Source string column.</param>
    /// <param name="tokenizer">Tokenizer to use.</param>
    /// <returns>Array of <see cref="TokenizerResult"/> for each row.</returns>
    public static TokenizerResult[] Tokenize(this StringColumn column, ITokenizer tokenizer)
    {
        ArgumentNullException.ThrowIfNull(column);
        ArgumentNullException.ThrowIfNull(tokenizer);
        int len = column.Length;
        var results = new TokenizerResult[len];

        for (int i = 0; i < len; i++)
        {
            results[i] = tokenizer.Encode(column[i] ?? string.Empty);
        }

        return results;
    }
}
