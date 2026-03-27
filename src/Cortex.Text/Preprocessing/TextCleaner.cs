using System.Text.RegularExpressions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.Text.Preprocessing;

/// <summary>
/// Configurable text cleaning pipeline step. Implements <see cref="ITransformer"/>.
/// Supports lowercase, HTML stripping, URL/email/number removal, and whitespace normalization.
/// All regex patterns are pre-compiled for performance.
/// </summary>
public sealed class TextCleaner : ITransformer
{
    private static readonly Regex HtmlTagsRegex = new(@"<[^>]+>", RegexOptions.Compiled);
    private static readonly Regex UrlRegex = new(@"https?://\S+|www\.\S+", RegexOptions.Compiled);
    private static readonly Regex EmailRegex = new(@"\S+@\S+\.\S+", RegexOptions.Compiled);
    private static readonly Regex NumberRegex = new(@"\b\d+\b", RegexOptions.Compiled);
    private static readonly Regex WhitespaceRegex = new(@"\s+", RegexOptions.Compiled);

    private string _inputColumn = "text";

    /// <summary>Whether to convert text to lowercase.</summary>
    public bool Lowercase { get; set; } = true;

    /// <summary>Whether to strip HTML tags.</summary>
    public bool StripHtml { get; set; } = true;

    /// <summary>Whether to remove URLs.</summary>
    public bool RemoveUrls { get; set; } = true;

    /// <summary>Whether to remove email addresses.</summary>
    public bool RemoveEmails { get; set; } = true;

    /// <summary>Whether to remove standalone numbers.</summary>
    public bool RemoveNumbers { get; set; } = false;

    /// <summary>Whether to normalize whitespace (collapse multiple spaces to one).</summary>
    public bool NormalizeWhitespace { get; set; } = true;

    /// <inheritdoc />
    public string Name => "TextCleaner";

    /// <summary>
    /// Creates a new text cleaner with default settings (lowercase, strip HTML, remove URLs/emails, normalize whitespace).
    /// </summary>
    public TextCleaner() { }

    /// <summary>
    /// Set the input column name to process.
    /// </summary>
    /// <param name="column">Column name containing text.</param>
    /// <returns>This instance for chaining.</returns>
    public TextCleaner WithColumn(string column)
    {
        ArgumentNullException.ThrowIfNull(column);
        _inputColumn = column;
        return this;
    }

    /// <summary>
    /// Clean a single text string according to configured flags.
    /// </summary>
    /// <param name="text">Input text.</param>
    /// <returns>Cleaned text.</returns>
    public string Clean(string text)
    {
        ArgumentNullException.ThrowIfNull(text);
        if (StripHtml) text = HtmlTagsRegex.Replace(text, " ");
        if (RemoveUrls) text = UrlRegex.Replace(text, " ");
        if (RemoveEmails) text = EmailRegex.Replace(text, " ");
        if (RemoveNumbers) text = NumberRegex.Replace(text, " ");
        if (Lowercase) text = text.ToLowerInvariant();
        if (NormalizeWhitespace) text = WhitespaceRegex.Replace(text, " ").Trim();
        return text;
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
            result[i] = val is null ? null : Clean(val);
        }

        var newCol = new StringColumn(_inputColumn, result);
        return df.ReplaceColumn(_inputColumn, newCol);
    }
}
