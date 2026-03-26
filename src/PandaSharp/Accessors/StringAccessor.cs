using System.Text;
using System.Text.RegularExpressions;
using PandaSharp.Column;

namespace PandaSharp.Accessors;

/// <summary>
/// Vectorized string operations accessible via StringColumn.Str property.
/// All operations are null-propagating: null input → null output.
/// </summary>
public class StringAccessor
{
    private readonly StringColumn _column;
    private DictEncoding? _dict;

    internal StringAccessor(StringColumn column, DictEncoding? cachedDict = null)
    {
        _column = column;
        _dict = cachedDict;
    }

    private string?[] Values => _column.GetValues();
    private int Length => _column.Length;
    private string Name => _column.Name;

    /// <summary>
    /// Get or build dictionary encoding. When K unique values &lt;&lt; N rows,
    /// operations are O(K) instead of O(N) — e.g., 6K uniques in 14.7M rows = 2000x faster.
    /// </summary>
    private DictEncoding GetDict()
    {
        if (_dict is null)
        {
            _dict = DictEncoding.Encode(_column);
            _column.CacheDictEncoding(_dict); // cache for reuse across Str accessors
        }
        return _dict;
    }

    /// <summary>Whether dictionary encoding is worthwhile (K &lt; N/10).</summary>
    private bool ShouldUseDict => Length > 10_000;

    // -- Matching --

    public Column<bool> Contains(string substring)
    {
        if (ShouldUseDict)
        {
            var mask = GetDict().Contains(substring);
            return new Column<bool>(Name, mask);
        }
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Contains(substring);
        return Column<bool>.FromNullable(Name, result);
    }

    public Column<bool> StartsWith(string prefix)
    {
        if (ShouldUseDict)
        {
            var mask = GetDict().StartsWith(prefix);
            return new Column<bool>(Name, mask);
        }
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.StartsWith(prefix);
        return Column<bool>.FromNullable(Name, result);
    }

    public Column<bool> EndsWith(string suffix)
    {
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.EndsWith(suffix);
        return Column<bool>.FromNullable(Name, result);
    }

    public Column<bool> Match(string pattern)
    {
        var regex = new Regex(pattern);
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i] is { } s ? regex.IsMatch(s) : null;
        return Column<bool>.FromNullable(Name, result);
    }

    /// <summary>
    /// Extract first capture group from regex pattern.
    /// </summary>
    public StringColumn Extract(string pattern, int group = 1)
    {
        var regex = new Regex(pattern);
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            if (Values[i] is { } s)
            {
                var match = regex.Match(s);
                result[i] = match.Success && match.Groups.Count > group ? match.Groups[group].Value : null;
            }
        }
        return new StringColumn(Name, result);
    }

    // -- Transformation --

    public StringColumn Replace(string old, string @new)
    {
        if (ShouldUseDict) return GetDict().Replace(Name, old, @new);
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Replace(old, @new);
        return new StringColumn(Name, result);
    }

    public StringColumn Upper()
    {
        if (ShouldUseDict) return GetDict().TransformUniques(Name, s => s.ToUpperInvariant());

        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.ToUpperInvariant();
        return StringColumn.CreateOwned(Name, result);
    }

    public StringColumn Lower()
    {
        if (ShouldUseDict) return GetDict().TransformUniques(Name, s => s.ToLowerInvariant());

        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.ToLowerInvariant();
        return StringColumn.CreateOwned(Name, result);
    }

    public StringColumn Title()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i] is { } s ? System.Globalization.CultureInfo.InvariantCulture.TextInfo.ToTitleCase(s.ToLowerInvariant()) : null;
        return new StringColumn(Name, result);
    }

    public StringColumn Capitalize()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            if (Values[i] is { Length: > 0 } s)
                result[i] = char.ToUpperInvariant(s[0]) + s[1..].ToLowerInvariant();
            else
                result[i] = Values[i];
        }
        return new StringColumn(Name, result);
    }

    // -- Trimming --

    public StringColumn Strip() => Trim();

    public StringColumn Trim()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Trim();
        return new StringColumn(Name, result);
    }

    public StringColumn LStrip()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.TrimStart();
        return new StringColumn(Name, result);
    }

    public StringColumn RStrip()
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.TrimEnd();
        return new StringColumn(Name, result);
    }

    // -- Padding --

    public StringColumn Pad(int width, char fillChar = ' ', bool leftPad = true)
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i] is { } s
                ? (leftPad ? s.PadLeft(width, fillChar) : s.PadRight(width, fillChar))
                : null;
        return new StringColumn(Name, result);
    }

    // -- Slicing --

    public StringColumn Slice(int start, int? length = null)
    {
        if (ShouldUseDict && start >= 0 && length.HasValue)
            return GetDict().Slice(Name, start, length.Value);

        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            if (Values[i] is { } s)
            {
                int actualStart = start < 0 ? Math.Max(0, s.Length + start) : Math.Min(start, s.Length);
                int actualLen = length ?? (s.Length - actualStart);
                actualLen = Math.Min(actualLen, s.Length - actualStart);
                result[i] = actualLen > 0 ? s.Substring(actualStart, actualLen) : "";
            }
        }
        return new StringColumn(Name, result);
    }

    // -- Metrics --

    public Column<int> Len()
    {
        if (ShouldUseDict) return GetDict().Len(Name);
        var result = new int?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Length;
        return Column<int>.FromNullable(Name, result);
    }

    public Column<int> Count(string substring)
    {
        var result = new int?[Length];
        for (int i = 0; i < Length; i++)
        {
            if (Values[i] is { } s)
            {
                if (substring.Length == 0)
                {
                    // Convention: empty substring occurs at every position + end = Length + 1
                    result[i] = s.Length + 1;
                    continue;
                }
                int count = 0, idx = 0;
                while ((idx = s.IndexOf(substring, idx, StringComparison.Ordinal)) >= 0)
                {
                    count++;
                    idx += substring.Length;
                }
                result[i] = count;
            }
        }
        return Column<int>.FromNullable(Name, result);
    }

    public Column<int> Find(string substring)
    {
        var result = new int?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.IndexOf(substring, StringComparison.Ordinal) ?? null;
        return Column<int>.FromNullable(Name, result);
    }

    // -- Split / Join / Repeat / Cat --

    public StringColumn Repeat(int times)
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i] is { } s ? string.Concat(Enumerable.Repeat(s, times)) : null;
        return new StringColumn(Name, result);
    }

    /// <summary>
    /// Concatenate all non-null strings in the column with a separator.
    /// </summary>
    public string Cat(string separator = "")
    {
        return string.Join(separator, Values.Where(v => v is not null));
    }

    /// <summary>
    /// Split each string by separator and return a DataFrame with one column per split part.
    /// Columns are named {Name}_0, {Name}_1, etc.
    /// </summary>
    public DataFrame Split(string separator, int maxParts = -1)
    {
        // First pass: determine max number of parts
        int maxCols = 0;
        var splitResults = new string?[Length][];
        for (int i = 0; i < Length; i++)
        {
            if (Values[i] is { } s)
            {
                var parts = maxParts > 0 ? s.Split(separator, maxParts) : s.Split(separator);
                splitResults[i] = parts;
                if (parts.Length > maxCols) maxCols = parts.Length;
            }
            else
            {
                splitResults[i] = [];
            }
        }

        // Build columns
        var columns = new List<IColumn>();
        for (int c = 0; c < maxCols; c++)
        {
            var colValues = new string?[Length];
            for (int r = 0; r < Length; r++)
            {
                var parts = splitResults[r];
                colValues[r] = parts is { Length: > 0 } && c < parts.Length ? parts[c]?.Trim() : null;
            }
            columns.Add(new StringColumn($"{Name}_{c}", colValues));
        }

        return new DataFrame(columns);
    }

    /// <summary>Pad strings with leading zeros to width. Like pandas str.zfill().</summary>
    public StringColumn ZFill(int width)
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i]?.PadLeft(width, '0');
        return new StringColumn(Name, result);
    }

    /// <summary>Center strings within width, padded with fillchar. Like pandas str.center().</summary>
    public StringColumn Center(int width, char fillChar = ' ')
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            var s = _column[i];
            if (s is null) { result[i] = null; continue; }
            if (s.Length >= width) { result[i] = s; continue; }
            int totalPad = width - s.Length;
            int leftPad = totalPad / 2;
            int rightPad = totalPad - leftPad;
            result[i] = new string(fillChar, leftPad) + s + new string(fillChar, rightPad);
        }
        return new StringColumn(Name, result);
    }

    /// <summary>Left-justify strings within width. Like pandas str.ljust().</summary>
    public StringColumn LJust(int width, char fillChar = ' ')
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i]?.PadRight(width, fillChar);
        return new StringColumn(Name, result);
    }

    /// <summary>Right-justify strings within width. Like pandas str.rjust().</summary>
    public StringColumn RJust(int width, char fillChar = ' ')
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = _column[i]?.PadLeft(width, fillChar);
        return new StringColumn(Name, result);
    }

    // -- Case-insensitive operations --

    /// <summary>
    /// Case-insensitive contains check using OrdinalIgnoreCase.
    /// </summary>
    public Column<bool> ContainsIgnoreCase(string substring)
    {
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Contains(substring, StringComparison.OrdinalIgnoreCase);
        return Column<bool>.FromNullable(Name, result);
    }

    /// <summary>
    /// Case-insensitive starts-with check using OrdinalIgnoreCase.
    /// </summary>
    public Column<bool> StartsWithIgnoreCase(string prefix)
    {
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.StartsWith(prefix, StringComparison.OrdinalIgnoreCase);
        return Column<bool>.FromNullable(Name, result);
    }

    /// <summary>
    /// Case-insensitive ends-with check using OrdinalIgnoreCase.
    /// </summary>
    public Column<bool> EndsWithIgnoreCase(string suffix)
    {
        var result = new bool?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.EndsWith(suffix, StringComparison.OrdinalIgnoreCase);
        return Column<bool>.FromNullable(Name, result);
    }

    /// <summary>
    /// Case-insensitive string replacement using OrdinalIgnoreCase.
    /// </summary>
    public StringColumn ReplaceIgnoreCase(string oldValue, string newValue)
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
            result[i] = Values[i]?.Replace(oldValue, newValue, StringComparison.OrdinalIgnoreCase);
        return new StringColumn(Name, result);
    }

    // -- Unicode normalization --

    /// <summary>
    /// Normalize Unicode strings. Skips ASCII-only strings as they are already normalized.
    /// </summary>
    public StringColumn NormalizeUnicode(NormalizationForm form = NormalizationForm.FormC)
    {
        var result = new string?[Length];
        for (int i = 0; i < Length; i++)
        {
            var s = Values[i];
            if (s is null) { result[i] = null; continue; }
            // ASCII-only strings are already in any normalization form
            if (IsAscii(s))
            {
                result[i] = s;
                continue;
            }
            result[i] = s.IsNormalized(form) ? s : s.Normalize(form);
        }
        return new StringColumn(Name, result);
    }

    private static bool IsAscii(string s)
    {
        for (int i = 0; i < s.Length; i++)
            if (s[i] > 127) return false;
        return true;
    }
}
