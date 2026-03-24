using System.Globalization;
using System.IO.Compression;
using PandaSharp.Column;

namespace PandaSharp.IO;

public static class CsvReader
{
    public static DataFrame Read(string path, CsvReadOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var stream = File.OpenRead(path);

        // Auto-detect gzip by extension
        if (path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase))
        {
            using var gzip = new GZipStream(stream, CompressionMode.Decompress);
            using var memStream = new MemoryStream();
            gzip.CopyTo(memStream);
            memStream.Position = 0;
            return Read(memStream, options);
        }

        return Read(stream, options);
    }

    public static DataFrame Read(Stream stream, CsvReadOptions? options = null)
    {
        options ??= new CsvReadOptions();

        // Schema fast path: skip inference, parse directly into typed builders
        if (options.Schema is not null)
            return ReadWithSchema(stream, options);

        // Two-phase approach: read entire stream once, then:
        // (1) sample first N rows for type inference
        // (2) parse all data using the inferred schema's typed builders
        // This avoids storing all rows as intermediate string arrays
        return ReadWithInferredSchema(stream, options);
    }

    private static DataFrame ReadWithInferredSchema(Stream stream, CsvReadOptions options)
    {
        // Read entire stream as bytes once — avoids double encoding (string → bytes)
        byte[] rawBytes;
        using (var memStream = new MemoryStream())
        {
            stream.CopyTo(memStream);
            rawBytes = memStream.ToArray();
        }
        // Skip BOM if present (StreamReader does this automatically, but we're reading raw)
        int bomLen = 0;
        var preamble = options.Encoding.GetPreamble();
        if (preamble.Length > 0 && rawBytes.Length >= preamble.Length)
        {
            bool hasBom = true;
            for (int b = 0; b < preamble.Length; b++)
                if (rawBytes[b] != preamble[b]) { hasBom = false; break; }
            if (hasBom) bomLen = preamble.Length;
        }
        var content = options.Encoding.GetString(rawBytes, bomLen, rawBytes.Length - bomLen);

        // Phase 1: Sample first N lines for type inference using StringReader
        using var sampleReader = new StringReader(content);

        for (int i = 0; i < options.SkipRows; i++)
            sampleReader.ReadLine();

        string[] columnNames;
        if (options.HasHeader)
        {
            var headerLine = sampleReader.ReadLine()
                ?? throw new InvalidDataException("CSV file is empty.");
            columnNames = ParseLine(headerLine, options.Delimiter, options.QuoteChar, options.StrictQuoting);
        }
        else
        {
            var firstLine = sampleReader.ReadLine()
                ?? throw new InvalidDataException("CSV file is empty.");
            columnNames = Enumerable.Range(0, ParseLine(firstLine, options.Delimiter, options.QuoteChar, options.StrictQuoting).Length)
                .Select(i => $"Column{i}").ToArray();
        }

        var sampleData = new List<string?[]>();
        for (int s = 0; s < options.SampleRows; s++)
        {
            var line = ReadRecord(sampleReader, options.QuoteChar);
            if (line is null) break;
            if (string.IsNullOrWhiteSpace(line)) { s--; continue; }
            if (options.CommentChar.HasValue && line[0] == options.CommentChar.Value) { s--; continue; }

            var fields = ParseLine(line, options.Delimiter, options.QuoteChar, options.StrictQuoting);
            var normalized = new string?[columnNames.Length];
            for (int i = 0; i < Math.Min(fields.Length, columnNames.Length); i++)
                normalized[i] = IsNullValue(fields[i], options.NullValues) ? null : fields[i];
            sampleData.Add(normalized);
        }

        // Infer types from sample
        var schema = new (string Name, Type Type)[columnNames.Length];
        for (int c = 0; c < columnNames.Length; c++)
        {
            if (options.ColumnTypes?.TryGetValue(columnNames[c], out var overrideType) == true)
            {
                schema[c] = (columnNames[c], overrideType);
                continue;
            }
            var samples = sampleData.Select(r => r[c]).ToList();
            schema[c] = (columnNames[c], TypeInference.InferType(samples, options.DateFormat));
        }

        // Phase 2: Parse ALL data using the schema's typed builders — reuse original bytes
        using var dataStream = new MemoryStream(rawBytes);
        var schemaOptions = new CsvReadOptions
        {
            Delimiter = options.Delimiter,
            HasHeader = options.HasHeader,
            QuoteChar = options.QuoteChar,
            CommentChar = options.CommentChar,
            Encoding = options.Encoding,
            SkipRows = options.SkipRows,
            NullValues = options.NullValues,
            DateFormat = options.DateFormat,
            StrictQuoting = options.StrictQuoting,
            Schema = schema
        };
        return ReadWithSchema(dataStream, schemaOptions);
    }

    public static IEnumerable<DataFrame> ReadChunked(string path, int chunkSize, CsvReadOptions? options = null)
    {
        options ??= new CsvReadOptions();
        using var stream = File.OpenRead(path);
        using var reader = new StreamReader(stream, options.Encoding);

        for (int i = 0; i < options.SkipRows; i++)
            reader.ReadLine();

        string[] columnNames;
        var chunk = new List<string?[]>();

        if (options.HasHeader)
        {
            var headerLine = reader.ReadLine()
                ?? throw new InvalidDataException("CSV file is empty.");
            columnNames = ParseLine(headerLine, options.Delimiter, options.QuoteChar, options.StrictQuoting);
        }
        else
        {
            // No header: read first data line to determine column count, then buffer it
            var firstLine = reader.ReadLine()
                ?? throw new InvalidDataException("CSV file is empty.");
            var firstFields = ParseLine(firstLine, options.Delimiter, options.QuoteChar, options.StrictQuoting);
            columnNames = Enumerable.Range(0, firstFields.Length).Select(i => $"Column{i}").ToArray();

            // Buffer the first line so it's included in the first chunk
            var normalized = new string?[columnNames.Length];
            for (int i = 0; i < Math.Min(firstFields.Length, columnNames.Length); i++)
                normalized[i] = IsNullValue(firstFields[i], options.NullValues) ? null : firstFields[i];
            chunk.Add(normalized);
        }

        string? line;
        while ((line = ReadRecord(reader, options.QuoteChar)) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (options.CommentChar.HasValue && line[0] == options.CommentChar.Value) continue;

            var fields = ParseLine(line, options.Delimiter, options.QuoteChar, options.StrictQuoting);
            var normalized = new string?[columnNames.Length];
            for (int i = 0; i < Math.Min(fields.Length, columnNames.Length); i++)
                normalized[i] = IsNullValue(fields[i], options.NullValues) ? null : fields[i];
            chunk.Add(normalized);

            if (chunk.Count >= chunkSize)
            {
                yield return BuildDataFrame(columnNames, chunk, options);
                chunk.Clear();
            }
        }

        if (chunk.Count > 0)
            yield return BuildDataFrame(columnNames, chunk, options);
    }

    private static DataFrame BuildDataFrame(string[] columnNames, List<string?[]> rawData, CsvReadOptions options)
    {
        var types = new Type[columnNames.Length];
        for (int c = 0; c < columnNames.Length; c++)
        {
            if (options.ColumnTypes?.TryGetValue(columnNames[c], out var t) == true) { types[c] = t; continue; }
            var samples = rawData.Select(r => r[c]).Take(options.SampleRows).ToList();
            types[c] = TypeInference.InferType(samples, options.DateFormat);
        }
        var columns = new IColumn[columnNames.Length];
        for (int c = 0; c < columnNames.Length; c++)
            columns[c] = BuildColumn(columnNames[c], types[c], rawData, c, options.DateFormat);
        return new DataFrame(columns);
    }

    /// <summary>Build a typed column from raw string data. Used by CSV, Excel, and HTML readers.</summary>
    internal static IColumn BuildColumnPublic(string name, Type type, IList<string?[]> rawData, int colIndex, string? dateFormat)
        => BuildColumn(name, type, rawData, colIndex, dateFormat);

    private static IColumn BuildColumn(string name, Type type, IList<string?[]> rawData, int colIndex, string? dateFormat)
    {
        int rows = rawData.Count;

        if (type == typeof(int))
        {
            var values = new int?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? int.Parse(s, CultureInfo.InvariantCulture) : null;
            return Column<int>.FromNullable(name, values);
        }
        if (type == typeof(long))
        {
            var values = new long?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? long.Parse(s, CultureInfo.InvariantCulture) : null;
            return Column<long>.FromNullable(name, values);
        }
        if (type == typeof(double))
        {
            var values = new double?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? double.Parse(s, CultureInfo.InvariantCulture) : null;
            return Column<double>.FromNullable(name, values);
        }
        if (type == typeof(bool))
        {
            var values = new bool?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? bool.Parse(s) : null;
            return Column<bool>.FromNullable(name, values);
        }
        if (type == typeof(DateTime))
        {
            var values = new DateTime?[rows];
            for (int r = 0; r < rows; r++)
            {
                if (rawData[r][colIndex] is { } s)
                {
                    values[r] = dateFormat is not null
                        ? DateTime.ParseExact(s, dateFormat, CultureInfo.InvariantCulture)
                        : DateTime.Parse(s, CultureInfo.InvariantCulture);
                }
            }
            return Column<DateTime>.FromNullable(name, values);
        }

        // String fallback
        var strValues = new string?[rows];
        for (int r = 0; r < rows; r++)
            strValues[r] = rawData[r][colIndex];
        return new StringColumn(name, strValues);
    }

    internal static string[] ParseLine(string line, char delimiter, char quoteChar, bool strictQuoting = false)
    {
        if (line.Length == 0) return [""];
        var fields = new List<string>();
        int i = 0;
        bool lastWasDelimiter = false;
        while (i < line.Length)
        {
            lastWasDelimiter = false;

            if (line[i] == quoteChar)
            {
                // Quoted field — fast path: scan for closing quote without allocating StringBuilder
                i++; // skip opening quote
                int start = i;
                bool closedProperly = false;
                bool hasEscapedQuotes = false;

                // First pass: scan to find closing quote and check for escaped quotes
                int scanEnd = i;
                while (scanEnd < line.Length)
                {
                    if (line[scanEnd] == quoteChar)
                    {
                        if (scanEnd + 1 < line.Length && line[scanEnd + 1] == quoteChar)
                        {
                            hasEscapedQuotes = true;
                            scanEnd += 2;
                        }
                        else
                        {
                            closedProperly = true;
                            break;
                        }
                    }
                    else scanEnd++;
                }

                if (strictQuoting && !closedProperly)
                    throw new FormatException($"Unclosed quoted field at position {start - 1} in line: {line}");

                if (!hasEscapedQuotes)
                {
                    // Fast path: no escaped quotes, just slice the string
                    fields.Add(line[start..scanEnd]);
                }
                else
                {
                    // Slow path: unescape "" → " using StringBuilder
                    var field = new System.Text.StringBuilder(scanEnd - start);
                    for (int j = start; j < scanEnd; j++)
                    {
                        if (line[j] == quoteChar && j + 1 < scanEnd && line[j + 1] == quoteChar)
                        {
                            field.Append(quoteChar);
                            j++; // skip the second quote
                        }
                        else
                            field.Append(line[j]);
                    }
                    fields.Add(field.ToString());
                }

                i = closedProperly ? scanEnd + 1 : scanEnd; // skip past closing quote
                if (i < line.Length && line[i] == delimiter) { i++; lastWasDelimiter = true; }
            }
            else
            {
                // Unquoted field
                int start = i;
                while (i < line.Length && line[i] != delimiter) i++;
                fields.Add(line[start..i]);
                if (i < line.Length) { i++; lastWasDelimiter = true; }
            }
        }
        // If line ended with delimiter, add trailing empty field
        if (lastWasDelimiter) fields.Add("");
        return fields.ToArray();
    }

    /// <summary>
    /// Read a complete CSV record from a TextReader, handling quoted fields that span multiple lines.
    /// Returns null at end of stream.
    /// </summary>
    internal static string? ReadRecord(TextReader reader, char quoteChar)
    {
        var firstLine = reader.ReadLine();
        if (firstLine is null) return null;

        // Fast path: no quotes → single line record
        if (!firstLine.Contains(quoteChar))
            return firstLine;

        // Count unescaped quotes. If odd, the record spans multiple lines.
        int quoteCount = 0;
        for (int i = 0; i < firstLine.Length; i++)
        {
            if (firstLine[i] == quoteChar)
            {
                quoteCount++;
                // Skip escaped quotes ""
                if (i + 1 < firstLine.Length && firstLine[i + 1] == quoteChar)
                {
                    i++; // skip the pair
                    quoteCount++; // still even
                }
            }
        }

        // Even number of quotes → record is complete
        if (quoteCount % 2 == 0)
            return firstLine;

        // Odd quotes → record continues on next line(s)
        var sb = new System.Text.StringBuilder(firstLine.Length * 2);
        sb.Append(firstLine);

        while (true)
        {
            var nextLine = reader.ReadLine();
            if (nextLine is null)
                break; // EOF inside quoted field — best effort

            sb.Append('\n');
            sb.Append(nextLine);

            // Recount quotes in the accumulated record
            for (int i = 0; i < nextLine.Length; i++)
            {
                if (nextLine[i] == quoteChar)
                {
                    quoteCount++;
                    if (i + 1 < nextLine.Length && nextLine[i + 1] == quoteChar)
                    {
                        i++;
                        quoteCount++;
                    }
                }
            }

            // Even quotes → record is now complete
            if (quoteCount % 2 == 0)
                break;
        }

        return sb.ToString();
    }

    private static bool IsNullValue(string value, string[] nullValues)
    {
        foreach (var nv in nullValues)
            if (value == nv) return true;
        return false;
    }

    private static bool IsNullSpan(ReadOnlySpan<char> value, string[] nullValues)
    {
        foreach (var nv in nullValues)
            if (value.SequenceEqual(nv.AsSpan())) return true;
        return false;
    }

    /// <summary>
    /// Fast path: user provides schema, skip inference entirely.
    /// Parses numeric fields directly without intermediate string[] per row for numeric columns.
    /// </summary>
    private static DataFrame ReadWithSchema(Stream stream, CsvReadOptions options)
    {
        using var reader = new StreamReader(stream, options.Encoding);
        var schema = options.Schema!;

        for (int i = 0; i < options.SkipRows; i++)
            reader.ReadLine();

        // Skip header if present
        if (options.HasHeader)
            reader.ReadLine();

        int colCount = schema.Length;

        // Pre-allocate typed builders
        var intBuilders = new List<int?>[colCount];
        var longBuilders = new List<long?>[colCount];
        var doubleBuilders = new List<double?>[colCount];
        var floatBuilders = new List<float?>[colCount];
        var boolBuilders = new List<bool?>[colCount];
        var stringBuilders = new List<string?>[colCount];

        for (int c = 0; c < colCount; c++)
        {
            var t = schema[c].Type;
            if (t == typeof(int)) intBuilders[c] = new List<int?>(1024);
            else if (t == typeof(long)) longBuilders[c] = new List<long?>(1024);
            else if (t == typeof(double)) doubleBuilders[c] = new List<double?>(1024);
            else if (t == typeof(float)) floatBuilders[c] = new List<float?>(1024);
            else if (t == typeof(bool)) boolBuilders[c] = new List<bool?>(1024);
            else stringBuilders[c] = new List<string?>(1024);
        }

        // Parse using Span<char> splitting to avoid string allocation for numeric fields
        var nullSet = new HashSet<string>(options.NullValues);
        char delim = options.Delimiter;
        char quote = options.QuoteChar;
        string? line;

        // Fast path: use ReadLine directly instead of ReadRecord when possible.
        // ReadRecord handles multi-line quoted fields but adds overhead for the common
        // single-line case. We try ReadLine first and fall back to ReadRecord only when
        // we detect an incomplete quoted field (odd number of quotes).
        while ((line = reader.ReadLine()) is not null)
        {
            if (string.IsNullOrWhiteSpace(line)) continue;
            if (options.CommentChar.HasValue && line[0] == options.CommentChar.Value) continue;

            // Check if this line has an incomplete multi-line quoted field
            if (line.Contains(quote))
            {
                int qCount = 0;
                for (int ci = 0; ci < line.Length; ci++)
                    if (line[ci] == quote) qCount++;
                if (qCount % 2 != 0)
                {
                    // Incomplete — read continuation lines
                    var sb = new System.Text.StringBuilder(line);
                    while (qCount % 2 != 0)
                    {
                        var next = reader.ReadLine();
                        if (next is null) break;
                        sb.Append('\n').Append(next);
                        for (int ci = 0; ci < next.Length; ci++)
                            if (next[ci] == quote) qCount++;
                    }
                    line = sb.ToString();
                }
            }

            // Check if line contains quotes — if so, use full ParseLine for correctness
            if (line.Contains(quote))
            {
                var fields = ParseLine(line, delim, '"', options.StrictQuoting);
                for (int c = 0; c < Math.Min(fields.Length, colCount); c++)
                {
                    var f = fields[c];
                    bool iN = nullSet.Contains(f);
                    var t2 = schema[c].Type;
                    if (t2 == typeof(int)) intBuilders[c]!.Add(iN ? null : int.Parse(f, System.Globalization.CultureInfo.InvariantCulture));
                    else if (t2 == typeof(long)) longBuilders[c]!.Add(iN ? null : long.Parse(f, System.Globalization.CultureInfo.InvariantCulture));
                    else if (t2 == typeof(double)) doubleBuilders[c]!.Add(iN ? null : double.Parse(f, System.Globalization.CultureInfo.InvariantCulture));
                    else if (t2 == typeof(float)) floatBuilders[c]!.Add(iN ? null : float.Parse(f, System.Globalization.CultureInfo.InvariantCulture));
                    else if (t2 == typeof(bool)) boolBuilders[c]!.Add(iN ? null : bool.Parse(f));
                    else stringBuilders[c]!.Add(iN ? null : f);
                }
                continue;
            }

            // Span-based field splitting for non-quoted lines
            ReadOnlySpan<char> lineSpan = line.AsSpan();
            int fieldStart = 0;
            for (int c = 0; c < colCount; c++)
            {
                int fieldEnd = fieldStart;
                while (fieldEnd < lineSpan.Length && lineSpan[fieldEnd] != delim)
                    fieldEnd++;

                var fieldSpan = lineSpan[fieldStart..fieldEnd];
                fieldStart = fieldEnd + 1; // skip delimiter

                var t = schema[c].Type;

                if (fieldSpan.Length == 0 || IsNullSpan(fieldSpan, options.NullValues))
                {
                    if (t == typeof(int)) intBuilders[c]!.Add(null);
                    else if (t == typeof(long)) longBuilders[c]!.Add(null);
                    else if (t == typeof(double)) doubleBuilders[c]!.Add(null);
                    else if (t == typeof(float)) floatBuilders[c]!.Add(null);
                    else if (t == typeof(bool)) boolBuilders[c]!.Add(null);
                    else stringBuilders[c]!.Add(null);
                }
                else if (t == typeof(int))
                    intBuilders[c]!.Add(int.Parse(fieldSpan, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture));
                else if (t == typeof(long))
                    longBuilders[c]!.Add(long.Parse(fieldSpan, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture));
                else if (t == typeof(double))
                    doubleBuilders[c]!.Add(double.Parse(fieldSpan, System.Globalization.NumberStyles.Float | System.Globalization.NumberStyles.AllowThousands, System.Globalization.CultureInfo.InvariantCulture));
                else if (t == typeof(float))
                    floatBuilders[c]!.Add(float.Parse(fieldSpan, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture));
                else if (t == typeof(bool))
                    boolBuilders[c]!.Add(bool.Parse(fieldSpan));
                else
                    stringBuilders[c]!.Add(fieldSpan.ToString()); // only strings allocate
            }
        }

        // Build columns
        var columns = new IColumn[colCount];
        for (int c = 0; c < colCount; c++)
        {
            var (name, t) = schema[c];
            if (t == typeof(int)) columns[c] = Column<int>.FromNullable(name, intBuilders[c]!.ToArray());
            else if (t == typeof(long)) columns[c] = Column<long>.FromNullable(name, longBuilders[c]!.ToArray());
            else if (t == typeof(double)) columns[c] = Column<double>.FromNullable(name, doubleBuilders[c]!.ToArray());
            else if (t == typeof(float)) columns[c] = Column<float>.FromNullable(name, floatBuilders[c]!.ToArray());
            else if (t == typeof(bool)) columns[c] = Column<bool>.FromNullable(name, boolBuilders[c]!.ToArray());
            else columns[c] = new StringColumn(name, stringBuilders[c]!.ToArray());
        }

        return new DataFrame(columns);
    }
}
