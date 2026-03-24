using AngleSharp;
using AngleSharp.Dom;
using PandaSharp.Column;

namespace PandaSharp.IO;

public static class HtmlTableReader
{
    /// <summary>
    /// Read HTML tables from a string. Returns one DataFrame per table found.
    /// Like pandas.read_html().
    /// </summary>
    public static async Task<List<DataFrame>> ReadHtmlAsync(string html, int? tableIndex = null)
    {
        var config = Configuration.Default;
        var context = BrowsingContext.New(config);
        var document = await context.OpenAsync(req => req.Content(html));

        var tables = document.QuerySelectorAll("table");
        var results = new List<DataFrame>();

        for (int t = 0; t < tables.Length; t++)
        {
            if (tableIndex.HasValue && t != tableIndex.Value) continue;
            results.Add(ParseTable(tables[t]));
        }

        return results;
    }

    public static List<DataFrame> ReadHtml(string html, int? tableIndex = null)
        => ReadHtmlAsync(html, tableIndex).GetAwaiter().GetResult();

    /// <summary>Read HTML tables from a URL.</summary>
    public static async Task<List<DataFrame>> ReadHtmlFromUrlAsync(string url, int? tableIndex = null)
    {
        using var http = new HttpClient();
        var html = await http.GetStringAsync(url);
        return await ReadHtmlAsync(html, tableIndex);
    }

    /// <summary>Read first HTML table from a file.</summary>
    public static DataFrame ReadHtmlFile(string path, int tableIndex = 0)
    {
        var html = File.ReadAllText(path);
        var tables = ReadHtml(html, tableIndex);
        return tables.Count > 0 ? tables[0] : new DataFrame();
    }

    private static DataFrame ParseTable(IElement table)
    {
        var rows = table.QuerySelectorAll("tr").ToList();
        if (rows.Count == 0) return new DataFrame();

        // First row with <th> or first <tr> is the header
        var headerRow = rows[0];
        var headerCells = headerRow.QuerySelectorAll("th").ToList();
        bool hasHeader = headerCells.Count > 0;
        if (!hasHeader)
            headerCells = headerRow.QuerySelectorAll("td").ToList();

        var columnNames = headerCells.Select((cell, i) =>
        {
            var text = cell.TextContent.Trim();
            return string.IsNullOrEmpty(text) ? $"Column{i}" : text;
        }).ToArray();

        int dataStart = hasHeader ? 1 : 0;
        int colCount = columnNames.Length;

        // Read data
        var rawData = new List<string?[]>();
        for (int r = dataStart; r < rows.Count; r++)
        {
            var cells = rows[r].QuerySelectorAll("td").ToList();
            var row = new string?[colCount];
            for (int c = 0; c < Math.Min(cells.Count, colCount); c++)
            {
                var text = cells[c].TextContent.Trim();
                row[c] = string.IsNullOrEmpty(text) ? null : text;
            }
            rawData.Add(row);
        }

        // Infer types and build columns
        var columns = new List<IColumn>();
        for (int c = 0; c < colCount; c++)
        {
            var samples = rawData.Select(row => row[c]).Take(100).ToList();
            var type = TypeInference.InferType(samples);
            columns.Add(CsvReader.BuildColumnPublic(columnNames[c], type, rawData, c, null));
        }

        return new DataFrame(columns);
    }
}
