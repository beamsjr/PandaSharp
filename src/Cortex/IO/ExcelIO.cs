using ClosedXML.Excel;
using Cortex.Column;

namespace Cortex.IO;

public class ExcelReadOptions
{
    /// <summary>Sheet name to read (null = first sheet).</summary>
    public string? SheetName { get; set; }
    /// <summary>Whether the first row is a header.</summary>
    public bool HasHeader { get; set; } = true;
}

public static class ExcelIO
{
    public static DataFrame ReadExcel(string path, ExcelReadOptions? options = null)
    {
        using var stream = File.OpenRead(path);
        return ReadExcel(stream, options);
    }

    public static DataFrame ReadExcel(Stream stream, ExcelReadOptions? options = null)
    {
        options ??= new ExcelReadOptions();
        using var workbook = new XLWorkbook(stream);
        var sheet = options.SheetName is not null
            ? workbook.Worksheet(options.SheetName)
            : workbook.Worksheets.First();

        var usedRange = sheet.RangeUsed();
        if (usedRange is null) return new DataFrame();

        int firstRow = usedRange.FirstRow().RowNumber();
        int lastRow = usedRange.LastRow().RowNumber();
        int firstCol = usedRange.FirstColumn().ColumnNumber();
        int lastCol = usedRange.LastColumn().ColumnNumber();

        int colCount = lastCol - firstCol + 1;
        int headerRow = firstRow;
        int dataStart = options.HasHeader ? firstRow + 1 : firstRow;
        int rowCount = lastRow - dataStart + 1;

        // Read column names
        var columnNames = new string[colCount];
        for (int c = 0; c < colCount; c++)
        {
            if (options.HasHeader)
                columnNames[c] = sheet.Cell(headerRow, firstCol + c).GetString();
            else
                columnNames[c] = $"Column{c}";

            if (string.IsNullOrEmpty(columnNames[c]))
                columnNames[c] = $"Column{c}";
        }

        // Read data as strings, then infer types
        var rawData = new string?[rowCount][];
        for (int r = 0; r < rowCount; r++)
        {
            rawData[r] = new string?[colCount];
            for (int c = 0; c < colCount; c++)
            {
                var cell = sheet.Cell(dataStart + r, firstCol + c);
                rawData[r][c] = cell.IsEmpty() ? null : cell.GetString();
            }
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

    public static void WriteExcel(DataFrame df, string path, string sheetName = "Sheet1")
    {
        using var workbook = new XLWorkbook();
        var sheet = workbook.Worksheets.Add(sheetName);

        // Write headers
        for (int c = 0; c < df.ColumnCount; c++)
            sheet.Cell(1, c + 1).Value = df.ColumnNames[c];

        // Write data
        for (int r = 0; r < df.RowCount; r++)
        {
            for (int c = 0; c < df.ColumnCount; c++)
            {
                var val = df[df.ColumnNames[c]].GetObject(r);
                if (val is not null)
                {
                    var cell = sheet.Cell(r + 2, c + 1);
                    switch (val)
                    {
                        case int i: cell.Value = i; break;
                        case long l: cell.Value = l; break;
                        case double d: cell.Value = d; break;
                        case float f: cell.Value = f; break;
                        case bool b: cell.Value = b; break;
                        case DateTime dt: cell.Value = dt; break;
                        default: cell.Value = val.ToString(); break;
                    }
                }
            }
        }

        workbook.SaveAs(path);
    }

    /// <summary>List sheet names in an Excel file.</summary>
    public static string[] ListSheets(string path)
    {
        using var workbook = new XLWorkbook(path);
        return workbook.Worksheets.Select(ws => ws.Name).ToArray();
    }
}
