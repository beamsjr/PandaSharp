using System.Text.Json;
using PandaSharp.Column;

namespace PandaSharp.IO;

public enum JsonOrient
{
    Records,   // [{"col1": val1, "col2": val2}, ...]
    Columns    // {"col1": [val1, val2, ...], "col2": [...]}
}

public static class JsonReader
{
    public static DataFrame Read(string path)
    {
        var json = File.ReadAllText(path);
        return ReadString(json);
    }

    /// <summary>
    /// Read a JSON Lines (.jsonl) file where each line is a separate JSON object.
    /// </summary>
    public static DataFrame ReadLines(string path)
    {
        var lines = File.ReadAllLines(path)
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();
        return ReadLinesFromStrings(lines);
    }

    /// <summary>
    /// Read JSON Lines from a string (each line is a JSON object).
    /// </summary>
    public static DataFrame ReadLinesString(string jsonLines)
    {
        var lines = jsonLines.Split('\n')
            .Where(l => !string.IsNullOrWhiteSpace(l))
            .ToArray();
        return ReadLinesFromStrings(lines);
    }

    private static DataFrame ReadLinesFromStrings(string[] lines)
    {
        if (lines.Length == 0) return new DataFrame();

        // Parse each line as a JSON object
        var json = "[" + string.Join(",", lines) + "]";
        return ReadString(json);
    }

    public static DataFrame ReadString(string json)
    {
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;

        if (root.ValueKind == JsonValueKind.Array)
            return ReadRecordOriented(root);
        if (root.ValueKind == JsonValueKind.Object)
            return ReadColumnOriented(root);

        throw new InvalidDataException("JSON must be an array (records) or object (columns).");
    }

    private static DataFrame ReadRecordOriented(JsonElement array)
    {
        var records = new List<Dictionary<string, string?>>();
        var allKeys = new List<string>();
        var keySet = new HashSet<string>();

        foreach (var element in array.EnumerateArray())
        {
            var record = new Dictionary<string, string?>();
            foreach (var prop in element.EnumerateObject())
            {
                if (keySet.Add(prop.Name)) allKeys.Add(prop.Name);
                record[prop.Name] = prop.Value.ValueKind == JsonValueKind.Null
                    ? null
                    : prop.Value.ToString();
            }
            records.Add(record);
        }

        // Build raw data
        var rawData = new List<string?[]>();
        foreach (var record in records)
        {
            var row = new string?[allKeys.Count];
            for (int c = 0; c < allKeys.Count; c++)
                row[c] = record.GetValueOrDefault(allKeys[c]);
            rawData.Add(row);
        }

        // Infer types and build columns
        var columns = new IColumn[allKeys.Count];
        for (int c = 0; c < allKeys.Count; c++)
        {
            var samples = rawData.Select(r => r[c]).ToList();
            var type = TypeInference.InferType(samples);
            columns[c] = BuildColumn(allKeys[c], type, rawData, c);
        }

        return new DataFrame(columns);
    }

    private static DataFrame ReadColumnOriented(JsonElement obj)
    {
        var columns = new List<IColumn>();

        foreach (var prop in obj.EnumerateObject())
        {
            if (prop.Value.ValueKind != JsonValueKind.Array)
                throw new InvalidDataException($"Column '{prop.Name}' must be an array.");

            var values = new List<string?>();
            foreach (var elem in prop.Value.EnumerateArray())
            {
                values.Add(elem.ValueKind == JsonValueKind.Null ? null : elem.ToString());
            }

            var type = TypeInference.InferType(values);
            var rawData = values.Select(v => new string?[] { v }).ToList();
            columns.Add(BuildColumn(prop.Name, type, rawData, 0));
        }

        return new DataFrame(columns);
    }

    private static IColumn BuildColumn(string name, Type type, List<string?[]> rawData, int colIndex)
    {
        int rows = rawData.Count;
        if (type == typeof(int))
        {
            var values = new int?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? int.Parse(s, System.Globalization.CultureInfo.InvariantCulture) : null;
            return Column<int>.FromNullable(name, values);
        }
        if (type == typeof(double))
        {
            var values = new double?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? double.Parse(s, System.Globalization.CultureInfo.InvariantCulture) : null;
            return Column<double>.FromNullable(name, values);
        }
        if (type == typeof(bool))
        {
            var values = new bool?[rows];
            for (int r = 0; r < rows; r++)
                values[r] = rawData[r][colIndex] is { } s ? bool.Parse(s) : null;
            return Column<bool>.FromNullable(name, values);
        }

        var strValues = new string?[rows];
        for (int r = 0; r < rows; r++)
            strValues[r] = rawData[r][colIndex];
        return new StringColumn(name, strValues);
    }
}
