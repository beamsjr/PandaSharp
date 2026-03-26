using System.Text.Json;

namespace PandaSharp.IO;

public static class JsonWriter
{
    public static void Write(DataFrame df, string path, JsonOrient orient = JsonOrient.Records)
    {
        var json = WriteString(df, orient);
        File.WriteAllText(path, json);
    }

    public static string WriteString(DataFrame df, JsonOrient orient = JsonOrient.Records)
    {
        var options = new JsonWriterOptions { Indented = true };
        using var stream = new MemoryStream();
        using (var writer = new Utf8JsonWriter(stream, options))
        {
            if (orient == JsonOrient.Records)
                WriteRecords(writer, df);
            else
                WriteColumns(writer, df);
        }
        return System.Text.Encoding.UTF8.GetString(stream.ToArray());
    }

    private static void WriteRecords(Utf8JsonWriter writer, DataFrame df)
    {
        writer.WriteStartArray();
        for (int r = 0; r < df.RowCount; r++)
        {
            writer.WriteStartObject();
            foreach (var name in df.ColumnNames)
            {
                writer.WritePropertyName(name);
                WriteValue(writer, df[name].GetObject(r));
            }
            writer.WriteEndObject();
        }
        writer.WriteEndArray();
    }

    private static void WriteColumns(Utf8JsonWriter writer, DataFrame df)
    {
        writer.WriteStartObject();
        foreach (var name in df.ColumnNames)
        {
            writer.WritePropertyName(name);
            writer.WriteStartArray();
            var col = df[name];
            for (int r = 0; r < col.Length; r++)
                WriteValue(writer, col.GetObject(r));
            writer.WriteEndArray();
        }
        writer.WriteEndObject();
    }

    private static void WriteValue(Utf8JsonWriter writer, object? value)
    {
        switch (value)
        {
            case null:
                writer.WriteNullValue();
                break;
            case int i:
                writer.WriteNumberValue(i);
                break;
            case long l:
                writer.WriteNumberValue(l);
                break;
            case float f:
                if (float.IsNaN(f) || float.IsInfinity(f))
                    writer.WriteNullValue();
                else
                    writer.WriteNumberValue(f);
                break;
            case double d:
                if (double.IsNaN(d) || double.IsInfinity(d))
                    writer.WriteNullValue();
                else
                    writer.WriteNumberValue(d);
                break;
            case bool b:
                writer.WriteBooleanValue(b);
                break;
            case DateTime dt:
                writer.WriteStringValue(dt.ToString("yyyy-MM-ddTHH:mm:ss"));
                break;
            case string s:
                writer.WriteStringValue(s);
                break;
            default:
                writer.WriteStringValue(value.ToString());
                break;
        }
    }
}
