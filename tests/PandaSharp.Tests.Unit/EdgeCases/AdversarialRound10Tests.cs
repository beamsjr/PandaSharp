using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.GroupBy;
using PandaSharp.IO;
using PandaSharp.Statistics;
using Xunit;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class AdversarialRound10Tests
{
    // ===============================================================
    // Bug 135: AvroWriter.EscapeJsonString does not escape control
    //          characters. Column names containing \n, \t, \0 etc.
    //          produce invalid JSON in the Avro schema, causing
    //          the Avro reader to throw a JsonException on read-back.
    // ===============================================================

    [Fact]
    public void AvroRoundTrip_ColumnNameWithNewline_ShouldSurvive()
    {
        var col = new Column<double>("col\nname", new double[] { 1.0, 2.0 });
        var df = new DataFrame(col);

        var path = Path.Combine(Path.GetTempPath(), $"avro_newline_colname_{Guid.NewGuid()}.avro");
        try
        {
            AvroWriter.Write(df, path);
            var loaded = AvroReader.Read(path);

            loaded.ColumnNames.Should().Contain("col\nname");
            var loadedCol = loaded["col\nname"];
            ((double)loadedCol.GetObject(0)!).Should().Be(1.0);
            ((double)loadedCol.GetObject(1)!).Should().Be(2.0);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===============================================================
    // Bug 136: JsonWriter.WriteValue calls Utf8JsonWriter.WriteNumberValue
    //          with double.NaN, which throws ArgumentException because
    //          JSON does not support NaN as a numeric value.
    // ===============================================================

    [Fact]
    public void JsonWriter_NaN_ShouldNotThrow()
    {
        var col = Column<double>.FromNullable("Val", new double?[] { 1.0, double.NaN, 3.0 });
        var df = new DataFrame(col);

        // This should not throw; NaN should be written as null or "NaN" string
        var act = () => JsonWriter.WriteString(df);
        act.Should().NotThrow("JsonWriter should handle NaN gracefully instead of throwing");
    }

    [Fact]
    public void JsonWriter_Infinity_ShouldNotThrow()
    {
        var col = new Column<double>("Val", new double[] { double.PositiveInfinity, double.NegativeInfinity });
        var df = new DataFrame(col);

        var act = () => JsonWriter.WriteString(df);
        act.Should().NotThrow("JsonWriter should handle Infinity gracefully instead of throwing");
    }

    // ===============================================================
    // Bug 137: CSV header parsing uses ReadLine() instead of ReadRecord()
    //          for header lines. A column name containing a newline
    //          (properly quoted as "col\nname") breaks because ReadLine
    //          splits at the newline, reading only the first part.
    // ===============================================================

    [Fact]
    public void CsvRoundTrip_ColumnNameWithNewline_ShouldSurvive()
    {
        var col = new Column<double>("col\nname", new double[] { 1.0, 2.0 });
        var df = new DataFrame(col);

        var path = Path.Combine(Path.GetTempPath(), $"csv_newline_colname_{Guid.NewGuid()}.csv");
        try
        {
            CsvWriter.Write(df, path);
            var loaded = CsvReader.Read(path);

            loaded.ColumnNames.Should().HaveCount(1);
            loaded.ColumnNames[0].Should().Be("col\nname",
                "CSV reader should handle quoted column names with embedded newlines");
            loaded.RowCount.Should().Be(2);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===============================================================
    // Bug 138: CSV header parsing uses ReadLine() in the ReadWithSchema
    //          path too (line 471-472), same root cause as Bug 137 but
    //          on the fast schema path.
    // ===============================================================

    [Fact]
    public void CsvRoundTrip_ColumnNameWithComma_ShouldSurvive()
    {
        // Column name with comma must be quoted in CSV header
        var col = new Column<double>("col,name", new double[] { 1.0, 2.0 });
        var df = new DataFrame(col);

        var path = Path.Combine(Path.GetTempPath(), $"csv_comma_colname_{Guid.NewGuid()}.csv");
        try
        {
            CsvWriter.Write(df, path);
            var loaded = CsvReader.Read(path);

            loaded.ColumnNames.Should().HaveCount(1,
                "a single column name containing a comma should not be split into two columns");
            loaded.ColumnNames[0].Should().Be("col,name");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===============================================================
    // Bug 139: CSV header parsing does not handle quoted column names
    //          containing double-quote characters. The writer properly
    //          escapes "col""name" but the reader's header parsing via
    //          ReadLine+ParseLine for header should handle this — let's
    //          verify.
    // ===============================================================

    [Fact]
    public void CsvRoundTrip_ColumnNameWithQuote_ShouldSurvive()
    {
        var col = new Column<double>("col\"name", new double[] { 1.0, 2.0 });
        var df = new DataFrame(col);

        var path = Path.Combine(Path.GetTempPath(), $"csv_quote_colname_{Guid.NewGuid()}.csv");
        try
        {
            CsvWriter.Write(df, path);
            var loaded = CsvReader.Read(path);

            loaded.ColumnNames.Should().HaveCount(1);
            loaded.ColumnNames[0].Should().Be("col\"name",
                "CSV round-trip should preserve column names with embedded quotes");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===============================================================
    // Bug 140: AvroWriter.EscapeJsonString doesn't escape tab character.
    //          Tab (\t) is a JSON control character that must be escaped
    //          as \\t. The current implementation only escapes backslash
    //          and double-quote.
    // ===============================================================

    [Fact]
    public void AvroRoundTrip_ColumnNameWithTab_ShouldSurvive()
    {
        var col = new Column<int>("col\tname", new int[] { 10, 20 });
        var df = new DataFrame(col);

        var path = Path.Combine(Path.GetTempPath(), $"avro_tab_colname_{Guid.NewGuid()}.avro");
        try
        {
            AvroWriter.Write(df, path);
            var loaded = AvroReader.Read(path);

            loaded.ColumnNames.Should().Contain("col\tname");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ===============================================================
    // Bug 141: JsonWriter crashes on float NaN/Infinity too.
    //          Same root cause as Bug 136 but for float columns.
    // ===============================================================

    [Fact]
    public void JsonWriter_FloatNaN_ShouldNotThrow()
    {
        var col = new Column<float>("Val", new float[] { float.NaN, float.PositiveInfinity, float.NegativeInfinity });
        var df = new DataFrame(col);

        var act = () => JsonWriter.WriteString(df);
        act.Should().NotThrow("JsonWriter should handle float NaN/Infinity gracefully");
    }
}
