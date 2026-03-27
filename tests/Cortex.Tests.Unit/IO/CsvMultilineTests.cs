using System.Text;
using FluentAssertions;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class CsvMultilineTests
{
    private static Stream ToStream(string content) =>
        new MemoryStream(Encoding.UTF8.GetBytes(content));

    [Fact]
    public void MultilineQuotedField_SingleNewline()
    {
        var csv = "Name,Bio\nAlice,\"Hello\nWorld\"\nBob,Simple\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Bio")[0].Should().Be("Hello\nWorld");
        df.GetStringColumn("Bio")[1].Should().Be("Simple");
    }

    [Fact]
    public void MultilineQuotedField_MultipleNewlines()
    {
        var csv = "Text\n\"Line1\nLine2\nLine3\"\n\"Single\"\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Text")[0].Should().Be("Line1\nLine2\nLine3");
        df.GetStringColumn("Text")[1].Should().Be("Single");
    }

    [Fact]
    public void MultilineQuotedField_WithCommas()
    {
        var csv = "A,B\n\"hello,\nworld\",123\nfoo,456\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("A")[0].Should().Be("hello,\nworld");
        df.GetColumn<int>("B")[0].Should().Be(123);
    }

    [Fact]
    public void MultilineQuotedField_WithEscapedQuotes()
    {
        var csv = "Text\n\"She said \"\"hi\"\"\nthen left\"\n\"Normal\"\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Text")[0].Should().Be("She said \"hi\"\nthen left");
    }

    [Fact]
    public void MultilineQuotedField_MixedColumns()
    {
        var csv = "Id,Description,Value\n1,\"Multi\nLine\",100\n2,Single,200\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetColumn<int>("Id")[0].Should().Be(1);
        df.GetStringColumn("Description")[0].Should().Be("Multi\nLine");
        df.GetColumn<int>("Value")[0].Should().Be(100);
        df.GetColumn<int>("Id")[1].Should().Be(2);
    }

    [Fact]
    public void MultilineQuotedField_LastFieldInRow()
    {
        var csv = "A,B\nfoo,\"bar\nbaz\"\nqux,quux\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("B")[0].Should().Be("bar\nbaz");
    }

    [Fact]
    public void ReadRecord_CountsQuotesProperly()
    {
        // Regression: escaped quotes ("") should not cause false multiline detection
        var csv = "Text\n\"Has \"\"quotes\"\" inside\"\nNormal\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.GetStringColumn("Text")[0].Should().Be("Has \"quotes\" inside");
    }

    [Fact]
    public void CsvRoundTrip_WithNewlines()
    {
        var df = new Cortex.DataFrame(
            new Cortex.Column.StringColumn("Text", ["hello\nworld", "foo\nbar\nbaz"])
        );

        using var ms = new MemoryStream();
        CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);

        var loaded = CsvReader.Read(ms);
        loaded.RowCount.Should().Be(2);
        loaded.GetStringColumn("Text")[0].Should().Be("hello\nworld");
        loaded.GetStringColumn("Text")[1].Should().Be("foo\nbar\nbaz");
    }
}
