using System.Text;
using FluentAssertions;
using PandaSharp.IO;

namespace PandaSharp.Tests.Unit.IO;

public class CsvStressTests
{
    private static Stream ToStream(string content) =>
        new MemoryStream(Encoding.UTF8.GetBytes(content));

    [Fact]
    public void Csv_EmptyFile_Throws()
    {
        var act = () => CsvReader.Read(ToStream(""));
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void Csv_HeaderOnly_ZeroRows()
    {
        var df = CsvReader.Read(ToStream("A,B,C\n"));
        df.RowCount.Should().Be(0);
        df.ColumnNames.Should().Equal(["A", "B", "C"]);
    }

    [Fact]
    public void Csv_SingleColumn()
    {
        var df = CsvReader.Read(ToStream("Value\n1\n2\n3\n"));
        df.RowCount.Should().Be(3);
        df.ColumnCount.Should().Be(1);
    }

    [Fact]
    public void Csv_SingleRow()
    {
        var df = CsvReader.Read(ToStream("A,B\n1,x\n"));
        df.RowCount.Should().Be(1);
    }

    [Fact]
    public void Csv_ManyColumns()
    {
        var cols = string.Join(",", Enumerable.Range(0, 50).Select(i => $"Col{i}"));
        var vals = string.Join(",", Enumerable.Range(0, 50).Select(i => i.ToString()));
        var df = CsvReader.Read(ToStream($"{cols}\n{vals}\n"));

        df.ColumnCount.Should().Be(50);
        df.RowCount.Should().Be(1);
    }

    [Fact]
    public void Csv_SpecialCharactersInQuotes()
    {
        // Escaped quotes should work
        var df = CsvReader.Read(ToStream("Text\n\"Has \"\"quotes\"\"\"\n"));
        df.GetStringColumn("Text")[0].Should().Be("Has \"quotes\"");
    }

    [Fact]
    public void Csv_UnicodeContent()
    {
        var csv = "Name,City\nAlice,日本\nBob,München\n";
        var df = CsvReader.Read(ToStream(csv));
        df.GetStringColumn("City")[0].Should().Be("日本");
        df.GetStringColumn("City")[1].Should().Be("München");
    }

    [Fact]
    public void Csv_AllNullColumn()
    {
        var csv = "A,B\n1,\n2,\n3,\n";
        var df = CsvReader.Read(ToStream(csv));
        df["B"].NullCount.Should().Be(3);
    }

    [Fact]
    public void Csv_MixedNullValues()
    {
        var csv = "A\nNA\nN/A\nnull\nNULL\nNone\nactual\n";
        var df = CsvReader.Read(ToStream(csv));
        df["A"].NullCount.Should().Be(5);
        df.GetStringColumn("A")[5].Should().Be("actual");
    }

    [Fact]
    public void Csv_LargeFile_1000Rows()
    {
        var sb = new StringBuilder("Id,Value\n");
        for (int i = 0; i < 1000; i++)
            sb.AppendLine($"{i},{i * 1.5}");

        var df = CsvReader.Read(ToStream(sb.ToString()));
        df.RowCount.Should().Be(1000);
        df["Id"].DataType.Should().Be(typeof(int));
        df["Value"].DataType.Should().Be(typeof(double));
    }
}
