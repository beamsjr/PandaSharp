using System.Text;
using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class CsvTests
{
    private static Stream ToStream(string content) =>
        new MemoryStream(Encoding.UTF8.GetBytes(content));

    [Fact]
    public void ReadCsv_BasicFile_InfersTypes()
    {
        var csv = "Name,Age,Salary\nAlice,25,50000.5\nBob,30,62000.0\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(2);
        df.ColumnNames.Should().Equal(["Name", "Age", "Salary"]);
        df["Age"].DataType.Should().Be(typeof(int));
        df["Salary"].DataType.Should().Be(typeof(double));
        df["Name"].DataType.Should().Be(typeof(string));
    }

    [Fact]
    public void ReadCsv_WithNulls()
    {
        var csv = "A,B\n1,x\n,y\n3,\n";
        var df = CsvReader.Read(ToStream(csv));

        df.RowCount.Should().Be(3);
        df["A"].IsNull(1).Should().BeTrue();
        df["B"].IsNull(2).Should().BeTrue();
    }

    [Fact]
    public void ReadCsv_QuotedFields()
    {
        var csv = "Name,Bio\nAlice,\"Has a, comma\"\nBob,\"Says \"\"hello\"\"\"\n";
        var df = CsvReader.Read(ToStream(csv));

        df.GetStringColumn("Bio")[0].Should().Be("Has a, comma");
        df.GetStringColumn("Bio")[1].Should().Be("Says \"hello\"");
    }

    [Fact]
    public void ReadCsv_CustomDelimiter()
    {
        var tsv = "A\tB\n1\tx\n2\ty\n";
        var df = CsvReader.Read(ToStream(tsv), new CsvReadOptions { Delimiter = '\t' });

        df.RowCount.Should().Be(2);
        df.GetColumn<int>("A")[0].Should().Be(1);
    }

    [Fact]
    public void ReadCsv_CommentLines()
    {
        var csv = "A,B\n# this is a comment\n1,x\n2,y\n";
        var df = CsvReader.Read(ToStream(csv), new CsvReadOptions { CommentChar = '#' });

        df.RowCount.Should().Be(2);
    }

    [Fact]
    public void ReadCsv_BoolInference()
    {
        var csv = "Flag\ntrue\nfalse\ntrue\n";
        var df = CsvReader.Read(ToStream(csv));

        df["Flag"].DataType.Should().Be(typeof(bool));
        df.GetColumn<bool>("Flag")[0].Should().Be(true);
    }

    [Fact]
    public void WriteCsv_RoundTrips()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30]),
            new Column<double>("Score", [95.5, 87.3])
        );

        using var ms = new MemoryStream();
        CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);

        var df2 = CsvReader.Read(ms);
        df2.RowCount.Should().Be(2);
        df2.GetStringColumn("Name")[0].Should().Be("Alice");
        df2.GetColumn<int>("Age")[1].Should().Be(30);
        df2.GetColumn<double>("Score")[0].Should().Be(95.5);
    }

    [Fact]
    public void WriteCsv_HandlesNulls()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3])
        );

        using var ms = new MemoryStream();
        CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var content = new StreamReader(ms).ReadToEnd();

        content.Should().Contain("1\n\n3"); // null written as empty
    }

    [Fact]
    public void ReadCsv_ColumnTypeOverride()
    {
        var csv = "Id,Value\n1,2\n3,4\n";
        var df = CsvReader.Read(ToStream(csv), new CsvReadOptions
        {
            ColumnTypes = new Dictionary<string, Type> { ["Id"] = typeof(string) }
        });

        df["Id"].DataType.Should().Be(typeof(string));
        df.GetStringColumn("Id")[0].Should().Be("1");
    }

    // ===== Chunked reading =====

    [Fact]
    public void ReadChunked_WithHeader_SplitsIntoChunks()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, "Name,Age\nAlice,25\nBob,30\nCharlie,35\nDiana,28\nEve,42\n");

        var chunks = CsvReader.ReadChunked(path, chunkSize: 2).ToList();

        chunks.Should().HaveCount(3); // 2, 2, 1
        chunks[0].RowCount.Should().Be(2);
        chunks[1].RowCount.Should().Be(2);
        chunks[2].RowCount.Should().Be(1);

        chunks[0].GetStringColumn("Name")[0].Should().Be("Alice");
        chunks[2].GetStringColumn("Name")[0].Should().Be("Eve");

        File.Delete(path);
    }

    [Fact]
    public void ReadChunked_WithoutHeader_GeneratesColumnNames()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, "Alice,25\nBob,30\nCharlie,35\n");

        var chunks = CsvReader.ReadChunked(path, chunkSize: 2, new CsvReadOptions { HasHeader = false }).ToList();

        chunks.Should().HaveCount(2); // 2, 1
        chunks[0].RowCount.Should().Be(2);
        chunks[1].RowCount.Should().Be(1);
        chunks[0].ColumnNames.Should().Equal(["Column0", "Column1"]);
        chunks[0].GetStringColumn("Column0")[0].Should().Be("Alice");
        chunks[0].GetStringColumn("Column0")[1].Should().Be("Bob");
        chunks[1].GetStringColumn("Column0")[0].Should().Be("Charlie");

        File.Delete(path);
    }

    [Fact]
    public void ReadChunked_WithoutHeader_SingleChunk()
    {
        var path = Path.GetTempFileName();
        File.WriteAllText(path, "1,2,3\n4,5,6\n");

        var chunks = CsvReader.ReadChunked(path, chunkSize: 100, new CsvReadOptions { HasHeader = false }).ToList();

        chunks.Should().HaveCount(1);
        chunks[0].RowCount.Should().Be(2);
        chunks[0].ColumnNames.Should().Equal(["Column0", "Column1", "Column2"]);

        File.Delete(path);
    }

    [Fact]
    public void ReadChunked_WithoutHeader_PreservesAllRows()
    {
        var path = Path.GetTempFileName();
        var sb = new StringBuilder();
        for (int i = 0; i < 100; i++)
            sb.AppendLine($"{i},{i * 10}");
        File.WriteAllText(path, sb.ToString());

        var chunks = CsvReader.ReadChunked(path, chunkSize: 30, new CsvReadOptions { HasHeader = false }).ToList();
        var totalRows = chunks.Sum(c => c.RowCount);

        totalRows.Should().Be(100);
        chunks.Should().HaveCount(4); // 30, 30, 30, 10

        File.Delete(path);
    }
}
