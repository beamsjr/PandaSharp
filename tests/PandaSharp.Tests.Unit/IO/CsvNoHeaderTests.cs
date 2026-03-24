using System.Text;
using FluentAssertions;
using PandaSharp.IO;

namespace PandaSharp.Tests.Unit.IO;

public class CsvNoHeaderTests
{
    private static Stream ToStream(string content) =>
        new MemoryStream(Encoding.UTF8.GetBytes(content));

    [Fact]
    public void ReadCsv_NoHeader_GeneratesColumnNames()
    {
        var csv = "1,Alice,90.5\n2,Bob,85.0\n";
        var df = CsvReader.Read(ToStream(csv), new CsvReadOptions { HasHeader = false });

        df.RowCount.Should().Be(2);
        df.ColumnNames.Should().Equal(["Column0", "Column1", "Column2"]);
    }

    [Fact]
    public void ReadCsv_NoHeader_IncludesFirstRow()
    {
        var csv = "1,x\n2,y\n3,z\n";
        var df = CsvReader.Read(ToStream(csv), new CsvReadOptions { HasHeader = false });

        df.RowCount.Should().Be(3); // first line is data, not header
        df.GetColumn<int>("Column0")[0].Should().Be(1);
        df.GetStringColumn("Column1")[2].Should().Be("z");
    }

    [Fact]
    public void ReadCsv_NoHeader_TypeInference()
    {
        var csv = "1,2.5,true\n2,3.5,false\n";
        var df = CsvReader.Read(ToStream(csv), new CsvReadOptions { HasHeader = false });

        df["Column0"].DataType.Should().Be(typeof(int));
        df["Column1"].DataType.Should().Be(typeof(double));
        df["Column2"].DataType.Should().Be(typeof(bool));
    }
}
