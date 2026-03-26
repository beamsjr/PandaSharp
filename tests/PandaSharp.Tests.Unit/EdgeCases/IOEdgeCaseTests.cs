using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.IO;
using Xunit;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class CsvReaderEdgeCaseTests
{
    [Fact]
    public void Read_Empty_File_Throws()
    {
        using var stream = new MemoryStream(""u8.ToArray());

        var act = () => CsvReader.Read(stream);

        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void Read_Only_Headers_Returns_Zero_Rows()
    {
        using var stream = new MemoryStream("Name,Age,City\n"u8.ToArray());

        var df = CsvReader.Read(stream);

        df.ColumnCount.Should().Be(3);
        df.RowCount.Should().Be(0);
        df.ColumnNames.Should().Equal("Name", "Age", "City");
    }

    [Fact]
    public void Read_Single_Column()
    {
        var csv = "Value\n1\n2\n3\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var df = CsvReader.Read(stream);

        df.ColumnCount.Should().Be(1);
        df.RowCount.Should().Be(3);
    }

    [Fact]
    public void Read_Quoted_Fields_With_Newlines()
    {
        var csv = "Name,Bio\nAlice,\"Hello\nWorld\"\nBob,\"Simple\"\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var df = CsvReader.Read(stream);

        df.RowCount.Should().Be(2);
        var bio = df.GetStringColumn("Bio");
        bio[0].Should().Be("Hello\nWorld");
        bio[1].Should().Be("Simple");
    }
}

public class AvroRoundTripEdgeCaseTests
{
    [Fact]
    public void Avro_Empty_DataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>()),
            new StringColumn("B", Array.Empty<string?>()));

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(0);
        df2.ColumnCount.Should().Be(2);
    }

    [Fact]
    public void Avro_Single_Row()
    {
        var df = new DataFrame(
            new Column<int>("Id", [42]),
            new StringColumn("Name", ["test"]));

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(1);
        df2.GetColumn<int>("Id")[0].Should().Be(42);
        df2.GetStringColumn("Name")[0].Should().Be("test");
    }

    [Fact]
    public void Avro_Null_Only_Column()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Text", [null, null, null]));

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(3);
        for (int i = 0; i < 3; i++)
            df2["Text"].IsNull(i).Should().BeTrue();
    }

    [Fact]
    public void Avro_Very_Long_Strings()
    {
        var longStr = new string('x', 5000);
        var df = new DataFrame(
            new StringColumn("Data", [longStr, "short", longStr]));

        using var ms = new MemoryStream();
        AvroWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = AvroReader.Read(ms);

        df2.RowCount.Should().Be(3);
        df2.GetStringColumn("Data")[0].Should().Be(longStr);
        df2.GetStringColumn("Data")[1].Should().Be("short");
        df2.GetStringColumn("Data")[2].Should().Be(longStr);
    }
}

public class OrcRoundTripEdgeCaseTests
{
    [Fact]
    public void Orc_Empty_DataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>()),
            new StringColumn("B", Array.Empty<string?>()));

        using var ms = new MemoryStream();
        OrcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = OrcReader.Read(ms);

        df2.RowCount.Should().Be(0);
        df2.ColumnCount.Should().Be(2);
    }

    [Fact]
    public void Orc_Single_Row()
    {
        var df = new DataFrame(
            new Column<int>("Id", [42]),
            new StringColumn("Name", ["test"]));

        using var ms = new MemoryStream();
        OrcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = OrcReader.Read(ms);

        df2.RowCount.Should().Be(1);
        df2.GetColumn<int>("Id")[0].Should().Be(42);
        df2.GetStringColumn("Name")[0].Should().Be("test");
    }

    [Fact]
    public void Orc_All_Null_Column()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Text", [null, null, null]));

        using var ms = new MemoryStream();
        OrcWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var df2 = OrcReader.Read(ms);

        df2.RowCount.Should().Be(3);
        for (int i = 0; i < 3; i++)
            df2["Text"].IsNull(i).Should().BeTrue();
    }
}

public class ResiliencePipelineEdgeCaseTests
{
    [Fact]
    public async Task ZeroMaxRetries_TransientFailure_Throws_Immediately()
    {
        var options = new CloudStorageOptions { MaxRetries = 0 };
        var pipeline = new ResiliencePipeline(options);
        int callCount = 0;

        var act = () => pipeline.ExecuteAsync<int>(async ct =>
        {
            callCount++;
            throw new IOException("transient");
        });

        await act.Should().ThrowAsync<IOException>();
        callCount.Should().Be(1, "with 0 max retries, should call exactly once");
    }

    [Fact]
    public async Task Immediate_Success_Returns_Value()
    {
        var pipeline = new ResiliencePipeline();

        var result = await pipeline.ExecuteAsync<int>(async ct => 42);

        result.Should().Be(42);
    }

    [Fact]
    public async Task NonTransient_Exception_Is_Not_Retried()
    {
        var options = new CloudStorageOptions { MaxRetries = 3 };
        var pipeline = new ResiliencePipeline(options);
        int callCount = 0;

        var act = () => pipeline.ExecuteAsync<int>(async ct =>
        {
            callCount++;
            throw new ArgumentException("not transient");
        });

        await act.Should().ThrowAsync<ArgumentException>();
        callCount.Should().Be(1, "non-transient exceptions should not be retried");
    }
}
