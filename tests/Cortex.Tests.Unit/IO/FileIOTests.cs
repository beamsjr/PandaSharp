using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class FileIOTests : IDisposable
{
    private readonly string _tempDir;

    public FileIOTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir))
            Directory.Delete(_tempDir, recursive: true);
    }

    [Fact]
    public void CsvFile_RoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30]),
            new Column<double>("Score", [95.5, 87.3])
        );

        var path = Path.Combine(_tempDir, "test.csv");
        df.ToCsv(path);
        File.Exists(path).Should().BeTrue();

        var loaded = DataFrameIO.ReadCsv(path);
        loaded.RowCount.Should().Be(2);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
        loaded.GetColumn<int>("Age")[1].Should().Be(30);
    }

    [Fact]
    public void JsonFile_RoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("City", ["NYC", "LA"]),
            new Column<int>("Pop", [8_000_000, 4_000_000])
        );

        var path = Path.Combine(_tempDir, "test.json");
        df.ToJson(path);
        File.Exists(path).Should().BeTrue();

        var loaded = DataFrameIO.ReadJson(path);
        loaded.RowCount.Should().Be(2);
        loaded.GetStringColumn("City")[1].Should().Be("LA");
    }

    [Fact]
    public void ArrowFile_RoundTrip()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new StringColumn("Label", ["a", "b", "c"])
        );

        var path = Path.Combine(_tempDir, "test.arrow");
        df.ToArrow(path);
        File.Exists(path).Should().BeTrue();

        var loaded = DataFrameIO.ReadArrow(path);
        loaded.RowCount.Should().Be(3);
        loaded.GetColumn<int>("Id")[2].Should().Be(3);
    }

    [Fact]
    public void CsvFile_WithNulls_RoundTrip()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("A", [1, null, 3]),
            new StringColumn("B", ["x", null, "z"])
        );

        var path = Path.Combine(_tempDir, "nulls.csv");
        df.ToCsv(path);
        var loaded = DataFrameIO.ReadCsv(path);

        loaded.RowCount.Should().Be(3);
        loaded["A"].IsNull(1).Should().BeTrue();
        loaded["B"].IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void CsvFile_TSV()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Value", [1, 2])
        );

        var path = Path.Combine(_tempDir, "test.tsv");
        df.ToCsv(path, new CsvWriteOptions { Delimiter = '\t' });

        var loaded = DataFrameIO.ReadCsv(path, new CsvReadOptions { Delimiter = '\t' });
        loaded.RowCount.Should().Be(2);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
    }
}
