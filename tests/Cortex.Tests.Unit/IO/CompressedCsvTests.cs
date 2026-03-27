using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class CompressedCsvTests : IDisposable
{
    private readonly string _tempDir;

    public CompressedCsvTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_gzip_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
    }

    [Fact]
    public void GzipCsv_WriteAndRead_RoundTrip()
    {
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new Column<double>("Value", [1.5, 2.5, 3.5]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );

        var path = Path.Combine(_tempDir, "test.csv.gz");
        CsvWriter.Write(df, path);
        File.Exists(path).Should().BeTrue();

        var loaded = CsvReader.Read(path);
        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("Name")[2].Should().Be("Charlie");
    }

    [Fact]
    public void GzipCsv_CompressedSmallerThanPlain()
    {
        // Generate a large-ish CSV
        int n = 1000;
        var ids = new int[n];
        var values = new double[n];
        for (int i = 0; i < n; i++) { ids[i] = i; values[i] = i * 1.1; }
        var df = new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values)
        );

        var plainPath = Path.Combine(_tempDir, "plain.csv");
        var gzipPath = Path.Combine(_tempDir, "compressed.csv.gz");

        CsvWriter.Write(df, plainPath);
        CsvWriter.Write(df, gzipPath);

        var plainSize = new FileInfo(plainPath).Length;
        var gzipSize = new FileInfo(gzipPath).Length;

        gzipSize.Should().BeLessThan(plainSize);
    }

    [Fact]
    public void GzipCsv_PreservesAllTypes()
    {
        var df = new DataFrame(
            new Column<int>("Int", [10, 20]),
            new Column<double>("Dbl", [1.1, 2.2]),
            new StringColumn("Str", ["x", "y"])
        );

        var path = Path.Combine(_tempDir, "types.csv.gz");
        CsvWriter.Write(df, path);
        var loaded = CsvReader.Read(path);

        loaded.ColumnCount.Should().Be(3);
        loaded["Int"].DataType.Should().Be(typeof(int));
        loaded["Dbl"].DataType.Should().Be(typeof(double));
    }

    [Fact]
    public void PlainCsv_StillWorks()
    {
        var df = new DataFrame(new Column<int>("x", [1, 2, 3]));
        var path = Path.Combine(_tempDir, "plain.csv");
        CsvWriter.Write(df, path);
        var loaded = CsvReader.Read(path);
        loaded.RowCount.Should().Be(3);
    }
}
