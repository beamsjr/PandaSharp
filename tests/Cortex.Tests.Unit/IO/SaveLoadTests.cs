using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class SaveLoadTests : IDisposable
{
    private readonly string _tempDir;

    public SaveLoadTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_saveload_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
    }

    private DataFrame CreateTestDF() => new(
        new Column<int>("Id", [1, 2, 3]),
        new Column<double>("Value", [1.5, 2.5, 3.5]),
        new StringColumn("Name", ["Alice", "Bob", "Charlie"])
    );

    // ===== CSV =====

    [Fact]
    public void SaveLoad_Csv()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.csv");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void SaveLoad_CsvGz()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.csv.gz");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
    }

    [Fact]
    public void SaveLoad_Tsv()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.tsv");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
    }

    // ===== JSON =====

    [Fact]
    public void SaveLoad_Json()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.json");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
    }

    [Fact]
    public void SaveLoad_JsonLines()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.jsonl");
        df.Save(path);

        File.ReadAllLines(path).Length.Should().BeGreaterThanOrEqualTo(3);

        var loaded = DataFrameIO.Load(path);
        loaded.RowCount.Should().Be(3);
    }

    // ===== Parquet =====

    [Fact]
    public void SaveLoad_Parquet()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.parquet");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
        loaded.GetColumn<int>("Id")[2].Should().Be(3);
    }

    // ===== Arrow IPC =====

    [Fact]
    public void SaveLoad_Arrow()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.arrow");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
    }

    // ===== Excel =====

    [Fact]
    public void SaveLoad_Excel()
    {
        var df = CreateTestDF();
        var path = Path.Combine(_tempDir, "test.xlsx");
        df.Save(path);
        var loaded = DataFrameIO.Load(path);

        loaded.RowCount.Should().Be(3);
    }

    // ===== Error handling =====

    [Fact]
    public void Load_FileNotFound_Throws()
    {
        var act = () => DataFrameIO.Load("/nonexistent/path.csv");
        act.Should().Throw<FileNotFoundException>();
    }

    [Fact]
    public void Save_UnknownExtension_Throws()
    {
        var df = CreateTestDF();
        var act = () => df.Save(Path.Combine(_tempDir, "test.xyz"));
        act.Should().Throw<NotSupportedException>();
    }

    [Fact]
    public void Load_UnknownExtension_Throws()
    {
        File.WriteAllText(Path.Combine(_tempDir, "test.xyz"), "data");
        var act = () => DataFrameIO.Load(Path.Combine(_tempDir, "test.xyz"));
        act.Should().Throw<NotSupportedException>();
    }
}
