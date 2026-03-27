using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class ExcelTests : IDisposable
{
    private readonly string _tempDir;
    public ExcelTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_excel_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }
    public void Dispose() { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true); }

    [Fact]
    public void Excel_RoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<int>("Age", [25, 30]),
            new Column<double>("Score", [95.5, 87.3])
        );

        var path = Path.Combine(_tempDir, "test.xlsx");
        ExcelIO.WriteExcel(df, path);
        File.Exists(path).Should().BeTrue();

        var loaded = ExcelIO.ReadExcel(path);
        loaded.RowCount.Should().Be(2);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
    }

    [Fact]
    public void Excel_MultiSheet_Write()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));

        var path = Path.Combine(_tempDir, "multi.xlsx");
        ExcelIO.WriteExcel(df, path, "Data");

        var sheets = ExcelIO.ListSheets(path);
        sheets.Should().Contain("Data");
    }

    [Fact]
    public void Excel_ReadSpecificSheet()
    {
        var df = new DataFrame(new Column<int>("Val", [10, 20]));
        var path = Path.Combine(_tempDir, "sheets.xlsx");
        ExcelIO.WriteExcel(df, path, "MySheet");

        var loaded = ExcelIO.ReadExcel(path, new ExcelReadOptions { SheetName = "MySheet" });
        loaded.RowCount.Should().Be(2);
    }
}
