using FluentAssertions;
using Cortex.Column;
using Cortex.IO;

namespace Cortex.Tests.Unit.IO;

public class ParquetTests : IDisposable
{
    private readonly string _tempDir;
    public ParquetTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_parquet_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }
    public void Dispose() { if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true); }

    [Fact]
    public void Parquet_RoundTrip_IntAndString()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<int>("Age", [25, 30, 35])
        );

        var path = Path.Combine(_tempDir, "test.parquet");
        df.ToParquet(path);
        File.Exists(path).Should().BeTrue();

        var loaded = DataFrameIO.ReadParquet(path);
        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("Name")[0].Should().Be("Alice");
        loaded.GetColumn<int>("Age")[2].Should().Be(35);
    }

    [Fact]
    public void Parquet_RoundTrip_Double()
    {
        var df = new DataFrame(
            new Column<double>("Value", [1.5, 2.7, 3.14])
        );

        var path = Path.Combine(_tempDir, "doubles.parquet");
        df.ToParquet(path);
        var loaded = DataFrameIO.ReadParquet(path);

        loaded.GetColumn<double>("Value")[0].Should().Be(1.5);
        loaded.GetColumn<double>("Value")[2].Should().Be(3.14);
    }

    [Fact]
    public void Parquet_ColumnPruning()
    {
        var df = new DataFrame(
            new StringColumn("A", ["x", "y"]),
            new Column<int>("B", [1, 2]),
            new Column<double>("C", [10.0, 20.0])
        );

        var path = Path.Combine(_tempDir, "prune.parquet");
        df.ToParquet(path);

        var loaded = DataFrameIO.ReadParquet(path, new ParquetReadOptions { Columns = ["A", "C"] });
        loaded.ColumnCount.Should().Be(2);
        loaded.ColumnNames.Should().Contain("A");
        loaded.ColumnNames.Should().Contain("C");
        loaded.ColumnNames.Should().NotContain("B");
    }

    [Fact]
    public void Parquet_LargeFile()
    {
        int n = 10_000;
        var ids = new int[n];
        var vals = new double[n];
        for (int i = 0; i < n; i++) { ids[i] = i; vals[i] = i * 1.5; }

        var df = new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", vals)
        );

        var path = Path.Combine(_tempDir, "large.parquet");
        df.ToParquet(path);
        var loaded = DataFrameIO.ReadParquet(path);

        loaded.RowCount.Should().Be(n);
        loaded.GetColumn<int>("Id")[9999].Should().Be(9999);
    }

    // ===== Partitioned Parquet =====

    [Fact]
    public void WritePartitioned_CreatesHiveDirectories()
    {
        var df = new DataFrame(
            new StringColumn("region", ["US", "US", "EU", "EU"]),
            new Column<int>("sales", [100, 200, 300, 400])
        );

        var partDir = Path.Combine(_tempDir, "partitioned");
        ParquetIO.WritePartitioned(df, partDir, "region");

        Directory.Exists(Path.Combine(partDir, "region=US")).Should().BeTrue();
        Directory.Exists(Path.Combine(partDir, "region=EU")).Should().BeTrue();
        File.Exists(Path.Combine(partDir, "region=US", "part-0.parquet")).Should().BeTrue();
        File.Exists(Path.Combine(partDir, "region=EU", "part-0.parquet")).Should().BeTrue();
    }

    [Fact]
    public void ReadPartitioned_ReadsAllPartitions()
    {
        var df = new DataFrame(
            new StringColumn("region", ["US", "US", "EU", "EU", "EU"]),
            new Column<int>("sales", [100, 200, 300, 400, 500])
        );

        var partDir = Path.Combine(_tempDir, "read_part");
        ParquetIO.WritePartitioned(df, partDir, "region");

        var loaded = ParquetIO.ReadPartitioned(partDir);

        loaded.RowCount.Should().Be(5);
        loaded.ColumnNames.Should().Contain("sales");
        loaded.ColumnNames.Should().Contain("region");
    }

    [Fact]
    public void ReadPartitioned_PartitionColumnsHaveCorrectValues()
    {
        var df = new DataFrame(
            new Column<int>("year", [2023, 2023, 2024, 2024]),
            new Column<double>("revenue", [1.1, 2.2, 3.3, 4.4])
        );

        var partDir = Path.Combine(_tempDir, "year_part");
        ParquetIO.WritePartitioned(df, partDir, "year");

        var loaded = ParquetIO.ReadPartitioned(partDir);

        loaded.RowCount.Should().Be(4);
        loaded.ColumnNames.Should().Contain("year");
        loaded.ColumnNames.Should().Contain("revenue");

        // All year values should be either 2023 or 2024
        for (int i = 0; i < loaded.RowCount; i++)
        {
            var year = loaded.GetColumn<int>("year")[i];
            year.Should().BeOneOf(2023, 2024);
        }
    }

    [Fact]
    public void ReadPartitioned_MultiLevelPartition()
    {
        var df = new DataFrame(
            new StringColumn("region", ["US", "US", "EU", "EU"]),
            new Column<int>("year", [2023, 2024, 2023, 2024]),
            new Column<double>("sales", [1.0, 2.0, 3.0, 4.0])
        );

        var partDir = Path.Combine(_tempDir, "multi_part");
        ParquetIO.WritePartitioned(df, partDir, "region", "year");

        // Should create region=US/year=2023/, region=US/year=2024/, etc.
        Directory.Exists(Path.Combine(partDir, "region=US", "year=2023")).Should().BeTrue();
        Directory.Exists(Path.Combine(partDir, "region=EU", "year=2024")).Should().BeTrue();

        var loaded = ParquetIO.ReadPartitioned(partDir);

        loaded.RowCount.Should().Be(4);
        loaded.ColumnNames.Should().Contain("sales");
        loaded.ColumnNames.Should().Contain("region");
        loaded.ColumnNames.Should().Contain("year");
    }

    [Fact]
    public void ReadPartitioned_WithColumnPruning()
    {
        var df = new DataFrame(
            new StringColumn("region", ["US", "EU"]),
            new Column<int>("id", [1, 2]),
            new Column<double>("sales", [100.0, 200.0])
        );

        var partDir = Path.Combine(_tempDir, "prune_part");
        ParquetIO.WritePartitioned(df, partDir, "region");

        // Only read the 'sales' data column
        var loaded = ParquetIO.ReadPartitioned(partDir, new ParquetReadOptions { Columns = ["sales"] });

        // Should have sales + partition column 'region'
        loaded.ColumnNames.Should().Contain("sales");
        loaded.ColumnNames.Should().Contain("region");
        loaded.ColumnNames.Should().NotContain("id"); // pruned
    }

    [Fact]
    public void ReadPartitioned_RoundTrip_PreservesData()
    {
        var rng = new Random(42);
        int n = 1000;
        var regions = new string?[n];
        var values = new double[n];
        var cats = new[] { "A", "B", "C" };
        for (int i = 0; i < n; i++)
        {
            regions[i] = cats[rng.Next(cats.Length)];
            values[i] = rng.NextDouble() * 100;
        }

        var df = new DataFrame(
            new StringColumn("cat", regions),
            new Column<double>("val", values)
        );

        var partDir = Path.Combine(_tempDir, "roundtrip_part");
        ParquetIO.WritePartitioned(df, partDir, "cat");
        var loaded = ParquetIO.ReadPartitioned(partDir);

        loaded.RowCount.Should().Be(n);

        // Sum of values should match (order may differ due to partitioning)
        var originalSum = values.Sum();
        double loadedSum = 0;
        for (int i = 0; i < loaded.RowCount; i++)
            loadedSum += loaded.GetColumn<double>("val")[i]!.Value;
        loadedSum.Should().BeApproximately(originalSum, 0.01);
    }

    [Fact]
    public void ReadPartitioned_EmptyDirectory_Throws()
    {
        var emptyDir = Path.Combine(_tempDir, "empty");
        Directory.CreateDirectory(emptyDir);

        var act = () => ParquetIO.ReadPartitioned(emptyDir);
        act.Should().Throw<InvalidDataException>();
    }

    [Fact]
    public void ReadPartitioned_NonexistentDirectory_Throws()
    {
        var act = () => ParquetIO.ReadPartitioned("/nonexistent/path/xyz");
        act.Should().Throw<DirectoryNotFoundException>();
    }
}
