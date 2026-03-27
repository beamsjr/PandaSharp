using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Storage;

namespace Cortex.Tests.Unit.Storage;

public class SpilledDataFrameTests : IDisposable
{
    private readonly string _tempDir;
    private readonly List<SpilledDataFrame> _spills = new();

    public SpilledDataFrameTests()
    {
        _tempDir = Path.Combine(Path.GetTempPath(), $"pandasharp_spill_test_{Guid.NewGuid():N}");
        Directory.CreateDirectory(_tempDir);
    }

    public void Dispose()
    {
        foreach (var s in _spills) s.Dispose();
        if (Directory.Exists(_tempDir)) Directory.Delete(_tempDir, true);
    }

    private SpilledDataFrame Track(SpilledDataFrame s) { _spills.Add(s); return s; }
    private string TempPath(string name) => Path.Combine(_tempDir, name);

    private DataFrame CreateTestDF(int n = 1000)
    {
        var ids = new int[n];
        var values = new double[n];
        var labels = new string?[n];
        for (int i = 0; i < n; i++)
        {
            ids[i] = i;
            values[i] = i * 1.5;
            labels[i] = $"label_{i % 10}";
        }
        return new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values),
            new StringColumn("Label", labels)
        );
    }

    // ===== Spill and Materialize =====

    [Fact]
    public void Spill_CreatesFile()
    {
        var df = CreateTestDF();
        var path = TempPath("test.arrow");
        var spilled = Track(SpilledDataFrame.Spill(df, path));

        File.Exists(path).Should().BeTrue();
        spilled.RowCount.Should().Be(1000);
        spilled.ColumnCount.Should().Be(3);
        spilled.ColumnNames.Should().Equal(["Id", "Value", "Label"]);
    }

    [Fact]
    public void Materialize_ReturnsFullDataFrame()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("full.arrow")));
        var loaded = spilled.Materialize();

        loaded.RowCount.Should().Be(100);
        loaded.ColumnCount.Should().Be(3);
        loaded.GetColumn<int>("Id")[50].Should().Be(50);
        loaded.GetColumn<double>("Value")[99].Should().Be(99 * 1.5);
    }

    [Fact]
    public void Materialize_SelectedColumns()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("select.arrow")));
        var partial = spilled.Materialize("Id", "Value");

        partial.ColumnCount.Should().Be(2);
        partial.ColumnNames.Should().Equal(["Id", "Value"]);
        partial.RowCount.Should().Be(100);
    }

    // ===== Lazy Column Loading =====

    [Fact]
    public void ColumnAccess_LazyLoads()
    {
        var df = CreateTestDF();
        var spilled = Track(df.Spill(TempPath("lazy.arrow")));

        spilled.CachedColumnCount.Should().Be(0);

        var col = spilled["Value"];
        col.Should().NotBeNull();
        col.Length.Should().Be(1000);

        spilled.CachedColumnCount.Should().Be(1);
        spilled.IsLoaded("Value").Should().BeTrue();
        spilled.IsLoaded("Id").Should().BeFalse();
    }

    [Fact]
    public void Evict_RemovesFromCache()
    {
        var df = CreateTestDF();
        var spilled = Track(df.Spill(TempPath("evict.arrow")));

        _ = spilled["Value"];
        spilled.CachedColumnCount.Should().Be(1);

        spilled.Evict("Value");
        spilled.CachedColumnCount.Should().Be(0);

        // Can reload after eviction
        var col = spilled["Value"];
        col.Length.Should().Be(1000);
    }

    [Fact]
    public void EvictAll_ClearsCache()
    {
        var df = CreateTestDF();
        var spilled = Track(df.Spill(TempPath("evictall.arrow")));

        _ = spilled["Id"];
        _ = spilled["Value"];
        _ = spilled["Label"];
        spilled.CachedColumnCount.Should().Be(3);

        spilled.EvictAll();
        spilled.CachedColumnCount.Should().Be(0);
    }

    // ===== Open existing file =====

    [Fact]
    public void Open_ExistingFile()
    {
        var df = CreateTestDF(50);
        var path = TempPath("existing.arrow");
        Cortex.IO.ArrowIpcWriter.Write(df, path);

        var spilled = Track(SpilledDataFrame.Open(path));
        spilled.RowCount.Should().Be(50);
        spilled.ColumnCount.Should().Be(3);
        ((Column<int>)spilled["Id"])[0].Should().Be(0);
    }

    // ===== Head / Tail =====

    [Fact]
    public void Head_ReturnsFirstN()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("head.arrow")));
        var head = spilled.Head(5);

        head.RowCount.Should().Be(5);
        head.GetColumn<int>("Id")[4].Should().Be(4);
    }

    [Fact]
    public void Tail_ReturnsLastN()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("tail.arrow")));
        var tail = spilled.Tail(3);

        tail.RowCount.Should().Be(3);
        tail.GetColumn<int>("Id")[2].Should().Be(99);
    }

    // ===== Select to new spill file =====

    [Fact]
    public void Select_CreatesNewSpillWithSubset()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("orig.arrow")));
        var selected = Track(spilled.Select(TempPath("subset.arrow"), "Id", "Value"));

        selected.ColumnCount.Should().Be(2);
        selected.RowCount.Should().Be(100);
        selected.ColumnNames.Should().Equal(["Id", "Value"]);

        // Original file size should be larger
        new FileInfo(spilled.FilePath).Length.Should().BeGreaterThan(
            new FileInfo(selected.FilePath).Length);
    }

    // ===== Filter to new spill file =====

    [Fact]
    public void Filter_CreatesFilteredSpill()
    {
        var df = CreateTestDF(100);
        var spilled = Track(df.Spill(TempPath("filter_orig.arrow")));
        var filtered = Track(spilled.Filter(TempPath("filter_result.arrow"),
            d => {
                var col = d.GetColumn<double>("Value");
                var mask = new bool[d.RowCount];
                for (int i = 0; i < d.RowCount; i++)
                    mask[i] = col[i]!.Value > 100;
                return mask;
            }));

        filtered.RowCount.Should().BeLessThan(100);
        filtered.RowCount.Should().BeGreaterThan(0);
    }

    // ===== Dispose =====

    [Fact]
    public void Dispose_DeletesOwnedFile()
    {
        var df = CreateTestDF(10);
        var path = TempPath("dispose.arrow");
        var spilled = SpilledDataFrame.Spill(df, path);

        File.Exists(path).Should().BeTrue();
        spilled.Dispose();
        File.Exists(path).Should().BeFalse();
    }

    [Fact]
    public void Dispose_DoesNotDeleteOpenedFile()
    {
        var df = CreateTestDF(10);
        var path = TempPath("keep.arrow");
        Cortex.IO.ArrowIpcWriter.Write(df, path);

        var spilled = SpilledDataFrame.Open(path);
        spilled.Dispose();
        File.Exists(path).Should().BeTrue(); // not deleted — we don't own it
    }

    [Fact]
    public void AfterDispose_AccessThrows()
    {
        var df = CreateTestDF(10);
        var spilled = SpilledDataFrame.Spill(df, TempPath("dead.arrow"));
        spilled.Dispose();

        var act = () => spilled["Id"];
        act.Should().Throw<ObjectDisposedException>();
    }

    // ===== Extension method =====

    [Fact]
    public void SpillExtension_AutoPath()
    {
        var df = CreateTestDF(10);
        var spilled = Track(df.Spill());

        File.Exists(spilled.FilePath).Should().BeTrue();
        spilled.FilePath.Should().Contain("pandasharp_spill_");
        spilled.Materialize().RowCount.Should().Be(10);
    }

    // ===== Data integrity =====

    [Fact]
    public void RoundTrip_PreservesAllTypes()
    {
        var df = new DataFrame(
            new Column<int>("Int", [1, 2, 3]),
            new Column<double>("Dbl", [1.5, 2.5, 3.5]),
            new Column<long>("Long", [100L, 200L, 300L]),
            new StringColumn("Str", ["a", "b", "c"])
        );

        var spilled = Track(df.Spill(TempPath("types.arrow")));
        var loaded = spilled.Materialize();

        loaded.GetColumn<int>("Int")[2].Should().Be(3);
        loaded.GetColumn<double>("Dbl")[1].Should().Be(2.5);
        loaded.GetColumn<long>("Long")[0].Should().Be(100L);
        loaded.GetStringColumn("Str")[2].Should().Be("c");
    }

    [Fact]
    public void LargeDataFrame_SpillAndReload()
    {
        int n = 50_000;
        var ids = new int[n];
        var vals = new double[n];
        for (int i = 0; i < n; i++) { ids[i] = i; vals[i] = i * 0.1; }

        var df = new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", vals)
        );

        var spilled = Track(df.Spill(TempPath("large.arrow")));
        spilled.RowCount.Should().Be(50_000);

        var col = spilled["Value"];
        ((Column<double>)col)[49_999].Should().Be(49_999 * 0.1);
    }
}
