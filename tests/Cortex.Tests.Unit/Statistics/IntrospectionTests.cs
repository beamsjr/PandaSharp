using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Statistics;

namespace Cortex.Tests.Unit.Statistics;

public class IntrospectionTests
{
    // ── Explain ──

    [Fact]
    public void Explain_MixedTypeDataFrame_ReturnsCorrectMetadata()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["IntCol"] = new[] { 1, 2, 3, 4, 5 },
            ["DoubleCol"] = new[] { 1.1, 2.2, 3.3, 4.4, 5.5 },
            ["StringCol"] = new string?[] { "a", "b", null, "d", "e" },
            ["BoolCol"] = new[] { true, false, true, false, true },
            ["LongCol"] = new[] { 10L, 20L, 30L, 40L, 50L },
        });

        var plan = df.Explain();

        plan.RowCount.Should().Be(5);
        plan.ColumnCount.Should().Be(5);
        plan.Columns.Should().HaveCount(5);

        // Check types
        plan.Columns[0].DataType.Should().Be(typeof(int));
        plan.Columns[1].DataType.Should().Be(typeof(double));
        plan.Columns[2].DataType.Should().Be(typeof(string));
        plan.Columns[3].DataType.Should().Be(typeof(bool));
        plan.Columns[4].DataType.Should().Be(typeof(long));

        // Check names
        plan.Columns[0].Name.Should().Be("IntCol");
        plan.Columns[2].Name.Should().Be("StringCol");

        // Null counts: StringCol has 1 null, rest have 0
        plan.Columns[0].NullCount.Should().Be(0);
        plan.Columns[2].NullCount.Should().Be(1);
        plan.Columns[2].NullPercent.Should().BeApproximately(20.0, 0.01);

        // Memory estimates should be positive
        foreach (var col in plan.Columns)
            col.MemoryBytes.Should().BeGreaterThan(0);

        plan.EstimatedMemoryBytes.Should().BeGreaterThan(0);
        plan.EstimatedMemoryBytes.Should().Be(plan.Columns.Sum(c => c.MemoryBytes));
    }

    [Fact]
    public void Explain_IntColumn_MemoryMatchesExpected()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Vals"] = new[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 },
        });

        var plan = df.Explain();

        // 10 ints * 4 bytes = 40 bytes (no nulls, no bitmask)
        plan.Columns[0].MemoryBytes.Should().Be(40);
    }

    [Fact]
    public void Explain_EmptyDataFrame_HandlesGracefully()
    {
        var df = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        var plan = df.Explain();

        plan.RowCount.Should().Be(0);
        plan.ColumnCount.Should().Be(1);
        plan.Columns[0].NullPercent.Should().Be(0);
        plan.Columns[0].MemoryBytes.Should().Be(0);
    }

    [Fact]
    public void Explain_ToString_ProducesPrettyPrint()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["X"] = new[] { 1, 2, 3 },
        });

        var plan = df.Explain();
        var text = plan.ToString();

        text.Should().Contain("ExecutionPlan");
        text.Should().Contain("3 rows");
        text.Should().Contain("X");
    }

    // ── Benchmark ──

    [Fact]
    public void Benchmark_TrivialSelect_ReturnsAllStatFields()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3, 4, 5 },
            ["B"] = new[] { 10.0, 20.0, 30.0, 40.0, 50.0 },
        });

        var result = df.Benchmark(d => d.Select("A"), iterations: 10, warmup: 2);

        result.Iterations.Should().Be(10);
        result.MinMs.Should().BeGreaterThanOrEqualTo(0);
        result.MedianMs.Should().BeGreaterThanOrEqualTo(result.MinMs);
        result.P95Ms.Should().BeGreaterThanOrEqualTo(result.MedianMs);
        result.P99Ms.Should().BeGreaterThanOrEqualTo(result.P95Ms);
        result.MeanMs.Should().BeGreaterThanOrEqualTo(result.MinMs);
        result.Gen0Collections.Should().BeGreaterThanOrEqualTo(0);
        result.Gen1Collections.Should().BeGreaterThanOrEqualTo(0);
        result.Gen2Collections.Should().BeGreaterThanOrEqualTo(0);
    }

    [Fact]
    public void Benchmark_ToDataFrame_HasCorrectColumns()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1, 2, 3 },
        });

        var result = df.Benchmark(d => d.Head(1), iterations: 5, warmup: 1);
        var resultDf = result.ToDataFrame();

        resultDf.RowCount.Should().Be(1);
        resultDf.ColumnNames.Should().Contain("iterations");
        resultDf.ColumnNames.Should().Contain("min_ms");
        resultDf.ColumnNames.Should().Contain("median_ms");
        resultDf.ColumnNames.Should().Contain("p95_ms");
        resultDf.ColumnNames.Should().Contain("p99_ms");
        resultDf.ColumnNames.Should().Contain("mean_ms");
        resultDf.ColumnNames.Should().Contain("gen0_collections");
        resultDf.ColumnNames.Should().Contain("gen1_collections");
        resultDf.ColumnNames.Should().Contain("gen2_collections");
    }

    [Fact]
    public void Benchmark_SingleIteration_Works()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new[] { 1 },
        });

        var result = df.Benchmark(d => { }, iterations: 1, warmup: 0);

        result.Iterations.Should().Be(1);
        result.MinMs.Should().Be(result.MedianMs);
    }

    // ── ProfileToDataFrame ──

    [Fact]
    public void ProfileToDataFrame_MixedTypes_HasAllStatRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["IntCol"] = new[] { 10, 20, 30, 40 },
            ["DoubleCol"] = new[] { 1.5, 2.5, 3.5, 4.5 },
            ["StringCol"] = new string?[] { "alpha", "beta", "alpha", null },
            ["BoolCol"] = new[] { true, false, true, true },
        });

        var profile = df.ProfileToDataFrame();

        // Should have stat column + 4 data columns = 5 columns total
        profile.ColumnCount.Should().Be(5);
        profile.ColumnNames[0].Should().Be("stat");
        profile.ColumnNames.Should().Contain("IntCol");
        profile.ColumnNames.Should().Contain("DoubleCol");
        profile.ColumnNames.Should().Contain("StringCol");
        profile.ColumnNames.Should().Contain("BoolCol");

        // 14 stat rows
        profile.RowCount.Should().Be(14);

        // Verify the stat names in the first column
        var statCol = profile.GetStringColumn("stat");
        statCol[0].Should().Be("count");
        statCol[1].Should().Be("mean");
        statCol[2].Should().Be("std");
        statCol[3].Should().Be("min");
        statCol[4].Should().Be("25%");
        statCol[5].Should().Be("50%");
        statCol[6].Should().Be("75%");
        statCol[7].Should().Be("max");
        statCol[8].Should().Be("null%");
        statCol[9].Should().Be("unique");
        statCol[10].Should().Be("top");
        statCol[11].Should().Be("freq");
        statCol[12].Should().Be("dtype");
        statCol[13].Should().Be("memory_bytes");
    }

    [Fact]
    public void ProfileToDataFrame_NumericColumn_HasCorrectStats()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Vals"] = new[] { 1, 2, 3, 4, 5 },
        });

        var profile = df.ProfileToDataFrame();
        var valsCol = profile.GetStringColumn("Vals");

        // count = 5
        valsCol[0].Should().Be("5");
        // mean = 3
        double.Parse(valsCol[1]!).Should().BeApproximately(3.0, 0.01);
        // dtype
        valsCol[12].Should().Be("int32");
        // null%
        double.Parse(valsCol[8]!).Should().Be(0.0);
        // unique = 5
        valsCol[9].Should().Be("5");
        // memory_bytes > 0
        long.Parse(valsCol[13]!).Should().BeGreaterThan(0);
    }

    [Fact]
    public void ProfileToDataFrame_StringColumn_HasTopAndFreq()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Names"] = new string?[] { "Alice", "Bob", "Alice", "Alice", "Bob" },
        });

        var profile = df.ProfileToDataFrame();
        var namesCol = profile.GetStringColumn("Names");

        // count = 5
        namesCol[0].Should().Be("5");
        // unique = 2
        namesCol[9].Should().Be("2");
        // top = Alice (most frequent)
        namesCol[10].Should().Be("Alice");
        // freq = 3
        namesCol[11].Should().Be("3");
        // dtype = string
        namesCol[12].Should().Be("string");
    }

    [Fact]
    public void ProfileToDataFrame_WithNulls_ReportsNullPercent()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("NullableInts", new int?[] { 1, null, 3, null, 5 })
        );

        var profile = df.ProfileToDataFrame();
        var col = profile.GetStringColumn("NullableInts");

        // count = 3 (non-null)
        col[0].Should().Be("3");
        // null% = 40
        double.Parse(col[8]!).Should().BeApproximately(40.0, 0.01);
    }
}
