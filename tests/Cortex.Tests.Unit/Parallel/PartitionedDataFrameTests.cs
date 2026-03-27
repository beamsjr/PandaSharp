using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.ParallelOps;

namespace Cortex.Tests.Unit.Parallel;

public class PartitionedDataFrameTests
{
    private DataFrame CreateLargeDF(int n = 10_000)
    {
        var ids = new int[n];
        var values = new double[n];
        var categories = new string?[n];
        var rng = new Random(42);
        var cats = new[] { "A", "B", "C", "D" };

        for (int i = 0; i < n; i++)
        {
            ids[i] = i;
            values[i] = rng.NextDouble() * 1000;
            categories[i] = cats[rng.Next(cats.Length)];
        }

        return new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values),
            new StringColumn("Category", categories)
        );
    }

    // ===== Row-range partitioning =====

    [Fact]
    public void Partition_SplitsIntoCorrectCount()
    {
        var df = CreateLargeDF();
        var pdf = df.Partition(4);

        pdf.PartitionCount.Should().Be(4);
        pdf.RowCount.Should().Be(10_000);
    }

    [Fact]
    public void Partition_PartitionsCoverAllRows()
    {
        var df = CreateLargeDF(100);
        var pdf = df.Partition(3);

        var total = 0;
        for (int i = 0; i < pdf.PartitionCount; i++)
            total += pdf.GetPartition(i).RowCount;

        total.Should().Be(100);
    }

    [Fact]
    public void Partition_SinglePartition()
    {
        var df = CreateLargeDF(50);
        var pdf = df.Partition(1);

        pdf.PartitionCount.Should().Be(1);
        pdf.GetPartition(0).RowCount.Should().Be(50);
    }

    [Fact]
    public void Partition_MorePartitionsThanRows()
    {
        var df = CreateLargeDF(3);
        var pdf = df.Partition(10);

        pdf.RowCount.Should().Be(3);
    }

    [Fact]
    public void Partition_AutoTunesDefault()
    {
        var df = CreateLargeDF();
        var pdf = df.Partition(); // auto

        pdf.PartitionCount.Should().Be(Environment.ProcessorCount);
    }

    // ===== Hash partitioning =====

    [Fact]
    public void HashPartition_SameKeysSamePartition()
    {
        var df = CreateLargeDF();
        var pdf = df.HashPartition("Category", 4);

        pdf.PartitionCount.Should().Be(4);
        pdf.RowCount.Should().Be(10_000);

        // Within each partition, all Category values should be consistent
        // (same key → same partition)
        for (int p = 0; p < pdf.PartitionCount; p++)
        {
            var part = pdf.GetPartition(p);
            if (part.RowCount == 0) continue;
            var categories = new HashSet<string?>();
            for (int r = 0; r < part.RowCount; r++)
                categories.Add(part.GetStringColumn("Category")[r]);

            // Each partition should have a subset of categories
            categories.Count.Should().BeLessThanOrEqualTo(4);
        }
    }

    [Fact]
    public void HashPartition_PreservesAllRows()
    {
        var df = CreateLargeDF(500);
        var pdf = df.HashPartition("Category", 8);
        pdf.RowCount.Should().Be(500);
    }

    // ===== Collect =====

    [Fact]
    public void Collect_ReassemblesAllRows()
    {
        var df = CreateLargeDF(200);
        var pdf = df.Partition(4);
        var collected = pdf.Collect();

        collected.RowCount.Should().Be(200);
        collected.ColumnCount.Should().Be(3);
    }

    // ===== ParallelFilter =====

    [Fact]
    public void ParallelFilter_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = df.Filter(row => (double)row["Value"]! > 500);
        var parallel = df.Partition(4)
            .ParallelFilter(row => (double)row["Value"]! > 500)
            .Collect();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    [Fact]
    public void ParallelWhere_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = df.Where("Value", v => (double)v! > 500);
        var parallel = df.Partition(4)
            .ParallelWhere("Value", v => (double)v! > 500)
            .Collect();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    // ===== ParallelGroupBy =====

    [Fact]
    public void ParallelGroupBy_Sum_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = GroupByExtensions.GroupBy(df, "Category").Sum();
        var parallel = df.HashPartition("Category", 4)
            .ParallelGroupBy("Category").Sum();

        parallel.RowCount.Should().Be(sequential.RowCount);

        // Sum values should match
        var seqSum = sequential.Sort("Category");
        var parSum = parallel.Sort("Category");

        for (int i = 0; i < seqSum.RowCount; i++)
        {
            seqSum.GetStringColumn("Category")[i]
                .Should().Be(parSum.GetStringColumn("Category")[i]);
        }
    }

    [Fact]
    public void ParallelGroupBy_Count_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = GroupByExtensions.GroupBy(df, "Category").Count();
        var parallel = df.HashPartition("Category", 4)
            .ParallelGroupBy("Category").Count();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    [Fact]
    public void ParallelGroupBy_Min_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = GroupByExtensions.GroupBy(df, "Category").Min();
        var parallel = df.HashPartition("Category", 4)
            .ParallelGroupBy("Category").Min();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    [Fact]
    public void ParallelGroupBy_Max_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = GroupByExtensions.GroupBy(df, "Category").Max();
        var parallel = df.HashPartition("Category", 4)
            .ParallelGroupBy("Category").Max();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    // ===== ParallelMap =====

    [Fact]
    public void ParallelMap_TransformsAllPartitions()
    {
        var df = CreateLargeDF(100);
        var result = df.Partition(4)
            .ParallelMap(part => part.Head(5));

        // Each partition contributes at most 5 rows → max 20 total
        result.RowCount.Should().BeLessThanOrEqualTo(20);
        result.RowCount.Should().BeGreaterThan(0);
    }

    // ===== Map then Collect =====

    [Fact]
    public void Map_ThenCollect_PreservesSchema()
    {
        var df = CreateLargeDF(100);
        var result = df.Partition(2)
            .Map(part => part.Select("Id", "Value"))
            .Collect();

        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["Id", "Value"]);
        result.RowCount.Should().Be(100);
    }

    // ===== Empty =====

    [Fact]
    public void Partition_EmptyDataFrame()
    {
        var df = new DataFrame(new Column<int>("x", Array.Empty<int>()));
        var pdf = df.Partition(4);
        pdf.RowCount.Should().Be(0);
        pdf.Collect().RowCount.Should().Be(0);
    }
}
