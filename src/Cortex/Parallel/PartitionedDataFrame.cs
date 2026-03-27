using Cortex.Column;
using Cortex.Concat;
using Cortex.GroupBy;

namespace Cortex.ParallelOps;

/// <summary>
/// A DataFrame split into partitions for parallel execution.
/// Operations (Filter, GroupBy, Apply) execute on each partition independently
/// across threads, then merge results.
///
/// Usage:
///   var result = df.Partition(8)          // 8 row-range partitions
///       .ParallelGroupBy("Category").Sum();
///   // or:
///   var result = df.HashPartition("Key", 8)  // 8 hash partitions on "Key"
///       .ParallelFilter(row => (double)row["Value"]! > 500)
///       .Collect();
/// </summary>
public class PartitionedDataFrame
{
    private readonly DataFrame[] _partitions;

    /// <summary>Number of partitions.</summary>
    public int PartitionCount => _partitions.Length;

    /// <summary>Total row count across all partitions.</summary>
    public int RowCount => _partitions.Sum(p => p.RowCount);

    internal PartitionedDataFrame(DataFrame[] partitions) => _partitions = partitions;

    /// <summary>Access a specific partition.</summary>
    public DataFrame GetPartition(int index) => _partitions[index];

    /// <summary>
    /// Execute a function on each partition in parallel, then concatenate results.
    /// </summary>
    public DataFrame ParallelMap(Func<DataFrame, DataFrame> func)
    {
        var results = new DataFrame[_partitions.Length];
        Parallel.For(0, _partitions.Length, i =>
        {
            results[i] = func(_partitions[i]);
        });

        // Filter out empty partitions
        var nonEmpty = results.Where(r => r.RowCount > 0).ToArray();
        if (nonEmpty.Length == 0)
            return results.Length > 0 ? new DataFrame(results[0].ColumnNames.Select(n => results[0][n].Slice(0, 0)).ToArray()) : new DataFrame();
        if (nonEmpty.Length == 1)
            return nonEmpty[0];
        return ConcatExtensions.Concat(nonEmpty);
    }

    /// <summary>
    /// Filter each partition in parallel using a row predicate.
    /// </summary>
    public PartitionedDataFrame ParallelFilter(Func<DataFrameRow, bool> predicate)
    {
        var results = new DataFrame[_partitions.Length];
        Parallel.For(0, _partitions.Length, i =>
        {
            results[i] = _partitions[i].Filter(predicate);
        });
        return new PartitionedDataFrame(results);
    }

    /// <summary>
    /// Filter using a column predicate in parallel.
    /// </summary>
    public PartitionedDataFrame ParallelWhere(string column, Func<object?, bool> predicate)
    {
        var results = new DataFrame[_partitions.Length];
        Parallel.For(0, _partitions.Length, i =>
        {
            results[i] = _partitions[i].Where(column, predicate);
        });
        return new PartitionedDataFrame(results);
    }

    /// <summary>
    /// GroupBy + aggregate across partitions in parallel.
    /// Each partition computes its local GroupBy, then results are merged.
    /// For commutative aggregates (Sum, Count, Min, Max), this is exact.
    /// For Mean, it computes per-partition means then averages (approximate for uneven partitions).
    /// </summary>
    public DataFrame ParallelGroupBy(string[] keyColumns, AggFunc aggFunc)
    {
        var localResults = new DataFrame[_partitions.Length];
        Parallel.For(0, _partitions.Length, i =>
        {
            var grouped = _partitions[i].GroupBy(keyColumns);
            localResults[i] = aggFunc switch
            {
                AggFunc.Sum => grouped.Sum(),
                AggFunc.Mean => grouped.Mean(),
                AggFunc.Min => grouped.Min(),
                AggFunc.Max => grouped.Max(),
                AggFunc.Count => grouped.Count(),
                AggFunc.Median => grouped.Median(),
                AggFunc.Std => grouped.Std(),
                AggFunc.Var => grouped.Var(),
                AggFunc.First => grouped.First(),
                AggFunc.Last => grouped.Last(),
                _ => throw new NotSupportedException($"Unknown aggregate: {aggFunc}")
            };
        });

        // Merge: for Sum/Count, re-aggregate; for Min/Max, take global min/max
        var merged = ConcatExtensions.Concat(localResults.Where(r => r.RowCount > 0).ToArray());

        // For commutative aggregates, re-aggregate the merged result
        if (aggFunc is AggFunc.Sum or AggFunc.Count or AggFunc.Min or AggFunc.Max)
        {
            var reGrouped = merged.GroupBy(keyColumns);
            return aggFunc switch
            {
                AggFunc.Sum => reGrouped.Sum(),
                AggFunc.Count => reGrouped.Sum(), // sum of counts = total count
                AggFunc.Min => reGrouped.Min(),
                AggFunc.Max => reGrouped.Max(),
                _ => merged
            };
        }

        // For non-commutative (Mean, Std, etc.), just return merged partials
        // (this is approximate — exact would require tracking counts per group)
        return merged;
    }

    /// <summary>Convenience: single key column GroupBy.</summary>
    public PartitionedGroupBy ParallelGroupBy(params string[] keyColumns)
        => new(this, keyColumns);

    /// <summary>Collect all partitions back into a single DataFrame.</summary>
    public DataFrame Collect()
    {
        var nonEmpty = _partitions.Where(p => p.RowCount > 0).ToArray();
        if (nonEmpty.Length == 0)
            return _partitions.Length > 0
                ? new DataFrame(_partitions[0].ColumnNames.Select(n => _partitions[0][n].Slice(0, 0)).ToArray())
                : new DataFrame();
        if (nonEmpty.Length == 1)
            return nonEmpty[0];
        return ConcatExtensions.Concat(nonEmpty);
    }

    /// <summary>Apply a function to each partition in parallel, keep partitioned.</summary>
    public PartitionedDataFrame Map(Func<DataFrame, DataFrame> func)
    {
        var results = new DataFrame[_partitions.Length];
        Parallel.For(0, _partitions.Length, i =>
        {
            results[i] = func(_partitions[i]);
        });
        return new PartitionedDataFrame(results);
    }
}

/// <summary>Partitioned GroupBy result for fluent aggregation.</summary>
public class PartitionedGroupBy
{
    private readonly PartitionedDataFrame _pdf;
    private readonly string[] _keyColumns;

    internal PartitionedGroupBy(PartitionedDataFrame pdf, string[] keyColumns)
    {
        _pdf = pdf;
        _keyColumns = keyColumns;
    }

    public DataFrame Sum() => _pdf.ParallelGroupBy(_keyColumns, AggFunc.Sum);
    public DataFrame Mean() => _pdf.ParallelGroupBy(_keyColumns, AggFunc.Mean);
    public DataFrame Min() => _pdf.ParallelGroupBy(_keyColumns, AggFunc.Min);
    public DataFrame Max() => _pdf.ParallelGroupBy(_keyColumns, AggFunc.Max);
    public DataFrame Count() => _pdf.ParallelGroupBy(_keyColumns, AggFunc.Count);
}

/// <summary>Extension methods for creating partitioned DataFrames.</summary>
public static class PartitionExtensions
{
    /// <summary>
    /// Split a DataFrame into N row-range partitions (round-robin by row index).
    /// Auto-tunes to CPU core count if numPartitions is not specified.
    /// </summary>
    public static PartitionedDataFrame Partition(this DataFrame df, int? numPartitions = null)
    {
        int n = numPartitions ?? Environment.ProcessorCount;
        if (n < 1) n = 1;
        if (df.RowCount == 0) return new PartitionedDataFrame([df]);

        int partSize = (df.RowCount + n - 1) / n;
        var partitions = new DataFrame[n];
        for (int p = 0; p < n; p++)
        {
            int start = p * partSize;
            int end = Math.Min(start + partSize, df.RowCount);
            int len = end - start;
            if (len <= 0)
            {
                // Create empty partition with same schema
                partitions[p] = new DataFrame(df.ColumnNames.Select(name => df[name].Slice(0, 0)).ToArray());
                continue;
            }
            var cols = df.ColumnNames.Select(name => df[name].Slice(start, len)).ToArray();
            partitions[p] = new DataFrame(cols);
        }
        return new PartitionedDataFrame(partitions);
    }

    /// <summary>
    /// Split a DataFrame into N hash partitions based on a key column.
    /// Rows with the same key always land in the same partition (locality for GroupBy).
    /// </summary>
    public static PartitionedDataFrame HashPartition(this DataFrame df, string keyColumn, int? numPartitions = null)
    {
        int n = numPartitions ?? Environment.ProcessorCount;
        if (n < 1) n = 1;
        if (df.RowCount == 0) return new PartitionedDataFrame([df]);

        var keyCol = df[keyColumn];
        var buckets = new List<int>[n];
        for (int i = 0; i < n; i++) buckets[i] = new List<int>();

        for (int r = 0; r < df.RowCount; r++)
        {
            var key = keyCol.GetObject(r);
            int hash = key is not null ? (key.GetHashCode() & 0x7FFFFFFF) % n : 0;
            buckets[hash].Add(r);
        }

        var partitions = new DataFrame[n];
        for (int p = 0; p < n; p++)
        {
            if (buckets[p].Count == 0)
            {
                partitions[p] = new DataFrame(df.ColumnNames.Select(name => df[name].Slice(0, 0)).ToArray());
                continue;
            }
            int[] indices = buckets[p].ToArray();
            var cols = df.ColumnNames.Select(name => df[name].TakeRows(indices)).ToArray();
            partitions[p] = new DataFrame(cols);
        }

        return new PartitionedDataFrame(partitions);
    }
}
