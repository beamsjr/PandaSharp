using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.ParallelOps;

/// <summary>
/// Parallel execution engine for DataFrame operations.
/// Partitions data by row ranges and processes across threads.
/// Use for large DataFrames (100K+ rows) where the per-thread overhead is amortized.
/// </summary>
public static class ParallelDataFrame
{
    /// <summary>Default minimum rows per partition to avoid excessive thread overhead.</summary>
    private const int MinRowsPerPartition = 4096;

    /// <summary>
    /// Filter rows using a boolean mask, evaluated in parallel.
    /// </summary>
    public static DataFrame ParallelFilter(this DataFrame df, Func<DataFrameRow, bool> predicate,
        int? maxDegreeOfParallelism = null)
    {
        int rowCount = df.RowCount;
        if (rowCount == 0) return df;

        var mask = new bool[rowCount];
        var options = MakeOptions(maxDegreeOfParallelism);
        int partitionSize = GetPartitionSize(rowCount, options);

        Parallel.For(0, (rowCount + partitionSize - 1) / partitionSize, options, partIdx =>
        {
            int start = partIdx * partitionSize;
            int end = Math.Min(start + partitionSize, rowCount);
            for (int i = start; i < end; i++)
                mask[i] = predicate(df[i]);
        });

        return df.Filter(mask);
    }

    /// <summary>
    /// Filter using a column predicate, evaluated in parallel.
    /// Usage: df.ParallelWhere("Value", val => (double)val! > 500)
    /// </summary>
    public static DataFrame ParallelWhere(this DataFrame df, string column, Func<object?, bool> predicate,
        int? maxDegreeOfParallelism = null)
    {
        int rowCount = df.RowCount;
        if (rowCount == 0) return df;

        var col = df[column];
        var mask = new bool[rowCount];
        var options = MakeOptions(maxDegreeOfParallelism);
        int partitionSize = GetPartitionSize(rowCount, options);

        Parallel.For(0, (rowCount + partitionSize - 1) / partitionSize, options, partIdx =>
        {
            int start = partIdx * partitionSize;
            int end = Math.Min(start + partitionSize, rowCount);
            for (int i = start; i < end; i++)
                mask[i] = predicate(col.GetObject(i));
        });

        return df.Filter(mask);
    }

    /// <summary>
    /// Apply a function to each row in parallel, producing a new numeric column.
    /// </summary>
    public static DataFrame ParallelApply<T>(this DataFrame df, Func<DataFrameRow, T> func,
        string columnName, int? maxDegreeOfParallelism = null) where T : struct
    {
        int rowCount = df.RowCount;
        var values = new T?[rowCount];
        var options = MakeOptions(maxDegreeOfParallelism);
        int partitionSize = GetPartitionSize(rowCount, options);

        Parallel.For(0, (rowCount + partitionSize - 1) / partitionSize, options, partIdx =>
        {
            int start = partIdx * partitionSize;
            int end = Math.Min(start + partitionSize, rowCount);
            for (int i = start; i < end; i++)
            {
                try { values[i] = func(df[i]); }
                catch (InvalidOperationException) { values[i] = null; }
                catch (NullReferenceException) { values[i] = null; }
            }
        });

        return df.AddColumn(Column<T>.FromNullable(columnName, values));
    }

    /// <summary>
    /// Apply a function to each row in parallel, producing a new string column.
    /// </summary>
    public static DataFrame ParallelApply(this DataFrame df, Func<DataFrameRow, string?> func,
        string columnName, int? maxDegreeOfParallelism = null)
    {
        int rowCount = df.RowCount;
        var values = new string?[rowCount];
        var options = MakeOptions(maxDegreeOfParallelism);
        int partitionSize = GetPartitionSize(rowCount, options);

        Parallel.For(0, (rowCount + partitionSize - 1) / partitionSize, options, partIdx =>
        {
            int start = partIdx * partitionSize;
            int end = Math.Min(start + partitionSize, rowCount);
            for (int i = start; i < end; i++)
            {
                try { values[i] = func(df[i]); }
                catch (InvalidOperationException) { values[i] = null; }
                catch (NullReferenceException) { values[i] = null; }
            }
        });

        return df.AddColumn(StringColumn.CreateOwned(columnName, values));
    }

    /// <summary>
    /// Element-wise column addition in parallel across partitions.
    /// </summary>
    public static Column<T> ParallelAdd<T>(this Column<T> left, Column<T> right,
        int? maxDegreeOfParallelism = null)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");

        int len = left.Length;
        var leftArr = left.Buffer.Span.ToArray();
        var rightArr = right.Buffer.Span.ToArray();

        if (left.NullCount == 0 && right.NullCount == 0)
        {
            var resultArr = new T[len];
            var options = MakeOptions(maxDegreeOfParallelism);
            int partSize = GetPartitionSize(len, options);

            Parallel.For(0, (len + partSize - 1) / partSize, options, partIdx =>
            {
                int start = partIdx * partSize;
                int end = Math.Min(start + partSize, len);
                for (int i = start; i < end; i++)
                    resultArr[i] = leftArr[i] + rightArr[i];
            });

            return new Column<T>(left.Name, resultArr);
        }
        var vals = new T?[len];
        var optionsN = MakeOptions(maxDegreeOfParallelism);
        int partSizeN = GetPartitionSize(len, optionsN);

        Parallel.For(0, (len + partSizeN - 1) / partSizeN, optionsN, partIdx =>
        {
            int start = partIdx * partSizeN;
            int end = Math.Min(start + partSizeN, len);
            for (int i = start; i < end; i++)
                vals[i] = left.Nulls.IsNull(i) || right.Nulls.IsNull(i) ? null : leftArr[i] + rightArr[i];
        });

        return Column<T>.FromNullable(left.Name, vals);
    }

    /// <summary>
    /// Element-wise column multiplication in parallel.
    /// </summary>
    public static Column<T> ParallelMultiply<T>(this Column<T> left, Column<T> right,
        int? maxDegreeOfParallelism = null)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");

        int len = left.Length;
        var leftArr = left.Buffer.Span.ToArray();
        var rightArr = right.Buffer.Span.ToArray();

        if (left.NullCount == 0 && right.NullCount == 0)
        {
            var resultArr = new T[len];
            var options = MakeOptions(maxDegreeOfParallelism);
            int partSize = GetPartitionSize(len, options);

            Parallel.For(0, (len + partSize - 1) / partSize, options, partIdx =>
            {
                int start = partIdx * partSize;
                int end = Math.Min(start + partSize, len);
                for (int i = start; i < end; i++)
                    resultArr[i] = leftArr[i] * rightArr[i];
            });

            return new Column<T>(left.Name, resultArr);
        }

        var vals = new T?[len];
        var optionsN = MakeOptions(maxDegreeOfParallelism);
        int partSizeN = GetPartitionSize(len, optionsN);

        Parallel.For(0, (len + partSizeN - 1) / partSizeN, optionsN, partIdx =>
        {
            int start = partIdx * partSizeN;
            int end = Math.Min(start + partSizeN, len);
            for (int i = start; i < end; i++)
                vals[i] = left.Nulls.IsNull(i) || right.Nulls.IsNull(i) ? null : leftArr[i] * rightArr[i];
        });

        return Column<T>.FromNullable(left.Name, vals);
    }

    /// <summary>
    /// Scalar multiply in parallel.
    /// </summary>
    public static Column<T> ParallelMultiply<T>(this Column<T> col, T scalar,
        int? maxDegreeOfParallelism = null)
        where T : struct, INumber<T>
    {
        int len = col.Length;
        var srcArr = col.Buffer.Span.ToArray();

        if (col.NullCount == 0)
        {
            var resultArr = new T[len];
            var options = MakeOptions(maxDegreeOfParallelism);
            int partSize = GetPartitionSize(len, options);

            Parallel.For(0, (len + partSize - 1) / partSize, options, partIdx =>
            {
                int start = partIdx * partSize;
                int end = Math.Min(start + partSize, len);
                for (int i = start; i < end; i++)
                    resultArr[i] = srcArr[i] * scalar;
            });

            return new Column<T>(col.Name, resultArr);
        }

        var vals = new T?[len];
        var optionsN = MakeOptions(maxDegreeOfParallelism);
        int partSizeN = GetPartitionSize(len, optionsN);

        Parallel.For(0, (len + partSizeN - 1) / partSizeN, optionsN, partIdx =>
        {
            int start = partIdx * partSizeN;
            int end = Math.Min(start + partSizeN, len);
            for (int i = start; i < end; i++)
                vals[i] = col.Nulls.IsNull(i) ? null : srcArr[i] * scalar;
        });

        return Column<T>.FromNullable(col.Name, vals);
    }

    /// <summary>
    /// Parallel sum of a double column using partitioned accumulators.
    /// </summary>
    public static double ParallelSum(this Column<double> col, int? maxDegreeOfParallelism = null)
    {
        if (col.Length == 0) return 0;
        var arr = col.Buffer.Span.ToArray();
        var options = MakeOptions(maxDegreeOfParallelism);
        int numPartitions = Math.Max(1, Math.Min(
            Environment.ProcessorCount,
            maxDegreeOfParallelism ?? Environment.ProcessorCount));
        int partSize = Math.Max(MinRowsPerPartition, (col.Length + numPartitions - 1) / numPartitions);
        int actualPartitions = (col.Length + partSize - 1) / partSize;

        var partialSums = new double[actualPartitions];

        Parallel.For(0, actualPartitions, options, partIdx =>
        {
            int start = partIdx * partSize;
            int end = Math.Min(start + partSize, col.Length);
            double sum = 0;
            if (col.NullCount == 0)
            {
                for (int i = start; i < end; i++)
                    sum += arr[i];
            }
            else
            {
                for (int i = start; i < end; i++)
                    if (!col.Nulls.IsNull(i)) sum += arr[i];
            }
            partialSums[partIdx] = sum;
        });

        double total = 0;
        for (int i = 0; i < actualPartitions; i++)
            total += partialSums[i];
        return total;
    }

    private static ParallelOptions MakeOptions(int? maxDegreeOfParallelism) => new()
    {
        MaxDegreeOfParallelism = maxDegreeOfParallelism ?? Environment.ProcessorCount
    };

    private static int GetPartitionSize(int totalItems, ParallelOptions options)
    {
        int threads = options.MaxDegreeOfParallelism > 0
            ? options.MaxDegreeOfParallelism
            : Environment.ProcessorCount;
        return Math.Max(MinRowsPerPartition, (totalItems + threads - 1) / threads);
    }
}
