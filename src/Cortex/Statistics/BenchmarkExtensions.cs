using System.Buffers;
using System.Diagnostics;
using Cortex.Column;

namespace Cortex.Statistics;

/// <summary>
/// Result of benchmarking an action against a DataFrame.
/// All timing fields are in milliseconds.
/// </summary>
public record BenchmarkResult(
    int Iterations,
    double MinMs,
    double MedianMs,
    double P95Ms,
    double P99Ms,
    double MeanMs,
    long Gen0Collections,
    long Gen1Collections,
    long Gen2Collections)
{
    /// <summary>
    /// Convert the benchmark result into a single-row DataFrame for easy analysis.
    /// </summary>
    public DataFrame ToDataFrame()
    {
        var columns = new IColumn[]
        {
            new Column<int>("iterations", new[] { Iterations }),
            new Column<double>("min_ms", new[] { MinMs }),
            new Column<double>("median_ms", new[] { MedianMs }),
            new Column<double>("p95_ms", new[] { P95Ms }),
            new Column<double>("p99_ms", new[] { P99Ms }),
            new Column<double>("mean_ms", new[] { MeanMs }),
            new Column<long>("gen0_collections", new[] { Gen0Collections }),
            new Column<long>("gen1_collections", new[] { Gen1Collections }),
            new Column<long>("gen2_collections", new[] { Gen2Collections }),
        };
        return new DataFrame(columns);
    }
}

public static class BenchmarkExtensions
{
    /// <summary>
    /// Benchmark an action against this DataFrame.
    /// Uses Stopwatch.GetTimestamp() for zero-allocation high-resolution timing.
    /// Tracks GC collections via GC.CollectionCount().
    /// Uses ArrayPool for timing storage to avoid allocation during measurement.
    /// </summary>
    /// <param name="df">The DataFrame to pass to the action.</param>
    /// <param name="action">The action to benchmark.</param>
    /// <param name="iterations">Number of timed iterations (default 100).</param>
    /// <param name="warmup">Number of warmup iterations (default 3).</param>
    public static BenchmarkResult Benchmark(
        this DataFrame df,
        Action<DataFrame> action,
        int iterations = 100,
        int warmup = 3)
    {
        if (iterations < 1)
            throw new ArgumentOutOfRangeException(nameof(iterations), "Must be >= 1.");

        // Rent from ArrayPool to avoid allocation in the hot path
        double[] timings = ArrayPool<double>.Shared.Rent(iterations);

        try
        {
            double tickFreq = (double)Stopwatch.Frequency;

            // Warmup: run the action to JIT and stabilize
            for (int i = 0; i < warmup; i++)
                action(df);

            // Force a full GC before measurement to get clean baseline
            GC.Collect();
            GC.WaitForPendingFinalizers();
            GC.Collect();

            long gen0Before = GC.CollectionCount(0);
            long gen1Before = GC.CollectionCount(1);
            long gen2Before = GC.CollectionCount(2);

            // Timed iterations using Stopwatch.GetTimestamp() — zero allocation
            for (int i = 0; i < iterations; i++)
            {
                long start = Stopwatch.GetTimestamp();
                action(df);
                long end = Stopwatch.GetTimestamp();
                timings[i] = (end - start) * 1000.0 / tickFreq;
            }

            long gen0After = GC.CollectionCount(0);
            long gen1After = GC.CollectionCount(1);
            long gen2After = GC.CollectionCount(2);

            // Compute statistics from the rented array (only first `iterations` elements)
            var span = timings.AsSpan(0, iterations);

            // Sort in-place for percentile calculation
            span.Sort();

            double min = span[0];
            double median = Percentile(span, 0.50);
            double p95 = Percentile(span, 0.95);
            double p99 = Percentile(span, 0.99);

            double sum = 0;
            for (int i = 0; i < iterations; i++)
                sum += span[i];
            double mean = sum / iterations;

            return new BenchmarkResult(
                Iterations: iterations,
                MinMs: min,
                MedianMs: median,
                P95Ms: p95,
                P99Ms: p99,
                MeanMs: mean,
                Gen0Collections: gen0After - gen0Before,
                Gen1Collections: gen1After - gen1Before,
                Gen2Collections: gen2After - gen2Before);
        }
        finally
        {
            ArrayPool<double>.Shared.Return(timings);
        }
    }

    private static double Percentile(Span<double> sorted, double p)
    {
        int n = sorted.Length;
        if (n == 1) return sorted[0];
        double rank = p * (n - 1);
        int lower = (int)rank;
        int upper = lower + 1;
        if (upper >= n) return sorted[lower];
        double frac = rank - lower;
        return sorted[lower] + frac * (sorted[upper] - sorted[lower]);
    }
}
