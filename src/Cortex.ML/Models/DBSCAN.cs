using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Density-Based Spatial Clustering of Applications with Noise (DBSCAN).
/// Identifies clusters of varying shape based on density, marking low-density points as noise (-1).
/// Uses a sorted-index optimization for efficient eps-neighborhood queries.
/// </summary>
public class DBSCAN
{
    private const int NoiseLabel = -1;
    private const int Unclassified = -2;

    /// <summary>Maximum distance between two samples to be considered neighbors.</summary>
    public double Eps { get; }

    /// <summary>Minimum number of samples in an eps-neighborhood to form a core point.</summary>
    public int MinSamples { get; }

    /// <summary>Distance metric to use.</summary>
    public DistanceMetric Metric { get; }

    /// <summary>Cluster label for each training sample (-1 = noise). Available after fitting.</summary>
    public int[]? Labels { get; private set; }

    /// <summary>Indices of core samples. Available after fitting.</summary>
    public int[]? CoreSampleIndices { get; private set; }

    /// <summary>Number of clusters found (excluding noise).</summary>
    public int NClusters { get; private set; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Create a DBSCAN model.</summary>
    /// <param name="eps">Neighborhood radius (default 0.5).</param>
    /// <param name="minSamples">Min samples to form a core point (default 5).</param>
    /// <param name="metric">Distance metric (default Euclidean).</param>
    public DBSCAN(double eps = 0.5, int minSamples = 5, DistanceMetric metric = DistanceMetric.Euclidean)
    {
        if (eps <= 0) throw new ArgumentOutOfRangeException(nameof(eps), "eps must be > 0.");
        if (minSamples < 1) throw new ArgumentOutOfRangeException(nameof(minSamples), "minSamples must be >= 1.");

        Eps = eps;
        MinSamples = minSamples;
        Metric = metric;
    }

    /// <summary>Fit the DBSCAN model to feature matrix X.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public DBSCAN Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];

        if (n == 0)
        {
            Labels = Array.Empty<int>();
            NClusters = 0;
            return this;
        }

        var xData = X.ToArray();

        // Pre-compute sorted order along first dimension for neighborhood pruning
        var sortedByDim0 = new int[n];
        var dim0Values = new double[n];
        for (int i = 0; i < n; i++)
        {
            sortedByDim0[i] = i;
            dim0Values[i] = xData[i * d]; // first feature
        }
        Array.Sort(dim0Values, sortedByDim0);

        // Pre-compute neighborhoods using sorted order for pruning
        var neighborhoods = new List<int>[n];
        Parallel.For(0, n, i =>
        {
            neighborhoods[i] = FindNeighbors(xData, n, d, i, dim0Values, sortedByDim0);
        });

        // Identify core points
        var isCore = new bool[n];
        var coreIndices = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (neighborhoods[i].Count >= MinSamples)
            {
                isCore[i] = true;
                coreIndices.Add(i);
            }
        }

        // Expand clusters
        var labels = new int[n];
        Array.Fill(labels, Unclassified);
        int currentCluster = 0;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] != Unclassified) continue;
            if (!isCore[i]) continue;

            // Start new cluster
            labels[i] = currentCluster;
            var queue = new Queue<int>(neighborhoods[i]);

            while (queue.Count > 0)
            {
                int q = queue.Dequeue();

                if (labels[q] == NoiseLabel)
                {
                    // Border point: absorb into cluster
                    labels[q] = currentCluster;
                    continue;
                }

                if (labels[q] != Unclassified) continue;

                labels[q] = currentCluster;

                if (isCore[q])
                {
                    foreach (int neighbor in neighborhoods[q])
                    {
                        if (labels[neighbor] == Unclassified || labels[neighbor] == NoiseLabel)
                            queue.Enqueue(neighbor);
                    }
                }
            }

            currentCluster++;
        }

        // Mark remaining unclassified as noise
        for (int i = 0; i < n; i++)
        {
            if (labels[i] == Unclassified)
                labels[i] = NoiseLabel;
        }

        Labels = labels;
        CoreSampleIndices = coreIndices.ToArray();
        NClusters = currentCluster;
        IsFitted = true;
        return this;
    }

    // -- Private helpers --

    private List<int> FindNeighbors(double[] xData, int n, int d,
        int pointIdx, double[] dim0Values, int[] sortedByDim0)
    {
        var neighbors = new List<int>();
        double pointDim0 = xData[pointIdx * d];

        // Binary search for range [pointDim0 - eps, pointDim0 + eps] in sorted first-dimension
        int lo = LowerBound(dim0Values, pointDim0 - Eps);
        int hi = UpperBound(dim0Values, pointDim0 + Eps);

        for (int si = lo; si < hi; si++)
        {
            int j = sortedByDim0[si];
            double dist = ComputeDistance(xData, pointIdx * d, xData, j * d, d);
            if (dist <= Eps)
                neighbors.Add(j);
        }

        return neighbors;
    }

    private double ComputeDistance(double[] a, int aOff,
        double[] b, int bOff, int len)
    {
        double sum = 0;
        switch (Metric)
        {
            case DistanceMetric.Euclidean:
                for (int i = 0; i < len; i++)
                {
                    double diff = a[aOff + i] - b[bOff + i];
                    sum += diff * diff;
                }
                return Math.Sqrt(sum);

            case DistanceMetric.Manhattan:
                for (int i = 0; i < len; i++)
                    sum += Math.Abs(a[aOff + i] - b[bOff + i]);
                return sum;

            case DistanceMetric.Minkowski:
                // Default to p=2 for DBSCAN
                for (int i = 0; i < len; i++)
                {
                    double diff = a[aOff + i] - b[bOff + i];
                    sum += diff * diff;
                }
                return Math.Sqrt(sum);

            default:
                throw new InvalidOperationException($"Unknown distance metric: {Metric}");
        }
    }

    private static int LowerBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (sorted[mid] < value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }

    private static int UpperBound(double[] sorted, double value)
    {
        int lo = 0, hi = sorted.Length;
        while (lo < hi)
        {
            int mid = lo + (hi - lo) / 2;
            if (sorted[mid] <= value) lo = mid + 1;
            else hi = mid;
        }
        return lo;
    }
}
