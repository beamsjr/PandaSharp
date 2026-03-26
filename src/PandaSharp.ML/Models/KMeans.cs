using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// K-Means clustering using Lloyd's algorithm with k-means++ initialization.
/// Runs multiple random initializations and keeps the result with lowest inertia.
/// </summary>
public class KMeans : IModel
{
    /// <inheritdoc />
    public string Name => "KMeans";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>Number of clusters.</summary>
    public int NClusters { get; }

    /// <summary>Maximum iterations per run.</summary>
    public int MaxIter { get; }

    /// <summary>Convergence tolerance (centroid movement).</summary>
    public double Tolerance { get; }

    /// <summary>Random seed for reproducibility. Null for random.</summary>
    public int? Seed { get; }

    /// <summary>Number of initializations to try.</summary>
    public int NInit { get; }

    /// <summary>Cluster centroids of shape (n_clusters, n_features). Available after fitting.</summary>
    public Tensor<double>? ClusterCenters { get; private set; }

    /// <summary>Cluster label assigned to each training sample. Available after fitting.</summary>
    public int[]? Labels { get; private set; }

    /// <summary>Sum of squared distances of samples to their closest cluster center.</summary>
    public double Inertia { get; private set; } = double.PositiveInfinity;

    /// <summary>Number of iterations in the best run.</summary>
    public int NIter { get; private set; }

    /// <summary>Create a KMeans model.</summary>
    /// <param name="nClusters">Number of clusters.</param>
    /// <param name="maxIter">Maximum iterations (default 300).</param>
    /// <param name="tolerance">Convergence tolerance (default 1e-4).</param>
    /// <param name="seed">Random seed (default null).</param>
    /// <param name="nInit">Number of initializations (default 10).</param>
    public KMeans(int nClusters, int maxIter = 300, double tolerance = 1e-4,
        int? seed = null, int nInit = 10)
    {
        if (nClusters < 1) throw new ArgumentOutOfRangeException(nameof(nClusters), "Must be >= 1.");
        if (maxIter < 1) throw new ArgumentOutOfRangeException(nameof(maxIter), "Must be >= 1.");
        if (nInit < 1) throw new ArgumentOutOfRangeException(nameof(nInit), "Must be >= 1.");

        NClusters = nClusters;
        MaxIter = maxIter;
        Tolerance = tolerance;
        Seed = seed;
        NInit = nInit;
    }

    /// <summary>
    /// Fit the model. The target vector y is ignored (unsupervised) but accepted for IModel compatibility.
    /// Pass a dummy tensor for y if needed.
    /// </summary>
    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        return Fit(X);
    }

    /// <summary>Fit the model to feature matrix X.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public KMeans Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];

        if (NClusters > n)
            throw new ArgumentException($"Cannot create {NClusters} clusters from {n} samples.");

        var xData = X.ToArray();
        double bestInertia = double.PositiveInfinity;
        double[]? bestCentroids = null;
        int[]? bestLabels = null;
        int bestNIter = 0;

        var baseRng = Seed.HasValue ? new Random(Seed.Value) : new Random();

        for (int init = 0; init < NInit; init++)
        {
            int initSeed = baseRng.Next();
            var (centroids, labels, inertia, nIter) = RunSingleInit(xData, n, d, initSeed);

            if (inertia < bestInertia)
            {
                bestInertia = inertia;
                bestCentroids = centroids;
                bestLabels = labels;
                bestNIter = nIter;
            }
        }

        ClusterCenters = new Tensor<double>(bestCentroids!, NClusters, d);
        Labels = bestLabels;
        Inertia = bestInertia;
        NIter = bestNIter;
        IsFitted = true;
        return this;
    }

    /// <summary>Assign each sample to the nearest centroid.</summary>
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.ToArray();
        var centroids = ClusterCenters!.ToArray();
        var result = new double[n];

        Parallel.For(0, n, i =>
        {
            result[i] = FindNearestCentroid(xData, i * d, centroids, d);
        });

        return new Tensor<double>(result, n);
    }

    /// <summary>Returns the negative inertia (higher is better, for cross-validation compatibility).</summary>
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Shape[0] == 0) return 0.0;
        // Compute inertia on X
        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.ToArray();
        var centroids = ClusterCenters!.ToArray();
        double inertia = 0;

        for (int i = 0; i < n; i++)
        {
            int nearest = FindNearestCentroid(xData, i * d, centroids, d);
            inertia += SquaredDistance(xData, i * d, centroids, nearest * d, d);
        }

        return -inertia;
    }

    // -- Private helpers --

    private (double[] Centroids, int[] Labels, double Inertia, int NIter) RunSingleInit(
        double[] xData, int n, int d, int seed)
    {
        var rng = new Random(seed);
        var centroids = KMeansPlusPlusInit(xData, n, d, rng);
        var labels = new int[n];
        int nIter = 0;

        for (int iter = 0; iter < MaxIter; iter++)
        {
            nIter = iter + 1;

            // Assignment step (parallel)
            Parallel.For(0, n, i =>
            {
                labels[i] = FindNearestCentroid(centroids, xData, i * d, d);
            });

            // Update step
            var newCentroids = new double[NClusters * d];
            var counts = new int[NClusters];

            for (int i = 0; i < n; i++)
            {
                int c = labels[i];
                counts[c]++;
                int cOff = c * d;
                int xOff = i * d;
                for (int j = 0; j < d; j++)
                    newCentroids[cOff + j] += xData[xOff + j];
            }

            for (int c = 0; c < NClusters; c++)
            {
                int cOff = c * d;
                if (counts[c] > 0)
                {
                    for (int j = 0; j < d; j++)
                        newCentroids[cOff + j] /= counts[c];
                }
                else
                {
                    // Empty cluster: reinitialize to random sample
                    int randIdx = rng.Next(n) * d;
                    for (int j = 0; j < d; j++)
                        newCentroids[cOff + j] = xData[randIdx + j];
                }
            }

            // Check convergence
            double maxShift = 0;
            for (int c = 0; c < NClusters; c++)
            {
                double shift = 0;
                int cOff = c * d;
                for (int j = 0; j < d; j++)
                {
                    double diff = newCentroids[cOff + j] - centroids[cOff + j];
                    shift += diff * diff;
                }
                shift = Math.Sqrt(shift);
                if (shift > maxShift) maxShift = shift;
            }

            Array.Copy(newCentroids, centroids, centroids.Length);

            if (maxShift < Tolerance) break;
        }

        // Compute inertia
        double inertia = 0;
        for (int i = 0; i < n; i++)
        {
            int c = labels[i];
            inertia += SquaredDistanceFlat(xData, i * d, centroids, c * d, d);
        }

        return (centroids, labels, inertia, nIter);
    }

    /// <summary>K-means++ initialization: choose initial centroids spread out from each other.</summary>
    private double[] KMeansPlusPlusInit(double[] xData, int n, int d, Random rng)
    {
        var centroids = new double[NClusters * d];

        // Pick first centroid uniformly at random
        int first = rng.Next(n);
        for (int j = 0; j < d; j++)
            centroids[j] = xData[first * d + j];

        var minDist = new double[n];
        Array.Fill(minDist, double.MaxValue);

        for (int c = 1; c < NClusters; c++)
        {
            // Update min distances to nearest chosen centroid
            int prevOff = (c - 1) * d;
            double totalDist = 0;
            for (int i = 0; i < n; i++)
            {
                double dist = SquaredDistanceFlat(xData, i * d, centroids, prevOff, d);
                if (dist < minDist[i]) minDist[i] = dist;
                totalDist += minDist[i];
            }

            // Weighted random selection
            double target = rng.NextDouble() * totalDist;
            double cumulative = 0;
            int chosen = n - 1;
            for (int i = 0; i < n; i++)
            {
                cumulative += minDist[i];
                if (cumulative >= target) { chosen = i; break; }
            }

            int cOff = c * d;
            for (int j = 0; j < d; j++)
                centroids[cOff + j] = xData[chosen * d + j];
        }

        return centroids;
    }

    private static int FindNearestCentroid(double[] centroids, double[] xData, int xOff, int d)
    {
        int nClusters = centroids.Length / d;
        int best = 0;
        double bestDist = double.MaxValue;
        for (int c = 0; c < nClusters; c++)
        {
            double dist = SquaredDistanceFlat(xData, xOff, centroids, c * d, d);
            if (dist < bestDist) { bestDist = dist; best = c; }
        }
        return best;
    }

    private int FindNearestCentroid(double[] xData, int xOff,
        double[] centroids, int d)
    {
        int best = 0;
        double bestDist = double.MaxValue;
        for (int c = 0; c < NClusters; c++)
        {
            double dist = 0;
            int cOff = c * d;
            for (int j = 0; j < d; j++)
            {
                double diff = xData[xOff + j] - centroids[cOff + j];
                dist += diff * diff;
            }
            if (dist < bestDist) { bestDist = dist; best = c; }
        }
        return best;
    }

    private static double SquaredDistance(double[] a, int aOff,
        double[] b, int bOff, int len)
    {
        double sum = 0;
        for (int i = 0; i < len; i++)
        {
            double diff = a[aOff + i] - b[bOff + i];
            sum += diff * diff;
        }
        return sum;
    }

    private static double SquaredDistanceFlat(double[] a, int aOff,
        double[] b, int bOff, int len)
    {
        double sum = 0;
        for (int i = 0; i < len; i++)
        {
            double diff = a[aOff + i] - b[bOff + i];
            sum += diff * diff;
        }
        return sum;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
