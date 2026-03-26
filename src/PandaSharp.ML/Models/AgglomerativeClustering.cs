using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Linkage criterion for agglomerative clustering.
/// </summary>
public enum Linkage
{
    /// <summary>Distance between the closest pair of points in two clusters.</summary>
    Single,

    /// <summary>Distance between the farthest pair of points in two clusters.</summary>
    Complete,

    /// <summary>Average distance between all pairs of points in two clusters.</summary>
    Average,

    /// <summary>Minimizes total within-cluster variance (only valid with Euclidean distance).</summary>
    Ward
}

/// <summary>
/// Agglomerative (bottom-up) hierarchical clustering.
/// Starts with each sample as its own cluster and iteratively merges the two closest
/// clusters until the desired number of clusters is reached.
/// </summary>
public class AgglomerativeClustering
{
    /// <summary>Target number of clusters.</summary>
    public int NClusters { get; }

    /// <summary>Linkage criterion used for merging.</summary>
    public Linkage Linkage { get; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>Cluster label assigned to each training sample. Available after fitting.</summary>
    public int[]? Labels { get; private set; }

    /// <summary>
    /// Merge tree of shape (n_samples - 1, 2). Row i contains the two cluster indices
    /// merged at step i. Cluster indices &gt;= n_samples refer to merged clusters
    /// formed at step (index - n_samples).
    /// </summary>
    public int[,]? Children { get; private set; }

    /// <summary>Create an agglomerative clustering model.</summary>
    /// <param name="nClusters">Number of clusters to find (default 2).</param>
    /// <param name="linkage">Linkage criterion (default Ward).</param>
    public AgglomerativeClustering(int nClusters = 2, Linkage linkage = Linkage.Ward)
    {
        if (nClusters < 1)
            throw new ArgumentOutOfRangeException(nameof(nClusters), "nClusters must be >= 1.");

        NClusters = nClusters;
        Linkage = linkage;
    }

    /// <summary>Fit the model to feature matrix X, computing the cluster hierarchy and labels.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public AgglomerativeClustering Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];

        if (NClusters > n)
            throw new ArgumentException($"nClusters ({NClusters}) cannot exceed n_samples ({n}).");

        var xData = X.ToArray();

        // Compute pairwise distance matrix (condensed upper triangle)
        var dist = ComputeDistanceMatrix(xData, n, d);

        // Track which original points belong to each cluster
        // clusterMembers[i] = list of original sample indices in cluster i
        var clusterMembers = new List<List<int>>(2 * n);
        for (int i = 0; i < n; i++)
            clusterMembers.Add([i]);

        // Active clusters
        var active = new HashSet<int>(Enumerable.Range(0, n));

        // Distance cache between active cluster pairs
        // Key: (min(i,j), max(i,j)), Value: distance
        var distCache = new Dictionary<(int, int), double>();
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                distCache[(i, j)] = dist[CondensedIndex(i, j, n)];

        // For Ward linkage, track cluster centroids and sizes
        double[]? centroids = null;
        int[]? sizes = null;
        if (Linkage == Linkage.Ward)
        {
            centroids = new double[2 * n * d]; // allocate for merged clusters too
            sizes = new int[2 * n];
            for (int i = 0; i < n; i++)
            {
                sizes[i] = 1;
                for (int j = 0; j < d; j++)
                    centroids[i * d + j] = xData[i * d + j];
            }
        }

        var children = new int[n - 1, 2];
        int nextClusterId = n;
        int numActive = n;

        for (int step = 0; step < n - 1; step++)
        {
            if (numActive <= NClusters) break;

            // Find the pair of active clusters with minimum distance
            int bestA = -1, bestB = -1;
            double bestDist = double.MaxValue;

            foreach (var (pair, d2) in distCache)
            {
                if (!active.Contains(pair.Item1) || !active.Contains(pair.Item2))
                    continue;
                if (d2 < bestDist)
                {
                    bestDist = d2;
                    bestA = pair.Item1;
                    bestB = pair.Item2;
                }
            }

            if (bestA == -1) break;

            // Record merge
            children[step, 0] = bestA;
            children[step, 1] = bestB;

            // Create new merged cluster
            int newId = nextClusterId++;
            var newMembers = new List<int>(clusterMembers[bestA].Count + clusterMembers[bestB].Count);
            newMembers.AddRange(clusterMembers[bestA]);
            newMembers.AddRange(clusterMembers[bestB]);
            // Ensure clusterMembers list is big enough
            while (clusterMembers.Count <= newId)
                clusterMembers.Add([]);
            clusterMembers[newId] = newMembers;

            // Update Ward centroids
            if (Linkage == Linkage.Ward)
            {
                int sA = sizes![bestA], sB = sizes[bestB];
                sizes[newId] = sA + sB;
                // Ensure centroids array is large enough (already preallocated for 2*n)
                for (int j = 0; j < d; j++)
                    centroids![newId * d + j] = (centroids[bestA * d + j] * sA + centroids[bestB * d + j] * sB) / (sA + sB);
            }

            // Remove old clusters, add new one
            active.Remove(bestA);
            active.Remove(bestB);

            // Compute distances from newId to all remaining active clusters
            foreach (int c in active)
            {
                double newDist;
                if (Linkage == Linkage.Ward)
                {
                    newDist = WardDistance(centroids!, sizes!, newId, c, d);
                }
                else
                {
                    newDist = ComputeLinkageDistance(xData, d, clusterMembers[newId], clusterMembers[c]);
                }

                var key = newId < c ? (newId, c) : (c, newId);
                distCache[key] = newDist;
            }

            active.Add(newId);
            numActive--;
        }

        // Assign labels: each active cluster gets a label 0..NClusters-1
        var labels = new int[n];
        int label = 0;
        foreach (int clusterId in active)
        {
            foreach (int sampleIdx in clusterMembers[clusterId])
                labels[sampleIdx] = label;
            label++;
        }

        Labels = labels;
        Children = children;
        IsFitted = true;
        return this;
    }

    /// <summary>
    /// Fit the model. The target vector y is ignored (unsupervised) but accepted for compatibility.
    /// </summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="y">Ignored.</param>
    /// <returns>The fitted model.</returns>
    public AgglomerativeClustering Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        return Fit(X);
    }

    // -- Private helpers --

    private static double[] ComputeDistanceMatrix(double[] xData, int n, int d)
    {
        int size = n * (n - 1) / 2;
        var dist = new double[size];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double sum = 0;
                for (int k = 0; k < d; k++)
                {
                    double diff = xData[i * d + k] - xData[j * d + k];
                    sum += diff * diff;
                }
                dist[CondensedIndex(i, j, n)] = Math.Sqrt(sum);
            }
        }

        return dist;
    }

    private static int CondensedIndex(int i, int j, int n)
    {
        // Upper triangle condensed index for i < j
        if (i > j) (i, j) = (j, i);
        return n * i - i * (i + 1) / 2 + j - i - 1;
    }

    private double ComputeLinkageDistance(double[] xData, int d,
        List<int> membersA, List<int> membersB)
    {
        return Linkage switch
        {
            Linkage.Single => SingleLinkage(xData, d, membersA, membersB),
            Linkage.Complete => CompleteLinkage(xData, d, membersA, membersB),
            Linkage.Average => AverageLinkage(xData, d, membersA, membersB),
            _ => throw new InvalidOperationException("Ward linkage should use WardDistance instead.")
        };
    }

    private static double SingleLinkage(double[] xData, int d,
        List<int> membersA, List<int> membersB)
    {
        double minDist = double.MaxValue;
        foreach (int a in membersA)
        {
            foreach (int b in membersB)
            {
                double dist = EuclideanDistance(xData, a * d, xData, b * d, d);
                if (dist < minDist) minDist = dist;
            }
        }
        return minDist;
    }

    private static double CompleteLinkage(double[] xData, int d,
        List<int> membersA, List<int> membersB)
    {
        double maxDist = 0;
        foreach (int a in membersA)
        {
            foreach (int b in membersB)
            {
                double dist = EuclideanDistance(xData, a * d, xData, b * d, d);
                if (dist > maxDist) maxDist = dist;
            }
        }
        return maxDist;
    }

    private static double AverageLinkage(double[] xData, int d,
        List<int> membersA, List<int> membersB)
    {
        double sumDist = 0;
        int count = 0;
        foreach (int a in membersA)
        {
            foreach (int b in membersB)
            {
                sumDist += EuclideanDistance(xData, a * d, xData, b * d, d);
                count++;
            }
        }
        return sumDist / count;
    }

    private static double WardDistance(double[] centroids, int[] sizes,
        int clusterA, int clusterB, int d)
    {
        int sA = sizes[clusterA];
        int sB = sizes[clusterB];
        double dist2 = 0;
        for (int j = 0; j < d; j++)
        {
            double diff = centroids[clusterA * d + j] - centroids[clusterB * d + j];
            dist2 += diff * diff;
        }
        // Ward distance: sqrt(2 * nA * nB / (nA + nB) * ||cA - cB||^2)
        return Math.Sqrt(2.0 * sA * sB / (sA + sB) * dist2);
    }

    private static double EuclideanDistance(double[] a, int aOff, double[] b, int bOff, int len)
    {
        double sum = 0;
        for (int i = 0; i < len; i++)
        {
            double diff = a[aOff + i] - b[bOff + i];
            sum += diff * diff;
        }
        return Math.Sqrt(sum);
    }
}
