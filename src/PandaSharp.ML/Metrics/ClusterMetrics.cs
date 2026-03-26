using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Metrics;

/// <summary>
/// Clustering evaluation metrics for assessing the quality of cluster assignments.
/// </summary>
public static class ClusterMetrics
{
    /// <summary>
    /// Computes the mean Silhouette Score across all samples.
    /// Ranges from -1 (poor) to +1 (well-clustered). Noise labels (-1) are excluded.
    /// </summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="labels">Cluster label for each sample.</param>
    /// <returns>Mean silhouette score.</returns>
    public static double SilhouetteScore(Tensor<double> X, int[] labels)
    {
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");
        if (X.Shape[0] != labels.Length)
            throw new ArgumentException("Number of labels must match number of samples.");

        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.ToArray();

        // Identify unique non-noise clusters
        var uniqueLabels = labels.Where(l => l >= 0).Distinct().OrderBy(l => l).ToArray();
        if (uniqueLabels.Length < 2) return 0.0;

        // Map samples to clusters
        var clusterMembers = new Dictionary<int, List<int>>();
        foreach (int label in uniqueLabels)
            clusterMembers[label] = new List<int>();
        for (int i = 0; i < n; i++)
        {
            if (labels[i] >= 0)
                clusterMembers[labels[i]].Add(i);
        }

        var silhouettes = new double[n];
        int validCount = 0;

        Parallel.For(0, n, i =>
        {
            if (labels[i] < 0) return; // skip noise

            int myCluster = labels[i];
            var myMembers = clusterMembers[myCluster];

            // a(i) = mean distance to same-cluster points
            double a = 0;
            if (myMembers.Count > 1)
            {
                foreach (int j in myMembers)
                {
                    if (j == i) continue;
                    a += EuclideanDistance(xData, i * d, xData, j * d, d);
                }
                a /= (myMembers.Count - 1);
            }

            // b(i) = min mean distance to any other cluster
            double b = double.MaxValue;
            foreach (int otherLabel in uniqueLabels)
            {
                if (otherLabel == myCluster) continue;
                var otherMembers = clusterMembers[otherLabel];
                if (otherMembers.Count == 0) continue;

                double meanDist = 0;
                foreach (int j in otherMembers)
                    meanDist += EuclideanDistance(xData, i * d, xData, j * d, d);
                meanDist /= otherMembers.Count;

                if (meanDist < b) b = meanDist;
            }

            double maxAB = Math.Max(a, b);
            silhouettes[i] = maxAB > 0 ? (b - a) / maxAB : 0;
        });

        // Average over non-noise samples
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            if (labels[i] >= 0) { sum += silhouettes[i]; validCount++; }
        }

        return validCount > 0 ? sum / validCount : 0;
    }

    /// <summary>
    /// Computes the Davies-Bouldin Index. Lower values indicate better clustering.
    /// </summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="labels">Cluster label for each sample.</param>
    /// <returns>Davies-Bouldin index.</returns>
    public static double DaviesBouldinIndex(Tensor<double> X, int[] labels)
    {
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");
        if (X.Shape[0] != labels.Length)
            throw new ArgumentException("Number of labels must match number of samples.");

        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.ToArray();

        var uniqueLabels = labels.Where(l => l >= 0).Distinct().OrderBy(l => l).ToArray();
        int k = uniqueLabels.Length;
        if (k < 2) return 0.0;

        // Compute centroids and scatter (mean distance to centroid)
        var centroids = new double[k * d];
        var counts = new int[k];
        var labelToIdx = new Dictionary<int, int>();
        for (int c = 0; c < k; c++)
            labelToIdx[uniqueLabels[c]] = c;

        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            int c = labelToIdx[labels[i]];
            counts[c]++;
            int cOff = c * d;
            int xOff = i * d;
            for (int j = 0; j < d; j++)
                centroids[cOff + j] += xData[xOff + j];
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                int cOff = c * d;
                for (int j = 0; j < d; j++)
                    centroids[cOff + j] /= counts[c];
            }
        }

        // Scatter: mean distance from points to their centroid
        var scatter = new double[k];
        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            int c = labelToIdx[labels[i]];
            scatter[c] += EuclideanDistance(xData, i * d, centroids, c * d, d);
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0) scatter[c] /= counts[c];
        }

        // DB index: average of max R_ij for each cluster
        double dbSum = 0;
        for (int i = 0; i < k; i++)
        {
            double maxR = 0;
            for (int j = 0; j < k; j++)
            {
                if (i == j) continue;
                double centroidDist = EuclideanDistance(centroids, i * d, centroids, j * d, d);
                double r = centroidDist > 0 ? (scatter[i] + scatter[j]) / centroidDist : 0;
                if (r > maxR) maxR = r;
            }
            dbSum += maxR;
        }

        return dbSum / k;
    }

    /// <summary>
    /// Computes the Calinski-Harabasz Index (Variance Ratio Criterion). Higher values indicate better clustering.
    /// </summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <param name="labels">Cluster label for each sample.</param>
    /// <returns>Calinski-Harabasz index.</returns>
    public static double CalinskiHarabaszIndex(Tensor<double> X, int[] labels)
    {
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");
        if (X.Shape[0] != labels.Length)
            throw new ArgumentException("Number of labels must match number of samples.");

        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.ToArray();

        var uniqueLabels = labels.Where(l => l >= 0).Distinct().OrderBy(l => l).ToArray();
        int k = uniqueLabels.Length;
        if (k < 2) return 0.0;

        var labelToIdx = new Dictionary<int, int>();
        for (int c = 0; c < k; c++)
            labelToIdx[uniqueLabels[c]] = c;

        // Compute overall mean
        var globalMean = new double[d];
        int nValid = 0;
        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            nValid++;
            int xOff = i * d;
            for (int j = 0; j < d; j++)
                globalMean[j] += xData[xOff + j];
        }
        if (nValid == 0) return 0.0;
        for (int j = 0; j < d; j++)
            globalMean[j] /= nValid;

        // Compute cluster centroids and counts
        var centroids = new double[k * d];
        var counts = new int[k];

        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            int c = labelToIdx[labels[i]];
            counts[c]++;
            int cOff = c * d;
            int xOff = i * d;
            for (int j = 0; j < d; j++)
                centroids[cOff + j] += xData[xOff + j];
        }
        for (int c = 0; c < k; c++)
        {
            if (counts[c] > 0)
            {
                int cOff = c * d;
                for (int j = 0; j < d; j++)
                    centroids[cOff + j] /= counts[c];
            }
        }

        // Between-cluster dispersion (BG)
        double bgss = 0;
        for (int c = 0; c < k; c++)
        {
            double dist2 = 0;
            int cOff = c * d;
            for (int j = 0; j < d; j++)
            {
                double diff = centroids[cOff + j] - globalMean[j];
                dist2 += diff * diff;
            }
            bgss += counts[c] * dist2;
        }

        // Within-cluster dispersion (WG)
        double wgss = 0;
        for (int i = 0; i < n; i++)
        {
            if (labels[i] < 0) continue;
            int c = labelToIdx[labels[i]];
            int cOff = c * d;
            int xOff = i * d;
            for (int j = 0; j < d; j++)
            {
                double diff = xData[xOff + j] - centroids[cOff + j];
                wgss += diff * diff;
            }
        }

        if (wgss == 0) return 0.0;

        // CH = (BG / (k - 1)) / (WG / (n - k))
        return (bgss / (k - 1)) / (wgss / (nValid - k));
    }

    // -- Private helpers --

    private static double EuclideanDistance(double[] a, int aOff,
        double[] b, int bOff, int len)
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
