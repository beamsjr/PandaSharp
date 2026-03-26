using PandaSharp.ML.Native;
using PandaSharp.ML.Spatial;
using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>
/// Distance metric used by KNN models.
/// </summary>
public enum DistanceMetric
{
    /// <summary>L2 (Euclidean) distance.</summary>
    Euclidean,
    /// <summary>L1 (Manhattan) distance.</summary>
    Manhattan,
    /// <summary>Lp (Minkowski) distance with configurable p.</summary>
    Minkowski
}

/// <summary>
/// Brute-force K-Nearest Neighbors classifier.
/// Stores all training data and computes distances at predict time.
/// Supports uniform and distance-weighted voting.
/// Uses BLAS batch distance computation for Euclidean metric when available.
/// </summary>
public class KNearestNeighborsClassifier : IClassifier
{
    private Tensor<double>? _trainX;
    private double[]? _trainY;
    private double[]? _classes;
    private KDTree? _kdTree;

    /// <inheritdoc />
    public string Name => "KNearestNeighborsClassifier";

    /// <inheritdoc />
    public bool IsFitted { get; private set; }

    /// <summary>Number of neighbors to consider.</summary>
    public int K { get; }

    /// <summary>Weighting scheme: "uniform" gives equal weight; "distance" weights by inverse distance.</summary>
    public string Weights { get; }

    /// <summary>Distance metric to use.</summary>
    public DistanceMetric Metric { get; }

    /// <summary>Exponent for Minkowski distance (p=2 is Euclidean, p=1 is Manhattan).</summary>
    public double MinkowskiP { get; }

    /// <inheritdoc />
    public int NumClasses => _classes?.Length ?? 0;

    /// <summary>Create a KNN classifier.</summary>
    /// <param name="k">Number of neighbors (default 5).</param>
    /// <param name="weights">"uniform" or "distance" (default "uniform").</param>
    /// <param name="metric">Distance metric (default Euclidean).</param>
    /// <param name="minkowskiP">Exponent for Minkowski distance (default 2).</param>
    public KNearestNeighborsClassifier(int k = 5, string weights = "uniform",
        DistanceMetric metric = DistanceMetric.Euclidean, double minkowskiP = 2.0)
    {
        if (k < 1) throw new ArgumentOutOfRangeException(nameof(k), "k must be >= 1.");
        if (weights != "uniform" && weights != "distance")
            throw new ArgumentException("weights must be 'uniform' or 'distance'.", nameof(weights));
        if (minkowskiP < 1.0)
            throw new ArgumentOutOfRangeException(nameof(minkowskiP), "Minkowski p must be >= 1.");

        K = k;
        Weights = weights;
        Metric = metric;
        MinkowskiP = minkowskiP;
    }

    /// <inheritdoc />
    public IModel Fit(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));
        if (y.Length != X.Shape[0])
            throw new ArgumentException("y length must match number of samples in X.", nameof(y));

        _trainX = X;
        _trainY = y.ToArray();
        _classes = _trainY.Distinct().OrderBy(c => c).ToArray();

        // Build KD-tree for Euclidean metric when dimensionality is not too high
        int d = X.Shape[1];
        if (Metric == DistanceMetric.Euclidean && d <= 20)
        {
            _kdTree = new KDTree(X.ToArray(), X.Shape[0], d);
        }
        else
        {
            _kdTree = null;
        }

        IsFitted = true;
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int n = X.Shape[0];
        var result = new double[n];

        // Use KD-tree for Euclidean metric when available
        if (_kdTree != null)
        {
            PredictWithKDTree(X, result, n);
        }
        else if (Metric == DistanceMetric.Euclidean)
        {
            PredictBatchEuclidean(X, result, n, predictMode: true);
        }
        else
        {
            Parallel.For(0, n, i =>
            {
                var neighbors = FindKNearest(X, i);
                result[i] = MajorityVote(neighbors);
            });
        }

        return new Tensor<double>(result, n);
    }

    /// <inheritdoc />
    public Tensor<double> PredictProba(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int n = X.Shape[0];
        int nClasses = _classes!.Length;
        var result = new double[n * nClasses];

        if (_kdTree != null)
        {
            PredictProbaWithKDTree(X, result, n, nClasses);
        }
        else if (Metric == DistanceMetric.Euclidean)
        {
            PredictProbaBatchEuclidean(X, result, n, nClasses);
        }
        else
        {
            Parallel.For(0, n, i =>
            {
                var neighbors = FindKNearest(X, i);
                var proba = ComputeProba(neighbors);
                for (int c = 0; c < nClasses; c++)
                    result[i * nClasses + c] = proba[c];
            });
        }

        return new Tensor<double>(result, n, nClasses);
    }

    /// <summary>
    /// Computes accuracy on the given data.
    /// </summary>
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        EnsureFitted();
        if (X.Shape[0] == 0) return 0.0;
        var pred = Predict(X);
        var pSpan = pred.Span;
        var ySpan = y.Span;
        int correct = 0;
        for (int i = 0; i < y.Length; i++)
        {
            if (Math.Abs(pSpan[i] - ySpan[i]) < 1e-12) correct++;
        }
        return (double)correct / y.Length;
    }

    // -- KD-tree accelerated helpers --

    private void PredictWithKDTree(Tensor<double> X, double[] result, int nTest)
    {
        int nTrain = _trainX!.Shape[0];
        int d = _trainX.Shape[1];
        var testArr = X.ToArray();
        int k = Math.Min(K, nTrain);

        Parallel.For(0, nTest, i =>
        {
            var indices = new int[k];
            var distances = new double[k];
            _kdTree!.KnnQuery(testArr, i * d, k, indices, distances);

            var neighbors = new (int Index, double Distance)[k];
            for (int j = 0; j < k; j++)
                neighbors[j] = (indices[j], distances[j]);
            result[i] = MajorityVote(neighbors);
        });
    }

    private void PredictProbaWithKDTree(Tensor<double> X, double[] result, int nTest, int nClasses)
    {
        int nTrain = _trainX!.Shape[0];
        int d = _trainX.Shape[1];
        var testArr = X.ToArray();
        int k = Math.Min(K, nTrain);

        Parallel.For(0, nTest, i =>
        {
            var indices = new int[k];
            var distances = new double[k];
            _kdTree!.KnnQuery(testArr, i * d, k, indices, distances);

            var neighbors = new (int Index, double Distance)[k];
            for (int j = 0; j < k; j++)
                neighbors[j] = (indices[j], distances[j]);
            var proba = ComputeProba(neighbors);
            for (int c = 0; c < nClasses; c++)
                result[i * nClasses + c] = proba[c];
        });
    }

    // -- Batch BLAS distance helpers for Euclidean metric --

    private void PredictBatchEuclidean(Tensor<double> X, double[] result, int nTest, bool predictMode)
    {
        int nTrain = _trainX!.Shape[0];
        int d = _trainX.Shape[1];
        var testArr = X.ToArray();
        var trainArr = _trainX.ToArray();
        int k = Math.Min(K, nTrain);

        // Compute all pairwise squared distances at once using BLAS
        var sqDist = new double[nTest * nTrain];
        BlasOps.PairwiseDistances(testArr, trainArr, sqDist, nTest, nTrain, d);

        Parallel.For(0, nTest, i =>
        {
            // Build (index, distance) for this query row and partial sort for K nearest
            var neighbors = new (int Index, double Distance)[nTrain];
            int rowOff = i * nTrain;
            for (int j = 0; j < nTrain; j++)
            {
                double sd = sqDist[rowOff + j];
                if (sd < 0) sd = 0; // numerical fix
                neighbors[j] = (j, Math.Sqrt(sd));
            }
            Array.Sort(neighbors, (a, b) => a.Distance.CompareTo(b.Distance));
            var topK = neighbors.AsSpan(0, k).ToArray();
            result[i] = MajorityVote(topK);
        });
    }

    private void PredictProbaBatchEuclidean(Tensor<double> X, double[] result, int nTest, int nClasses)
    {
        int nTrain = _trainX!.Shape[0];
        int d = _trainX.Shape[1];
        var testArr = X.ToArray();
        var trainArr = _trainX.ToArray();
        int k = Math.Min(K, nTrain);

        var sqDist = new double[nTest * nTrain];
        BlasOps.PairwiseDistances(testArr, trainArr, sqDist, nTest, nTrain, d);

        Parallel.For(0, nTest, i =>
        {
            var neighbors = new (int Index, double Distance)[nTrain];
            int rowOff = i * nTrain;
            for (int j = 0; j < nTrain; j++)
            {
                double sd = sqDist[rowOff + j];
                if (sd < 0) sd = 0;
                neighbors[j] = (j, Math.Sqrt(sd));
            }
            Array.Sort(neighbors, (a, b) => a.Distance.CompareTo(b.Distance));
            var topK = neighbors.AsSpan(0, k).ToArray();
            var proba = ComputeProba(topK);
            for (int c = 0; c < nClasses; c++)
                result[i * nClasses + c] = proba[c];
        });
    }

    // -- Private helpers (fallback for non-Euclidean metrics) --

    /// <summary>Returns (index, distance) pairs for the K nearest neighbors of sample i in X.</summary>
    private (int Index, double Distance)[] FindKNearest(Tensor<double> X, int sampleIdx)
    {
        int nTrain = _trainX!.Shape[0];
        int nFeatures = _trainX.Shape[1];
        var xSpan = X.Span;
        var trainSpan = _trainX.Span;
        int queryOffset = sampleIdx * nFeatures;

        // Compute all distances
        var distances = new (int Index, double Distance)[nTrain];
        for (int j = 0; j < nTrain; j++)
        {
            int trainOffset = j * nFeatures;
            double dist = ComputeDistance(xSpan, queryOffset, trainSpan, trainOffset, nFeatures);
            distances[j] = (j, dist);
        }

        // Partial sort: get top K using a selection approach
        Array.Sort(distances, (a, b) => a.Distance.CompareTo(b.Distance));
        return distances.AsSpan(0, Math.Min(K, nTrain)).ToArray();
    }

    private double ComputeDistance(ReadOnlySpan<double> a, int aOff,
        ReadOnlySpan<double> b, int bOff, int len)
    {
        double sum = 0;
        switch (Metric)
        {
            case DistanceMetric.Euclidean:
                for (int i = 0; i < len; i++)
                {
                    double d = a[aOff + i] - b[bOff + i];
                    sum += d * d;
                }
                return Math.Sqrt(sum);

            case DistanceMetric.Manhattan:
                for (int i = 0; i < len; i++)
                    sum += Math.Abs(a[aOff + i] - b[bOff + i]);
                return sum;

            case DistanceMetric.Minkowski:
                double p = MinkowskiP;
                for (int i = 0; i < len; i++)
                    sum += Math.Pow(Math.Abs(a[aOff + i] - b[bOff + i]), p);
                return Math.Pow(sum, 1.0 / p);

            default:
                throw new InvalidOperationException($"Unknown distance metric: {Metric}");
        }
    }

    private double MajorityVote((int Index, double Distance)[] neighbors)
    {
        if (Weights == "uniform")
        {
            // Simple majority vote
            var counts = new Dictionary<double, int>();
            foreach (var (idx, _) in neighbors)
            {
                double label = _trainY![idx];
                counts.TryGetValue(label, out int c);
                counts[label] = c + 1;
            }
            return counts.MaxBy(kv => kv.Value).Key;
        }
        else
        {
            // Distance-weighted vote
            var weightedCounts = new Dictionary<double, double>();
            foreach (var (idx, dist) in neighbors)
            {
                double label = _trainY![idx];
                double w = dist < 1e-15 ? 1e15 : 1.0 / dist;
                weightedCounts.TryGetValue(label, out double c);
                weightedCounts[label] = c + w;
            }
            return weightedCounts.MaxBy(kv => kv.Value).Key;
        }
    }

    private double[] ComputeProba((int Index, double Distance)[] neighbors)
    {
        int nClasses = _classes!.Length;
        var proba = new double[nClasses];

        if (Weights == "uniform")
        {
            foreach (var (idx, _) in neighbors)
            {
                double label = _trainY![idx];
                int classIdx = Array.IndexOf(_classes, label);
                proba[classIdx] += 1.0;
            }
            double total = neighbors.Length;
            for (int c = 0; c < nClasses; c++)
                proba[c] /= total;
        }
        else
        {
            double totalWeight = 0;
            foreach (var (idx, dist) in neighbors)
            {
                double label = _trainY![idx];
                int classIdx = Array.IndexOf(_classes, label);
                double w = dist < 1e-15 ? 1e15 : 1.0 / dist;
                proba[classIdx] += w;
                totalWeight += w;
            }
            if (totalWeight > 0)
            {
                for (int c = 0; c < nClasses; c++)
                    proba[c] /= totalWeight;
            }
        }

        return proba;
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
