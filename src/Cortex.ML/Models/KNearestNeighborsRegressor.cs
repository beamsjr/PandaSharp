using Cortex.ML.Native;
using Cortex.ML.Spatial;
using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Brute-force K-Nearest Neighbors regressor.
/// Stores all training data and computes distances at predict time.
/// Predicts the mean (or distance-weighted mean) of the K nearest targets.
/// Uses BLAS batch distance computation for Euclidean metric when available.
/// </summary>
public class KNearestNeighborsRegressor : IModel
{
    private Tensor<double>? _trainX;
    private double[]? _trainY;
    private KDTree? _kdTree;

    /// <inheritdoc />
    public string Name => "KNearestNeighborsRegressor";

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

    /// <summary>Create a KNN regressor.</summary>
    /// <param name="k">Number of neighbors (default 5).</param>
    /// <param name="weights">"uniform" or "distance" (default "uniform").</param>
    /// <param name="metric">Distance metric (default Euclidean).</param>
    /// <param name="minkowskiP">Exponent for Minkowski distance (default 2).</param>
    public KNearestNeighborsRegressor(int k = 5, string weights = "uniform",
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
            PredictBatchEuclidean(X, result, n);
        }
        else
        {
            Parallel.For(0, n, i =>
            {
                var neighbors = FindKNearest(X, i);
                result[i] = ComputePrediction(neighbors);
            });
        }

        return new Tensor<double>(result, n);
    }

    /// <summary>
    /// Computes R² score on the given data.
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
        int n = y.Length;

        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += ySpan[i];
        yMean /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = ySpan[i] - pSpan[i];
            ssRes += diff * diff;
            double diffMean = ySpan[i] - yMean;
            ssTot += diffMean * diffMean;
        }

        return ssTot == 0 ? (ssRes == 0 ? 1.0 : 0.0) : 1.0 - ssRes / ssTot;
    }

    // -- KD-tree accelerated helper --

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
            result[i] = ComputePrediction(neighbors);
        });
    }

    // -- Batch BLAS distance helper for Euclidean metric --

    private void PredictBatchEuclidean(Tensor<double> X, double[] result, int nTest)
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
            result[i] = ComputePrediction(topK);
        });
    }

    // -- Private helpers (fallback for non-Euclidean metrics) --

    private (int Index, double Distance)[] FindKNearest(Tensor<double> X, int sampleIdx)
    {
        int nTrain = _trainX!.Shape[0];
        int nFeatures = _trainX.Shape[1];
        var xSpan = X.Span;
        var trainSpan = _trainX.Span;
        int queryOffset = sampleIdx * nFeatures;

        var distances = new (int Index, double Distance)[nTrain];
        for (int j = 0; j < nTrain; j++)
        {
            int trainOffset = j * nFeatures;
            double dist = ComputeDistance(xSpan, queryOffset, trainSpan, trainOffset, nFeatures);
            distances[j] = (j, dist);
        }

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

    private double ComputePrediction((int Index, double Distance)[] neighbors)
    {
        if (Weights == "uniform")
        {
            double sum = 0;
            foreach (var (idx, _) in neighbors)
                sum += _trainY![idx];
            return sum / neighbors.Length;
        }
        else
        {
            // If all distances are zero (duplicate training points), fall back to uniform weighting
            bool allZero = true;
            foreach (var (_, dist) in neighbors)
            {
                if (dist >= 1e-15) { allZero = false; break; }
            }
            if (allZero)
            {
                double sum = 0;
                foreach (var (idx, _) in neighbors)
                    sum += _trainY![idx];
                return sum / neighbors.Length;
            }

            double weightedSum = 0;
            double totalWeight = 0;
            foreach (var (idx, dist) in neighbors)
            {
                double w = dist < 1e-15 ? 1e15 : 1.0 / dist;
                weightedSum += w * _trainY![idx];
                totalWeight += w;
            }
            return totalWeight > 0 ? weightedSum / totalWeight : 0;
        }
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("Model has not been fitted. Call Fit() first.");
    }
}
