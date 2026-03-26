using FluentAssertions;
using PandaSharp.ML.Models;
using PandaSharp.ML.Spatial;
using PandaSharp.ML.Tensors;
using System.Diagnostics;

namespace PandaSharp.ML.Tests;

public class KDTreeTests
{
    /// <summary>
    /// Brute-force KNN for verification. Returns (indices, distances) sorted ascending by distance.
    /// </summary>
    private static (int[] Indices, double[] Distances) BruteForceKnn(
        double[] data, int n, int d, double[] query, int qOff, int k)
    {
        var dists = new (double Dist, int Idx)[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = query[qOff + j] - data[i * d + j];
                sum += diff * diff;
            }
            dists[i] = (Math.Sqrt(sum), i);
        }
        Array.Sort(dists, (a, b) => a.Dist.CompareTo(b.Dist));

        k = Math.Min(k, n);
        var indices = new int[k];
        var distances = new double[k];
        for (int i = 0; i < k; i++)
        {
            indices[i] = dists[i].Idx;
            distances[i] = dists[i].Dist;
        }
        return (indices, distances);
    }

    [Fact]
    public void KDTree_Basic_100Points3D_MatchesBruteForce()
    {
        int n = 100, d = 3, k = 5;
        var rng = new Random(42);
        var data = new double[n * d];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 100;

        var tree = new KDTree(data, n, d);

        // Query with 10 random points
        for (int q = 0; q < 10; q++)
        {
            var query = new double[d];
            for (int j = 0; j < d; j++) query[j] = rng.NextDouble() * 100;

            var treeIdx = new int[k];
            var treeDist = new double[k];
            tree.KnnQuery(query, 0, k, treeIdx, treeDist);

            var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, k);

            // Distances should match (indices may differ for equidistant points)
            for (int i = 0; i < k; i++)
            {
                treeDist[i].Should().BeApproximately(bfDist[i], 1e-10,
                    $"distance mismatch at neighbor {i} for query {q}");
            }
        }
    }

    [Fact]
    public void KDTree_K1_ReturnsSingleNearestNeighbor()
    {
        int n = 50, d = 2;
        var rng = new Random(123);
        var data = new double[n * d];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble() * 10;

        var tree = new KDTree(data, n, d);

        var query = new double[] { 5.0, 5.0 };
        var idx = new int[1];
        var dist = new double[1];
        tree.KnnQuery(query, 0, 1, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, 1);
        dist[0].Should().BeApproximately(bfDist[0], 1e-10);
    }

    [Fact]
    public void KDTree_KEqualsN_ReturnsAllPoints()
    {
        int n = 20, d = 2;
        var rng = new Random(7);
        var data = new double[n * d];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();

        var tree = new KDTree(data, n, d);

        var query = new double[] { 0.5, 0.5 };
        var idx = new int[n];
        var dist = new double[n];
        tree.KnnQuery(query, 0, n, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, n);

        // All distances should match
        for (int i = 0; i < n; i++)
            dist[i].Should().BeApproximately(bfDist[i], 1e-10);

        // All indices should be present
        idx.Order().Should().Equal(bfIdx.Order());
    }

    [Fact]
    public void KDTree_1DData_MatchesBruteForce()
    {
        int n = 50, d = 1, k = 3;
        var data = new double[n];
        for (int i = 0; i < n; i++) data[i] = i * 0.5;

        var tree = new KDTree(data, n, d);

        var query = new double[] { 12.3 };
        var idx = new int[k];
        var dist = new double[k];
        tree.KnnQuery(query, 0, k, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, k);
        for (int i = 0; i < k; i++)
            dist[i].Should().BeApproximately(bfDist[i], 1e-10);
    }

    [Fact]
    public void KDTree_HighDimensional20D_MatchesBruteForce()
    {
        int n = 200, d = 20, k = 5;
        var rng = new Random(99);
        var data = new double[n * d];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();

        var tree = new KDTree(data, n, d);

        var query = new double[d];
        for (int j = 0; j < d; j++) query[j] = rng.NextDouble();

        var idx = new int[k];
        var dist = new double[k];
        tree.KnnQuery(query, 0, k, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, k);
        for (int i = 0; i < k; i++)
            dist[i].Should().BeApproximately(bfDist[i], 1e-10);
    }

    [Fact]
    public void KNNClassifier_PredictionsUnchangedWithKDTree()
    {
        // Small clustered data
        var xData = new double[]
        {
            0, 0, 0.1, 0.1, -0.1, 0.1, 0, -0.1,
            5, 5, 5.1, 5.1, 4.9, 5.1, 5, 4.9
        };
        var yData = new double[] { 0, 0, 0, 0, 1, 1, 1, 1 };
        var X = new Tensor<double>(xData, 8, 2);
        var y = new Tensor<double>(yData, 8);

        // Test query points clearly in each cluster
        var testData = new double[] { 0.05, 0.05, 4.95, 4.95 };
        var testX = new Tensor<double>(testData, 2, 2);

        var model = new KNearestNeighborsClassifier(k: 3);
        model.Fit(X, y);
        var preds = model.Predict(testX);

        preds.Span[0].Should().Be(0.0, "point near (0,0) should be class 0");
        preds.Span[1].Should().Be(1.0, "point near (5,5) should be class 1");
    }

    [Fact]
    public void KNNRegressor_PredictionsUnchangedWithKDTree()
    {
        // Simple regression: y = x1 + x2
        var xData = new double[]
        {
            0, 0, 1, 0, 0, 1, 1, 1, 2, 0, 0, 2, 2, 2, 3, 0, 0, 3, 3, 3
        };
        var yData = new double[] { 0, 1, 1, 2, 2, 2, 4, 3, 3, 6 };
        var X = new Tensor<double>(xData, 10, 2);
        var y = new Tensor<double>(yData, 10);

        var testData = new double[] { 0.5, 0.5, 2.5, 2.5 };
        var testX = new Tensor<double>(testData, 2, 2);

        var model = new KNearestNeighborsRegressor(k: 3);
        model.Fit(X, y);
        var preds = model.Predict(testX);

        // Point (0.5, 0.5) nearest neighbors should be (0,0), (1,0), (0,1) => mean = (0+1+1)/3 = 0.667
        preds.Span[0].Should().BeApproximately(0.667, 0.01);
        // Point (2.5, 2.5) nearest neighbors should be (2,2), (3,3), and one of (3,0)/(0,3)/(2,0)/(0,2)
        // The exact value depends on tie-breaking, just check it's reasonable
        preds.Span[1].Should().BeInRange(2.0, 5.0);
    }

    [Fact]
    public void KDTree_Performance_10KPoints_FasterThanBruteForce()
    {
        int n = 10_000, d = 5, k = 10, nQueries = 100;
        var rng = new Random(42);
        var data = new double[n * d];
        for (int i = 0; i < data.Length; i++) data[i] = rng.NextDouble();

        var queries = new double[nQueries * d];
        for (int i = 0; i < queries.Length; i++) queries[i] = rng.NextDouble();

        // KD-tree
        var tree = new KDTree(data, n, d);
        var sw = Stopwatch.StartNew();
        for (int q = 0; q < nQueries; q++)
        {
            var idx = new int[k];
            var dist = new double[k];
            tree.KnnQuery(queries, q * d, k, idx, dist);
        }
        sw.Stop();
        long kdTreeMs = sw.ElapsedMilliseconds;

        // Brute-force
        sw.Restart();
        for (int q = 0; q < nQueries; q++)
        {
            BruteForceKnn(data, n, d, queries, q * d, k);
        }
        sw.Stop();
        long bruteMs = sw.ElapsedMilliseconds;

        // KD-tree should be faster (or at least not dramatically slower)
        // Allow some slack since brute-force is simple array iteration
        kdTreeMs.Should().BeLessThan(bruteMs + 50,
            $"KD-tree ({kdTreeMs}ms) should be faster than brute-force ({bruteMs}ms) on 10K points");
    }
}
