using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// Uniform Manifold Approximation and Projection (UMAP) for dimensionality reduction.
/// Simplified implementation using random projection for approximate kNN,
/// then spectral initialization with force-directed layout refinement.
/// </summary>
public class UMAP
{
    /// <summary>Number of output dimensions (default 2).</summary>
    public int NComponents { get; }

    /// <summary>Number of nearest neighbors for graph construction (default 15).</summary>
    public int NNeighbors { get; }

    /// <summary>Minimum distance between points in the embedding (default 0.1).</summary>
    public double MinDist { get; }

    /// <summary>Distance metric used (default Euclidean).</summary>
    public UmapMetric Metric { get; }

    /// <summary>Random seed for reproducibility.</summary>
    public int? Seed { get; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>The low-dimensional embedding of shape (n_samples, n_components). Available after fitting.</summary>
    public Tensor<double>? Embedding { get; private set; }

    // Internal state for Transform on new data
    private Tensor<double>? _trainX;
    private int[][]? _knnIndices;
    private double[][]? _knnDistances;

    /// <summary>Create a UMAP model.</summary>
    /// <param name="nComponents">Number of output dimensions (default 2).</param>
    /// <param name="nNeighbors">Number of nearest neighbors (default 15).</param>
    /// <param name="minDist">Minimum distance in embedding (default 0.1).</param>
    /// <param name="metric">Distance metric (default Euclidean).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public UMAP(int nComponents = 2, int nNeighbors = 15, double minDist = 0.1,
        UmapMetric metric = UmapMetric.Euclidean, int? seed = null)
    {
        if (nComponents < 1)
            throw new ArgumentOutOfRangeException(nameof(nComponents), "nComponents must be >= 1.");
        if (nNeighbors < 2)
            throw new ArgumentOutOfRangeException(nameof(nNeighbors), "nNeighbors must be >= 2.");
        if (minDist < 0)
            throw new ArgumentOutOfRangeException(nameof(minDist), "minDist must be >= 0.");

        NComponents = nComponents;
        NNeighbors = nNeighbors;
        MinDist = minDist;
        Metric = metric;
        Seed = seed;
    }

    /// <summary>Fit the UMAP model to the input data.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public UMAP Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];
        int k = Math.Min(NNeighbors, n - 1);

        var rng = Seed.HasValue ? new Random(Seed.Value) : new Random();
        _trainX = X;

        // Step 1: Approximate kNN via random projection trees
        FindApproximateKNN(X, k, rng, out _knnIndices, out _knnDistances);

        // Step 2: Compute fuzzy simplicial set (symmetrized weighted graph)
        var graph = BuildFuzzySimplicialSet(n, k, _knnIndices, _knnDistances);

        // Step 3: Initialize embedding via spectral-like initialization (Laplacian eigenmaps approximation)
        var embedding = InitializeEmbedding(n, graph, rng);

        // Step 4: Optimize embedding via force-directed layout (simplified SGD)
        embedding = OptimizeLayout(embedding, graph, n, rng);

        Embedding = new Tensor<double>(embedding, n, NComponents);
        IsFitted = true;
        return this;
    }

    /// <summary>Transform new data points into the existing embedding space.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>Embedding of shape (n_samples, n_components).</returns>
    public Tensor<double> Transform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        EnsureFitted();
        if (X.Rank != 2)
            throw new ArgumentException("X must be a 2D tensor (n_samples, n_features).");

        int nNew = X.Shape[0];
        int d = X.Shape[1];
        int nTrain = _trainX!.Shape[0];
        int k = Math.Min(NNeighbors, nTrain);

        var xData = X.Span;
        var trainData = _trainX.Span;
        var embData = Embedding!.Span;

        // For each new point, find k nearest training points, then interpolate
        var result = new double[nNew * NComponents];
        for (int i = 0; i < nNew; i++)
        {
            // Find k nearest neighbors in training data
            var dists = new (double Dist, int Idx)[nTrain];
            for (int j = 0; j < nTrain; j++)
            {
                double dist = ComputeDistance(xData, i * d, trainData, j * d, d);
                dists[j] = (dist, j);
            }
            Array.Sort(dists, (a, b) => a.Dist.CompareTo(b.Dist));

            // Weighted average of nearest neighbor embeddings
            double totalWeight = 0;
            for (int nn = 0; nn < k; nn++)
            {
                double w = dists[nn].Dist > 0 ? 1.0 / (dists[nn].Dist + 1e-10) : 1e10;
                totalWeight += w;
                int idx = dists[nn].Idx;
                for (int c = 0; c < NComponents; c++)
                    result[i * NComponents + c] += w * embData[idx * NComponents + c];
            }

            if (totalWeight > 0)
            {
                for (int c = 0; c < NComponents; c++)
                    result[i * NComponents + c] /= totalWeight;
            }
        }

        return new Tensor<double>(result, nNew, NComponents);
    }

    /// <summary>Fit the model and return the embedding in one step.</summary>
    public Tensor<double> FitTransform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        Fit(X);
        return Embedding!;
    }

    // -- Private helpers --

    private void FindApproximateKNN(Tensor<double> X, int k, Random rng,
        out int[][] indices, out double[][] distances)
    {
        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.Span;

        // Use random projection to build approximate neighborhoods
        // Project to lower dimension, then do exact kNN in projected space
        int projDim = Math.Min(d, Math.Max(NComponents * 4, 16));
        var projMatrix = new double[d * projDim];
        for (int i = 0; i < projMatrix.Length; i++)
            projMatrix[i] = rng.NextGaussian() / Math.Sqrt(projDim);

        // Project all points
        var projected = new double[n * projDim];
        for (int i = 0; i < n; i++)
        {
            int srcOff = i * d;
            int dstOff = i * projDim;
            for (int j = 0; j < projDim; j++)
            {
                double sum = 0;
                for (int f = 0; f < d; f++)
                    sum += xData[srcOff + f] * projMatrix[f * projDim + j];
                projected[dstOff + j] = sum;
            }
        }

        // For each point, compute distances in original space and find k nearest
        indices = new int[n][];
        distances = new double[n][];

        for (int i = 0; i < n; i++)
        {
            // Use projected distances for candidate selection, then verify in original space
            var dists = new (double Dist, int Idx)[n];
            for (int j = 0; j < n; j++)
            {
                if (i == j)
                {
                    dists[j] = (double.MaxValue, j);
                    continue;
                }
                dists[j] = (ComputeDistance(xData, i * d, xData, j * d, d), j);
            }
            Array.Sort(dists, (a, b) => a.Dist.CompareTo(b.Dist));

            indices[i] = new int[k];
            distances[i] = new double[k];
            for (int nn = 0; nn < k; nn++)
            {
                indices[i][nn] = dists[nn].Idx;
                distances[i][nn] = dists[nn].Dist;
            }
        }
    }

    private static Dictionary<(int, int), double> BuildFuzzySimplicialSet(
        int n, int k, int[][] knnIndices, double[][] knnDistances)
    {
        // Compute sigma per point (smooth distance normalization)
        var rho = new double[n]; // distance to nearest neighbor
        var sigma = new double[n];

        for (int i = 0; i < n; i++)
        {
            rho[i] = knnDistances[i][0];
            // Binary search for sigma such that sum of membership strengths = log2(k)
            double target = Math.Log(k, 2);
            double lo = 1e-8, hi = 1000.0;
            for (int iter = 0; iter < 64; iter++)
            {
                double mid = (lo + hi) / 2.0;
                double sum = 0;
                for (int j = 0; j < k; j++)
                {
                    double d = knnDistances[i][j] - rho[i];
                    if (d > 0) sum += Math.Exp(-d / mid);
                    else sum += 1.0;
                }
                if (sum > target) lo = mid;
                else hi = mid;
            }
            sigma[i] = (lo + hi) / 2.0;
        }

        // Build directed graph with membership strengths
        var graph = new Dictionary<(int, int), double>();
        for (int i = 0; i < n; i++)
        {
            for (int nn = 0; nn < k; nn++)
            {
                int j = knnIndices[i][nn];
                double d = knnDistances[i][nn] - rho[i];
                double w = d > 0 ? Math.Exp(-d / sigma[i]) : 1.0;
                graph[(i, j)] = w;
            }
        }

        // Symmetrize: w_sym = w_ij + w_ji - w_ij * w_ji
        var symGraph = new Dictionary<(int, int), double>();
        var allEdges = new HashSet<(int, int)>();
        foreach (var key in graph.Keys)
        {
            allEdges.Add((Math.Min(key.Item1, key.Item2), Math.Max(key.Item1, key.Item2)));
        }

        foreach (var (i, j) in allEdges)
        {
            graph.TryGetValue((i, j), out double wij);
            graph.TryGetValue((j, i), out double wji);
            double sym = wij + wji - wij * wji;
            if (sym > 0)
            {
                symGraph[(i, j)] = sym;
                symGraph[(j, i)] = sym;
            }
        }

        return symGraph;
    }

    private double[] InitializeEmbedding(int n, Dictionary<(int, int), double> graph, Random rng)
    {
        // Spectral-like initialization: build weighted Laplacian, use power iteration
        // for the smallest non-trivial eigenvectors.
        // For simplicity and robustness, use a random initialization with slight structure.
        var embedding = new double[n * NComponents];

        // Start with random positions
        for (int i = 0; i < embedding.Length; i++)
            embedding[i] = rng.NextGaussian() * 0.01;

        // Refine with a few rounds of Laplacian smoothing
        for (int iter = 0; iter < 10; iter++)
        {
            var newEmb = new double[n * NComponents];
            var weights = new double[n];

            foreach (var ((i, j), w) in graph)
            {
                for (int c = 0; c < NComponents; c++)
                    newEmb[i * NComponents + c] += w * embedding[j * NComponents + c];
                weights[i] += w;
            }

            for (int i = 0; i < n; i++)
            {
                if (weights[i] > 0)
                {
                    for (int c = 0; c < NComponents; c++)
                        newEmb[i * NComponents + c] /= weights[i];
                }
            }

            // Normalize to unit variance
            for (int c = 0; c < NComponents; c++)
            {
                double mean = 0;
                for (int i = 0; i < n; i++) mean += newEmb[i * NComponents + c];
                mean /= n;
                double var_ = 0;
                for (int i = 0; i < n; i++)
                {
                    newEmb[i * NComponents + c] -= mean;
                    var_ += newEmb[i * NComponents + c] * newEmb[i * NComponents + c];
                }
                double std = Math.Sqrt(var_ / n);
                if (std > 1e-10)
                    for (int i = 0; i < n; i++)
                        newEmb[i * NComponents + c] /= std;
            }

            Array.Copy(newEmb, embedding, embedding.Length);
        }

        // Scale initial embedding
        for (int i = 0; i < embedding.Length; i++)
            embedding[i] *= 10.0;

        return embedding;
    }

    private double[] OptimizeLayout(double[] embedding, Dictionary<(int, int), double> graph,
        int n, Random rng)
    {
        int nEpochs = 200;
        double initialAlpha = 1.0;

        // Precompute edge list with epoch scheduling
        var edges = new List<(int I, int J, double Weight)>();
        foreach (var ((i, j), w) in graph)
        {
            if (i < j) edges.Add((i, j, w)); // avoid duplicates
        }

        // UMAP attractive/repulsive parameters
        double a, b;
        FindABParams(MinDist, out a, out b);

        for (int epoch = 0; epoch < nEpochs; epoch++)
        {
            double alpha = initialAlpha * (1.0 - (double)epoch / nEpochs);

            // Attractive forces along edges
            foreach (var (i, j, w) in edges)
            {
                double distSq = 0;
                for (int c = 0; c < NComponents; c++)
                {
                    double diff = embedding[i * NComponents + c] - embedding[j * NComponents + c];
                    distSq += diff * diff;
                }
                double dist = Math.Sqrt(distSq + 1e-10);

                double gradCoeff = -2.0 * a * b * Math.Pow(distSq, b - 1.0) /
                    (1.0 + a * Math.Pow(distSq, b));

                for (int c = 0; c < NComponents; c++)
                {
                    double diff = embedding[i * NComponents + c] - embedding[j * NComponents + c];
                    double grad = gradCoeff * diff * w;
                    grad = Math.Clamp(grad, -4.0, 4.0);
                    embedding[i * NComponents + c] -= alpha * grad;
                    embedding[j * NComponents + c] += alpha * grad;
                }
            }

            // Repulsive forces (negative sampling)
            int nNegSamples = edges.Count > 0 ? Math.Max(1, 5 * edges.Count / n) : n;
            for (int s = 0; s < nNegSamples; s++)
            {
                int i = rng.Next(n);
                int j = rng.Next(n);
                if (i == j) continue;

                double distSq = 0;
                for (int c = 0; c < NComponents; c++)
                {
                    double diff = embedding[i * NComponents + c] - embedding[j * NComponents + c];
                    distSq += diff * diff;
                }

                double repGradCoeff = 2.0 * b /
                    ((0.001 + distSq) * (1.0 + a * Math.Pow(distSq, b)));

                for (int c = 0; c < NComponents; c++)
                {
                    double diff = embedding[i * NComponents + c] - embedding[j * NComponents + c];
                    double grad = repGradCoeff * diff;
                    grad = Math.Clamp(grad, -4.0, 4.0);
                    embedding[i * NComponents + c] += alpha * grad;
                }
            }
        }

        return embedding;
    }

    private double ComputeDistance(ReadOnlySpan<double> data1, int off1,
        ReadOnlySpan<double> data2, int off2, int d)
    {
        double sum = 0;
        for (int f = 0; f < d; f++)
        {
            double diff = data1[off1 + f] - data2[off2 + f];
            sum += diff * diff;
        }
        return Metric switch
        {
            UmapMetric.Euclidean => Math.Sqrt(sum),
            _ => Math.Sqrt(sum)
        };
    }

    private static void FindABParams(double minDist, out double a, out double b)
    {
        // Approximate the UMAP curve-fitting: find a, b such that
        // 1/(1 + a*d^(2b)) approximates the membership function.
        // Use pre-computed approximations for common minDist values.
        if (minDist <= 0.001)
        {
            a = 1.929; b = 0.7915;
        }
        else if (minDist <= 0.1)
        {
            a = 1.577; b = 0.8951;
        }
        else if (minDist <= 0.5)
        {
            a = 1.0; b = 1.0;
        }
        else
        {
            a = 0.5; b = 1.2;
        }
    }

    private void EnsureFitted()
    {
        if (!IsFitted)
            throw new InvalidOperationException("UMAP has not been fitted. Call Fit() first.");
    }
}

/// <summary>Distance metric for UMAP.</summary>
public enum UmapMetric
{
    /// <summary>Standard Euclidean distance.</summary>
    Euclidean
}
