using Cortex.ML.Tensors;

namespace Cortex.ML.Models;

/// <summary>
/// t-Distributed Stochastic Neighbor Embedding (t-SNE) for dimensionality reduction.
/// Uses gradient descent on KL divergence with early exaggeration.
/// Computes pairwise affinities via binary search for per-point sigma (perplexity).
/// </summary>
public class TSNE
{
    /// <summary>Number of output dimensions (default 2).</summary>
    public int NComponents { get; }

    /// <summary>Perplexity parameter controlling neighborhood size (default 30).</summary>
    public double Perplexity { get; }

    /// <summary>Learning rate for gradient descent (default 200).</summary>
    public double LearningRate { get; }

    /// <summary>Maximum number of gradient descent iterations (default 1000).</summary>
    public int MaxIterations { get; }

    /// <summary>Random seed for reproducibility.</summary>
    public int? Seed { get; }

    /// <summary>Whether the model has been fitted.</summary>
    public bool IsFitted { get; private set; }

    /// <summary>The low-dimensional embedding of shape (n_samples, n_components). Available after fitting.</summary>
    public Tensor<double>? Embedding { get; private set; }

    /// <summary>Final KL divergence between the high- and low-dimensional distributions.</summary>
    public double KLDivergence { get; private set; }

    /// <summary>Create a t-SNE model.</summary>
    /// <param name="nComponents">Number of output dimensions (default 2).</param>
    /// <param name="perplexity">Perplexity (default 30). Should be smaller than n_samples.</param>
    /// <param name="learningRate">Learning rate (default 200).</param>
    /// <param name="maxIterations">Maximum iterations (default 1000).</param>
    /// <param name="seed">Random seed for reproducibility.</param>
    public TSNE(int nComponents = 2, double perplexity = 30.0, double learningRate = 200.0,
        int maxIterations = 1000, int? seed = null)
    {
        if (nComponents < 1)
            throw new ArgumentOutOfRangeException(nameof(nComponents), "nComponents must be >= 1.");
        if (perplexity <= 0)
            throw new ArgumentOutOfRangeException(nameof(perplexity), "perplexity must be > 0.");
        if (learningRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(learningRate), "learningRate must be > 0.");
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), "maxIterations must be >= 1.");

        NComponents = nComponents;
        Perplexity = perplexity;
        LearningRate = learningRate;
        MaxIterations = maxIterations;
        Seed = seed;
    }

    /// <summary>Fit the t-SNE model to the input data.</summary>
    /// <param name="X">Feature matrix of shape (n_samples, n_features).</param>
    /// <returns>The fitted model.</returns>
    public TSNE Fit(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (X.Rank != 2)
            throw new ArgumentException("X must be 2D.", nameof(X));

        int n = X.Shape[0];
        int d = X.Shape[1];
        var xData = X.Span;

        if (Perplexity >= n)
            throw new ArgumentException($"Perplexity ({Perplexity}) must be less than n_samples ({n}).");

        var rng = Seed.HasValue ? new Random(Seed.Value) : new Random();

        // Step 1: Compute pairwise squared distances
        var distSq = ComputePairwiseDistancesSq(xData, n, d);

        // Step 2: Compute pairwise affinities P with Gaussian kernel
        // Binary search for per-point sigma to match target perplexity
        var P = ComputeAffinities(distSq, n);

        // Symmetrize: P_ij = (P_i|j + P_j|i) / (2*n)
        var Psym = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double val = (P[i * n + j] + P[j * n + i]) / (2.0 * n);
                Psym[i * n + j] = val;
                Psym[j * n + i] = val;
            }
        }

        // Step 3: Initialize embedding randomly
        var Y = new double[n * NComponents];
        for (int i = 0; i < Y.Length; i++)
            Y[i] = rng.NextGaussian() * 0.0001;

        // Step 4: Gradient descent with momentum
        var gains = new double[n * NComponents];
        var yIncs = new double[n * NComponents];
        for (int i = 0; i < gains.Length; i++) gains[i] = 1.0;

        double momentum;
        const int earlyExaggerationEnd = 250;
        const double earlyExaggerationFactor = 4.0;

        // Apply early exaggeration
        if (MaxIterations > earlyExaggerationEnd)
        {
            for (int i = 0; i < Psym.Length; i++)
                Psym[i] *= earlyExaggerationFactor;
        }

        var Q = new double[n * n];
        var grad = new double[n * NComponents];

        for (int iter = 0; iter < MaxIterations; iter++)
        {
            momentum = iter < 20 ? 0.5 : 0.8;

            // Remove early exaggeration after 250 iterations
            if (iter == earlyExaggerationEnd)
            {
                for (int i = 0; i < Psym.Length; i++)
                    Psym[i] /= earlyExaggerationFactor;
            }

            // Compute Q distribution (Student-t with 1 dof)
            Array.Clear(Q);
            double qSum = 0;
            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double dSq = 0;
                    for (int c = 0; c < NComponents; c++)
                    {
                        double diff = Y[i * NComponents + c] - Y[j * NComponents + c];
                        dSq += diff * diff;
                    }
                    double qVal = 1.0 / (1.0 + dSq);
                    Q[i * n + j] = qVal;
                    Q[j * n + i] = qVal;
                    qSum += 2.0 * qVal;
                }
            }

            if (qSum > 0)
            {
                for (int i = 0; i < n * n; i++)
                    Q[i] /= qSum;
            }

            // Clamp Q to avoid log(0)
            for (int i = 0; i < n * n; i++)
                Q[i] = Math.Max(Q[i], 1e-12);

            // Compute gradients
            Array.Clear(grad);
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    double pq = Psym[i * n + j] - Q[i * n + j];
                    double dSq = 0;
                    for (int c = 0; c < NComponents; c++)
                    {
                        double diff = Y[i * NComponents + c] - Y[j * NComponents + c];
                        dSq += diff * diff;
                    }
                    double mult = 4.0 * pq / (1.0 + dSq);
                    for (int c = 0; c < NComponents; c++)
                    {
                        double diff = Y[i * NComponents + c] - Y[j * NComponents + c];
                        grad[i * NComponents + c] += mult * diff;
                    }
                }
            }

            // Update with adaptive gains and momentum
            for (int i = 0; i < Y.Length; i++)
            {
                // Adaptive gain: increase if gradient and increment have opposite signs
                bool sameSign = (grad[i] > 0) == (yIncs[i] > 0);
                gains[i] = sameSign ? gains[i] * 0.8 : gains[i] + 0.2;
                gains[i] = Math.Max(gains[i], 0.01);

                yIncs[i] = momentum * yIncs[i] - LearningRate * gains[i] * grad[i];
                Y[i] += yIncs[i];
            }

            // Center embedding
            for (int c = 0; c < NComponents; c++)
            {
                double mean = 0;
                for (int i = 0; i < n; i++) mean += Y[i * NComponents + c];
                mean /= n;
                for (int i = 0; i < n; i++) Y[i * NComponents + c] -= mean;
            }
        }

        // Compute final KL divergence
        KLDivergence = ComputeKLDivergence(Psym, Y, n);

        Embedding = new Tensor<double>(Y, n, NComponents);
        IsFitted = true;
        return this;
    }

    /// <summary>Fit the model and return the embedding in one step.</summary>
    public Tensor<double> FitTransform(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        Fit(X);
        return Embedding!;
    }

    // -- Private helpers --

    private static double[] ComputePairwiseDistancesSq(ReadOnlySpan<double> data, int n, int d)
    {
        var dist = new double[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double sum = 0;
                int offI = i * d, offJ = j * d;
                for (int f = 0; f < d; f++)
                {
                    double diff = data[offI + f] - data[offJ + f];
                    sum += diff * diff;
                }
                dist[i * n + j] = sum;
                dist[j * n + i] = sum;
            }
        }
        return dist;
    }

    private double[] ComputeAffinities(double[] distSq, int n)
    {
        double targetEntropy = Math.Log(Perplexity, 2);
        var P = new double[n * n];

        for (int i = 0; i < n; i++)
        {
            // Binary search for sigma_i
            double betaLo = 1e-8;
            double betaHi = 1e8;
            double beta = 1.0; // beta = 1/(2*sigma^2)

            for (int iter = 0; iter < 50; iter++)
            {
                // Compute conditional probabilities P(j|i)
                double sumExp = 0;
                for (int j = 0; j < n; j++)
                {
                    if (j == i) continue;
                    P[i * n + j] = Math.Exp(-beta * distSq[i * n + j]);
                    sumExp += P[i * n + j];
                }

                if (sumExp > 0)
                {
                    for (int j = 0; j < n; j++)
                    {
                        if (j == i) { P[i * n + j] = 0; continue; }
                        P[i * n + j] /= sumExp;
                    }
                }

                // Compute entropy
                double entropy = 0;
                for (int j = 0; j < n; j++)
                {
                    if (P[i * n + j] > 1e-15)
                        entropy -= P[i * n + j] * Math.Log(P[i * n + j], 2);
                }

                double entropyDiff = entropy - targetEntropy;
                if (Math.Abs(entropyDiff) < 1e-5) break;

                if (entropyDiff > 0)
                    betaLo = beta;
                else
                    betaHi = beta;

                beta = (betaLo + betaHi) / 2.0;
            }
        }

        return P;
    }

    private double ComputeKLDivergence(double[] P, double[] Y, int n)
    {
        // Recompute Q from final embedding
        var Q = new double[n * n];
        double qSum = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dSq = 0;
                for (int c = 0; c < NComponents; c++)
                {
                    double diff = Y[i * NComponents + c] - Y[j * NComponents + c];
                    dSq += diff * diff;
                }
                double qVal = 1.0 / (1.0 + dSq);
                Q[i * n + j] = qVal;
                Q[j * n + i] = qVal;
                qSum += 2.0 * qVal;
            }
        }

        if (qSum > 0)
            for (int i = 0; i < n * n; i++)
                Q[i] /= qSum;

        double kl = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double p = Math.Max(P[i * n + j], 1e-12);
                double q = Math.Max(Q[i * n + j], 1e-12);
                kl += p * Math.Log(p / q);
            }
        }

        return kl;
    }
}
