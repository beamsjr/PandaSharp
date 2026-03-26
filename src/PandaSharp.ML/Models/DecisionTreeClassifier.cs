using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>Split criterion for classification trees.</summary>
public enum SplitCriterion
{
    /// <summary>Gini impurity.</summary>
    Gini,

    /// <summary>Shannon entropy.</summary>
    Entropy
}

/// <summary>
/// A node in a binary decision tree.
/// </summary>
public sealed class TreeNode
{
    /// <summary>Index of the feature used for splitting (-1 for leaf).</summary>
    public int FeatureIndex { get; set; } = -1;

    /// <summary>Threshold value for the split.</summary>
    public double Threshold { get; set; }

    /// <summary>Left child (samples where feature &lt;= threshold).</summary>
    public TreeNode? Left { get; set; }

    /// <summary>Right child (samples where feature &gt; threshold).</summary>
    public TreeNode? Right { get; set; }

    /// <summary>Class distribution at this node (normalized probabilities).</summary>
    public double[] ClassDistribution { get; set; } = Array.Empty<double>();

    /// <summary>Predicted class label at this node.</summary>
    public int PredictedClass { get; set; }

    /// <summary>Whether this node is a leaf.</summary>
    public bool IsLeaf => Left is null && Right is null;
}

/// <summary>
/// CART decision tree classifier with configurable split criteria,
/// depth limits, and feature sub-sampling.
/// </summary>
public class DecisionTreeClassifier : IClassifier
{
    private readonly SplitCriterion _criterion;
    private readonly int _maxDepth;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;
    private readonly int _maxFeatures;
    private readonly int _seed;

    private TreeNode? _root;
    private int _numClasses;
    private int _numFeatures;

    /// <inheritdoc />
    public string Name => "DecisionTreeClassifier";

    /// <inheritdoc />
    public bool IsFitted => _root is not null;

    /// <inheritdoc />
    public int NumClasses => _numClasses;

    /// <summary>The root node of the fitted tree (for inspection or ensemble use).</summary>
    internal TreeNode? Root => _root;

    /// <summary>
    /// Creates a new decision tree classifier.
    /// </summary>
    /// <param name="criterion">Split quality criterion.</param>
    /// <param name="maxDepth">Maximum depth of the tree. 0 or negative means unlimited.</param>
    /// <param name="minSamplesSplit">Minimum samples required to split a node.</param>
    /// <param name="minSamplesLeaf">Minimum samples required in a leaf.</param>
    /// <param name="maxFeatures">Maximum features to consider per split. 0 or negative means all features.</param>
    /// <param name="seed">Random seed for feature sub-sampling.</param>
    public DecisionTreeClassifier(
        SplitCriterion criterion = SplitCriterion.Gini,
        int maxDepth = 0,
        int minSamplesSplit = 2,
        int minSamplesLeaf = 1,
        int maxFeatures = 0,
        int seed = 42)
    {
        _criterion = criterion;
        _maxDepth = maxDepth;
        _minSamplesSplit = minSamplesSplit;
        _minSamplesLeaf = minSamplesLeaf;
        _maxFeatures = maxFeatures;
        _seed = seed;
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

        int nSamples = X.Shape[0];
        _numFeatures = X.Shape[1];

        // Extract raw arrays for performance
        var xData = X.ToArray();
        var yData = y.ToArray();

        // Determine number of classes
        _numClasses = 0;
        for (int i = 0; i < nSamples; i++)
        {
            int c = (int)yData[i];
            if (c + 1 > _numClasses) _numClasses = c + 1;
        }

        for (int i = 0; i < nSamples; i++)
            if ((int)yData[i] < 0) throw new ArgumentException("Labels must be non-negative.", nameof(y));

        // Pre-sort feature indices for faster split finding
        var sortedIndices = new int[_numFeatures][];
        for (int f = 0; f < _numFeatures; f++)
        {
            var indices = new int[nSamples];
            for (int i = 0; i < nSamples; i++) indices[i] = i;
            int feat = f;
            int nf = _numFeatures;
            Array.Sort(indices, (a, b) => xData[a * nf + feat].CompareTo(xData[b * nf + feat]));
            sortedIndices[f] = indices;
        }

        var sampleMask = new bool[nSamples];
        Array.Fill(sampleMask, true);

        int maxFeat = _maxFeatures > 0 ? Math.Min(_maxFeatures, _numFeatures) : _numFeatures;
        var rng = new Random(_seed);

        _root = BuildTree(xData, yData, sampleMask, nSamples, sortedIndices, 0, maxFeat, rng);
        return this;
    }

    /// <inheritdoc />
    public Tensor<double> Predict(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (!IsFitted) throw new InvalidOperationException("Model has not been fitted.");
        int nSamples = X.Shape[0];
        int nFeatures = X.Shape[1];
        var xData = X.ToArray();
        var result = new double[nSamples];

        for (int i = 0; i < nSamples; i++)
            result[i] = PredictSample(_root!, xData, i, nFeatures).PredictedClass;

        return new Tensor<double>(result, nSamples);
    }

    /// <inheritdoc />
    public Tensor<double> PredictProba(Tensor<double> X)
    {
        ArgumentNullException.ThrowIfNull(X);
        if (!IsFitted) throw new InvalidOperationException("Model has not been fitted.");
        int nSamples = X.Shape[0];
        int nFeatures = X.Shape[1];
        var xData = X.ToArray();
        var result = new double[nSamples * _numClasses];

        for (int i = 0; i < nSamples; i++)
        {
            var leaf = PredictSample(_root!, xData, i, nFeatures);
            Array.Copy(leaf.ClassDistribution, 0, result, i * _numClasses, _numClasses);
        }

        return new Tensor<double>(result, nSamples, _numClasses);
    }

    /// <inheritdoc />
    public double Score(Tensor<double> X, Tensor<double> y)
    {
        ArgumentNullException.ThrowIfNull(X);
        ArgumentNullException.ThrowIfNull(y);
        if (X.Shape[0] == 0) return 0.0;
        var preds = Predict(X);
        var predsArr = preds.ToArray();
        var yArr = y.ToArray();
        int correct = 0;
        for (int i = 0; i < yArr.Length; i++)
        {
            if ((int)predsArr[i] == (int)yArr[i]) correct++;
        }

        return (double)correct / yArr.Length;
    }

    /// <summary>
    /// Predict a single sample (used internally by ensembles).
    /// Returns the leaf node for probability access.
    /// </summary>
    internal TreeNode PredictSingleLeaf(double[] xData, int sampleOffset, int nFeatures)
    {
        return PredictSample(_root!, xData, sampleOffset, nFeatures);
    }

    private static TreeNode PredictSample(TreeNode node, double[] xData, int sampleIndex, int nFeatures)
    {
        var current = node;
        while (!current.IsLeaf)
        {
            double val = xData[sampleIndex * nFeatures + current.FeatureIndex];
            current = val <= current.Threshold ? current.Left! : current.Right!;
        }

        return current;
    }

    private TreeNode BuildTree(
        double[] xData, double[] yData, bool[] sampleMask, int nInMask,
        int[][] sortedIndices, int depth, int maxFeatures, Random rng)
    {
        int nClasses = _numClasses;
        int nFeatures = _numFeatures;

        // Compute class counts
        var classCounts = new double[nClasses];
        int total = 0;
        for (int i = 0; i < sampleMask.Length; i++)
        {
            if (!sampleMask[i]) continue;
            classCounts[(int)yData[i]]++;
            total++;
        }

        // Create leaf distribution
        var dist = new double[nClasses];
        int bestClass = 0;
        double bestCount = 0;
        for (int c = 0; c < nClasses; c++)
        {
            dist[c] = total > 0 ? classCounts[c] / total : 0;
            if (classCounts[c] > bestCount)
            {
                bestCount = classCounts[c];
                bestClass = c;
            }
        }

        // Stopping criteria
        bool isLeaf = total < _minSamplesSplit
                       || (_maxDepth > 0 && depth >= _maxDepth)
                       || bestCount == total; // pure node

        if (isLeaf)
        {
            return new TreeNode
            {
                ClassDistribution = dist,
                PredictedClass = bestClass
            };
        }

        // Select features to consider
        int[] featureCandidates;
        if (maxFeatures < nFeatures)
        {
            var all = new int[nFeatures];
            for (int i = 0; i < nFeatures; i++) all[i] = i;
            Shuffle(all, rng);
            featureCandidates = new int[maxFeatures];
            Array.Copy(all, featureCandidates, maxFeatures);
        }
        else
        {
            featureCandidates = new int[nFeatures];
            for (int i = 0; i < nFeatures; i++) featureCandidates[i] = i;
        }

        double parentImpurity = ComputeImpurity(classCounts, total);

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = double.NegativeInfinity;

        // Try each candidate feature
        foreach (int f in featureCandidates)
        {
            var sorted = sortedIndices[f];
            var leftCounts = new double[nClasses];
            int leftTotal = 0;

            for (int idx = 0; idx < sorted.Length - 1; idx++)
            {
                int si = sorted[idx];
                if (!sampleMask[si]) continue;

                int c = (int)yData[si];
                leftCounts[c]++;
                leftTotal++;

                int rightTotal = total - leftTotal;
                if (leftTotal < _minSamplesLeaf || rightTotal < _minSamplesLeaf)
                    continue;

                // Check for distinct values
                int nextSi = -1;
                for (int k = idx + 1; k < sorted.Length; k++)
                {
                    if (sampleMask[sorted[k]])
                    {
                        nextSi = sorted[k];
                        break;
                    }
                }

                if (nextSi < 0) break;

                double curVal = xData[si * nFeatures + f];
                double nextVal = xData[nextSi * nFeatures + f];
                if (curVal == nextVal) continue;

                // Compute right counts from parent - left
                var rightCounts = new double[nClasses];
                for (int cc = 0; cc < nClasses; cc++)
                    rightCounts[cc] = classCounts[cc] - leftCounts[cc];

                double leftImpurity = ComputeImpurity(leftCounts, leftTotal);
                double rightImpurity = ComputeImpurity(rightCounts, rightTotal);
                double gain = parentImpurity
                              - (double)leftTotal / total * leftImpurity
                              - (double)rightTotal / total * rightImpurity;

                if (gain > bestGain)
                {
                    bestGain = gain;
                    bestFeature = f;
                    bestThreshold = (curVal + nextVal) / 2.0;
                }
            }
        }

        if (bestFeature < 0)
        {
            return new TreeNode
            {
                ClassDistribution = dist,
                PredictedClass = bestClass
            };
        }

        // Split samples
        var leftMask = new bool[sampleMask.Length];
        var rightMask = new bool[sampleMask.Length];
        int leftN = 0, rightN = 0;
        for (int i = 0; i < sampleMask.Length; i++)
        {
            if (!sampleMask[i]) continue;
            if (xData[i * nFeatures + bestFeature] <= bestThreshold)
            {
                leftMask[i] = true;
                leftN++;
            }
            else
            {
                rightMask[i] = true;
                rightN++;
            }
        }

        var node = new TreeNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            ClassDistribution = dist,
            PredictedClass = bestClass,
            Left = BuildTree(xData, yData, leftMask, leftN, sortedIndices, depth + 1, maxFeatures, rng),
            Right = BuildTree(xData, yData, rightMask, rightN, sortedIndices, depth + 1, maxFeatures, rng)
        };

        return node;
    }

    private double ComputeImpurity(double[] classCounts, int total)
    {
        if (total == 0) return 0;
        return _criterion switch
        {
            SplitCriterion.Gini => ComputeGini(classCounts, total),
            SplitCriterion.Entropy => ComputeEntropy(classCounts, total),
            _ => ComputeGini(classCounts, total)
        };
    }

    private static double ComputeGini(double[] classCounts, int total)
    {
        double sum = 0;
        for (int i = 0; i < classCounts.Length; i++)
        {
            double p = classCounts[i] / total;
            sum += p * p;
        }

        return 1.0 - sum;
    }

    private static double ComputeEntropy(double[] classCounts, int total)
    {
        double sum = 0;
        for (int i = 0; i < classCounts.Length; i++)
        {
            if (classCounts[i] <= 0) continue;
            double p = classCounts[i] / total;
            sum -= p * Math.Log2(p);
        }

        return sum;
    }

    private static void Shuffle(int[] array, Random rng)
    {
        for (int i = array.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (array[i], array[j]) = (array[j], array[i]);
        }
    }
}
