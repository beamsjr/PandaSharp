using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Models;

/// <summary>Split criterion for regression trees.</summary>
public enum RegressorCriterion
{
    /// <summary>Mean squared error.</summary>
    MSE,

    /// <summary>Mean absolute error.</summary>
    MAE
}

/// <summary>
/// A node in a regression decision tree.
/// </summary>
public sealed class RegressionTreeNode
{
    /// <summary>Index of the feature used for splitting (-1 for leaf).</summary>
    public int FeatureIndex { get; set; } = -1;

    /// <summary>Threshold value for the split.</summary>
    public double Threshold { get; set; }

    /// <summary>Left child (samples where feature &lt;= threshold).</summary>
    public RegressionTreeNode? Left { get; set; }

    /// <summary>Right child (samples where feature &gt; threshold).</summary>
    public RegressionTreeNode? Right { get; set; }

    /// <summary>Predicted value at this node (mean of targets).</summary>
    public double PredictedValue { get; set; }

    /// <summary>Whether this node is a leaf.</summary>
    public bool IsLeaf => Left is null && Right is null;
}

/// <summary>
/// CART decision tree regressor with configurable split criteria (MSE or MAE),
/// depth limits, and feature sub-sampling.
/// </summary>
public class DecisionTreeRegressor : IModel
{
    private readonly RegressorCriterion _criterion;
    private readonly int _maxDepth;
    private readonly int _minSamplesSplit;
    private readonly int _minSamplesLeaf;
    private readonly int _maxFeatures;
    private readonly int _seed;

    private RegressionTreeNode? _root;
    private int _numFeatures;

    /// <inheritdoc />
    public string Name => "DecisionTreeRegressor";

    /// <inheritdoc />
    public bool IsFitted => _root is not null;

    /// <summary>The root node of the fitted tree.</summary>
    internal RegressionTreeNode? Root => _root;

    /// <summary>
    /// Creates a new decision tree regressor.
    /// </summary>
    public DecisionTreeRegressor(
        RegressorCriterion criterion = RegressorCriterion.MSE,
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

        var xData = X.ToArray();
        var yData = y.ToArray();

        // Pre-sort feature indices once (shared across all nodes)
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
            result[i] = PredictSample(_root!, xData, i, nFeatures);

        return new Tensor<double>(result, nSamples);
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
        int n = yArr.Length;

        double yMean = 0;
        for (int i = 0; i < n; i++) yMean += yArr[i];
        yMean /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double diff = yArr[i] - predsArr[i];
            ssRes += diff * diff;
            double diffMean = yArr[i] - yMean;
            ssTot += diffMean * diffMean;
        }

        return ssTot == 0 ? (ssRes == 0 ? 1.0 : 0.0) : 1.0 - ssRes / ssTot;
    }

    /// <summary>
    /// Predict a single sample value (used internally by ensembles).
    /// </summary>
    internal double PredictSingleValue(double[] xData, int sampleIndex, int nFeatures)
    {
        return PredictSample(_root!, xData, sampleIndex, nFeatures);
    }

    private static double PredictSample(RegressionTreeNode node, double[] xData, int sampleIndex, int nFeatures)
    {
        var current = node;
        while (!current.IsLeaf)
        {
            double val = xData[sampleIndex * nFeatures + current.FeatureIndex];
            current = val <= current.Threshold ? current.Left! : current.Right!;
        }

        return current.PredictedValue;
    }

    private RegressionTreeNode BuildTree(
        double[] xData, double[] yData, bool[] sampleMask, int nInMask,
        int[][] sortedIndices, int depth, int maxFeatures, Random rng)
    {
        int nFeatures = _numFeatures;

        // Compute mean and sum-of-squares of targets
        double sum = 0, sumSq = 0;
        int total = 0;
        for (int i = 0; i < sampleMask.Length; i++)
        {
            if (!sampleMask[i]) continue;
            double yi = yData[i];
            sum += yi;
            sumSq += yi * yi;
            total++;
        }

        double mean = total > 0 ? sum / total : 0;

        // Stopping criteria
        if (total < _minSamplesSplit || (_maxDepth > 0 && depth >= _maxDepth))
        {
            return new RegressionTreeNode { PredictedValue = mean };
        }

        // Check if all values are the same
        bool allSame = true;
        double firstVal = double.NaN;
        for (int i = 0; i < sampleMask.Length; i++)
        {
            if (!sampleMask[i]) continue;
            if (double.IsNaN(firstVal))
            {
                firstVal = yData[i];
            }
            else if (yData[i] != firstVal)
            {
                allSame = false;
                break;
            }
        }

        if (allSame)
        {
            return new RegressionTreeNode { PredictedValue = mean };
        }

        // Select features
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

        // Parent impurity: MSE = (sumSq/n) - mean^2
        double parentImpurity = _criterion == RegressorCriterion.MSE
            ? (sumSq / total) - (mean * mean)
            : ComputeImpurity(yData, sampleMask, mean, total);

        int bestFeature = -1;
        double bestThreshold = 0;
        double bestGain = double.NegativeInfinity;

        foreach (int f in featureCandidates)
        {
            var sorted = sortedIndices[f];

            if (_criterion == RegressorCriterion.MSE)
            {
                // O(1) per split using running sum and sum-of-squares
                // MSE = E[y²] - E[y]² = (sumSq/n) - (sum/n)²
                FindBestSplitMSE(xData, yData, sampleMask, sorted, f, nFeatures,
                    total, sum, sumSq, parentImpurity,
                    ref bestFeature, ref bestThreshold, ref bestGain);
            }
            else
            {
                // MAE requires O(N) per split (no incremental formula)
                FindBestSplitMAE(xData, yData, sampleMask, sorted, f, nFeatures,
                    total, sum, parentImpurity,
                    ref bestFeature, ref bestThreshold, ref bestGain);
            }
        }

        if (bestFeature < 0)
        {
            return new RegressionTreeNode { PredictedValue = mean };
        }

        // Split
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

        return new RegressionTreeNode
        {
            FeatureIndex = bestFeature,
            Threshold = bestThreshold,
            PredictedValue = mean,
            Left = BuildTree(xData, yData, leftMask, leftN, sortedIndices, depth + 1, maxFeatures, rng),
            Right = BuildTree(xData, yData, rightMask, rightN, sortedIndices, depth + 1, maxFeatures, rng)
        };
    }

    /// <summary>
    /// Find best split for MSE criterion using O(1) incremental impurity.
    /// MSE = (sumSq/n) - (sum/n)² = variance
    /// When we move sample i from right to left:
    ///   leftSum += y_i, leftSumSq += y_i², leftCount++
    ///   rightSum -= y_i, rightSumSq -= y_i², rightCount--
    /// Then MSE = sumSq/n - (sum/n)² for each side.
    /// Weighted impurity = (nL/n)*MSE_L + (nR/n)*MSE_R
    /// </summary>
    private void FindBestSplitMSE(
        double[] xData, double[] yData, bool[] sampleMask, int[] sorted,
        int featureIdx, int nFeatures,
        int totalCount, double totalSum, double totalSumSq,
        double parentImpurity,
        ref int bestFeature, ref double bestThreshold, ref double bestGain)
    {
        double leftSum = 0, leftSumSq = 0;
        int leftCount = 0;

        for (int idx = 0; idx < sorted.Length - 1; idx++)
        {
            int si = sorted[idx];
            if (!sampleMask[si]) continue;

            double yi = yData[si];
            leftSum += yi;
            leftSumSq += yi * yi;
            leftCount++;

            int rightCount = totalCount - leftCount;
            if (leftCount < _minSamplesLeaf || rightCount < _minSamplesLeaf)
                continue;

            // Find next valid sample to check for same feature value
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

            double curVal = xData[si * nFeatures + featureIdx];
            double nextVal = xData[nextSi * nFeatures + featureIdx];
            if (curVal == nextVal) continue;

            // O(1) MSE computation
            double leftMean = leftSum / leftCount;
            double leftMSE = (leftSumSq / leftCount) - (leftMean * leftMean);

            double rightSum = totalSum - leftSum;
            double rightSumSq = totalSumSq - leftSumSq;
            double rightMean = rightSum / rightCount;
            double rightMSE = (rightSumSq / rightCount) - (rightMean * rightMean);

            double gain = parentImpurity
                          - (double)leftCount / totalCount * leftMSE
                          - (double)rightCount / totalCount * rightMSE;

            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = featureIdx;
                bestThreshold = (curVal + nextVal) / 2.0;
            }
        }
    }

    /// <summary>
    /// Find best split for MAE criterion (requires O(N) per split — no incremental formula).
    /// </summary>
    private void FindBestSplitMAE(
        double[] xData, double[] yData, bool[] sampleMask, int[] sorted,
        int featureIdx, int nFeatures,
        int totalCount, double totalSum, double parentImpurity,
        ref int bestFeature, ref double bestThreshold, ref double bestGain)
    {
        double leftSum = 0;
        int leftCount = 0;

        for (int idx = 0; idx < sorted.Length - 1; idx++)
        {
            int si = sorted[idx];
            if (!sampleMask[si]) continue;

            leftSum += yData[si];
            leftCount++;

            int rightCount = totalCount - leftCount;
            if (leftCount < _minSamplesLeaf || rightCount < _minSamplesLeaf)
                continue;

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

            double curVal = xData[si * nFeatures + featureIdx];
            double nextVal = xData[nextSi * nFeatures + featureIdx];
            if (curVal == nextVal) continue;

            double leftMean = leftSum / leftCount;
            double rightSum = totalSum - leftSum;
            double rightMean = rightSum / rightCount;

            double leftImpurity = ComputeImpurityFromMean(yData, sampleMask, sorted, 0, idx + 1, leftMean, leftCount);
            double rightImpurity = ComputeImpurityFromMean(yData, sampleMask, sorted, idx + 1, sorted.Length, rightMean, rightCount);

            double gain = parentImpurity
                          - (double)leftCount / totalCount * leftImpurity
                          - (double)rightCount / totalCount * rightImpurity;

            if (gain > bestGain)
            {
                bestGain = gain;
                bestFeature = featureIdx;
                bestThreshold = (curVal + nextVal) / 2.0;
            }
        }
    }

    private double ComputeImpurity(double[] yData, bool[] mask, double mean, int total)
    {
        if (total == 0) return 0;

        if (_criterion == RegressorCriterion.MSE)
        {
            double sse = 0;
            for (int i = 0; i < mask.Length; i++)
            {
                if (!mask[i]) continue;
                double d = yData[i] - mean;
                sse += d * d;
            }

            return sse / total;
        }
        else // MAE
        {
            double sae = 0;
            for (int i = 0; i < mask.Length; i++)
            {
                if (!mask[i]) continue;
                sae += Math.Abs(yData[i] - mean);
            }

            return sae / total;
        }
    }

    private double ComputeImpurityFromMean(
        double[] yData, bool[] mask, int[] sorted, int start, int end, double mean, int total)
    {
        if (total == 0) return 0;

        // Only used for MAE now (MSE uses incremental computation)
        double sae = 0;
        for (int idx = start; idx < end; idx++)
        {
            int si = sorted[idx];
            if (!mask[si]) continue;
            sae += Math.Abs(yData[si] - mean);
        }

        return sae / total;
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
