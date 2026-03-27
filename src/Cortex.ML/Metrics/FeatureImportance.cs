using Cortex;
using Cortex.Column;

namespace Cortex.ML.Metrics;

/// <summary>
/// Permutation feature importance: measures how much a model's score drops
/// when each feature is randomly shuffled. Works with any model/metric.
/// </summary>
public static class FeatureImportance
{
    /// <summary>
    /// Compute permutation importance for each feature column.
    /// scorer: function that takes a DataFrame and returns a score (higher = better).
    /// Returns a DataFrame with columns [Feature, Importance, StdDev].
    /// </summary>
    public static DataFrame PermutationImportance(
        DataFrame df, Func<DataFrame, double> scorer,
        string[] featureColumns, int nRepeats = 5, int? seed = null)
    {
        double baselineScore = scorer(df);
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;

        var featureNames = new string[featureColumns.Length];
        var importances = new double[featureColumns.Length];
        var stdDevs = new double[featureColumns.Length];

        for (int f = 0; f < featureColumns.Length; f++)
        {
            featureNames[f] = featureColumns[f];
            var scores = new double[nRepeats];

            for (int r = 0; r < nRepeats; r++)
            {
                // Shuffle this feature column
                var shuffled = ShuffleColumn(df, featureColumns[f], rng);
                scores[r] = baselineScore - scorer(shuffled);
            }

            importances[f] = scores.Average();
            double mean = importances[f];
            stdDevs[f] = nRepeats > 1
                ? Math.Sqrt(scores.Sum(s => (s - mean) * (s - mean)) / (nRepeats - 1))
                : 0;
        }

        // Sort by importance descending
        var indices = Enumerable.Range(0, featureColumns.Length)
            .OrderByDescending(i => importances[i]).ToArray();

        return new DataFrame(
            new StringColumn("Feature", indices.Select(i => featureNames[i]).ToArray()),
            new Column<double>("Importance", indices.Select(i => importances[i]).ToArray()),
            new Column<double>("StdDev", indices.Select(i => stdDevs[i]).ToArray())
        );
    }

    private static DataFrame ShuffleColumn(DataFrame df, string column, Random rng)
    {
        var col = df[column];
        int n = df.RowCount;

        // Build shuffled indices
        var indices = new int[n];
        for (int i = 0; i < n; i++) indices[i] = i;
        for (int i = n - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (indices[i], indices[j]) = (indices[j], indices[i]);
        }

        // Create shuffled column via TakeRows
        var shuffledCol = col.TakeRows(indices);
        return df.Assign(column, shuffledCol);
    }
}
