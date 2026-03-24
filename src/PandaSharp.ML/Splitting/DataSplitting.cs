using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Splitting;

public static class DataSplitting
{
    /// <summary>Split DataFrame into train and test sets.</summary>
    public static (DataFrame Train, DataFrame Test) TrainTestSplit(
        this DataFrame df, double testFraction = 0.2, bool shuffle = true, int? seed = null,
        string? stratifyBy = null)
    {
        if (testFraction <= 0 || testFraction >= 1)
            throw new ArgumentOutOfRangeException(nameof(testFraction));

        int n = df.RowCount;
        int testSize = (int)Math.Round(n * testFraction);
        int trainSize = n - testSize;

        int[] indices;
        if (stratifyBy is not null)
            indices = StratifiedIndices(df, stratifyBy, testFraction, seed);
        else
        {
            indices = Enumerable.Range(0, n).ToArray();
            if (shuffle)
            {
                var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
                for (int i = n - 1; i > 0; i--)
                {
                    int j = rng.Next(i + 1);
                    (indices[i], indices[j]) = (indices[j], indices[i]);
                }
            }
        }

        var trainIdx = indices[..trainSize];
        var testIdx = indices[trainSize..];
        Array.Sort(trainIdx);
        Array.Sort(testIdx);

        return (
            new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(trainIdx))),
            new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(testIdx)))
        );
    }

    /// <summary>Three-way split: train, validation, test.</summary>
    public static (DataFrame Train, DataFrame Val, DataFrame Test) TrainValTestSplit(
        this DataFrame df, double valFraction = 0.15, double testFraction = 0.15,
        bool shuffle = true, int? seed = null)
    {
        var (trainVal, test) = df.TrainTestSplit(testFraction, shuffle, seed);
        double valOfRemaining = valFraction / (1 - testFraction);
        var (train, val) = trainVal.TrainTestSplit(valOfRemaining, shuffle, seed.HasValue ? seed.Value + 1 : null);
        return (train, val, test);
    }

    /// <summary>K-Fold cross-validation splits.</summary>
    public static IEnumerable<(int Fold, DataFrame Train, DataFrame Val)> KFold(
        this DataFrame df, int k = 5, bool shuffle = true, int? seed = null)
    {
        int n = df.RowCount;
        var indices = Enumerable.Range(0, n).ToArray();
        if (shuffle)
        {
            var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
            for (int i = n - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }

        int foldSize = n / k;
        for (int fold = 0; fold < k; fold++)
        {
            int valStart = fold * foldSize;
            int valEnd = fold == k - 1 ? n : valStart + foldSize;

            var valIdx = indices[valStart..valEnd];
            var trainIdx = indices[..valStart].Concat(indices[valEnd..]).ToArray();
            Array.Sort(trainIdx);
            Array.Sort(valIdx);

            yield return (fold,
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(trainIdx))),
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(valIdx)))
            );
        }
    }

    private static int[] StratifiedIndices(DataFrame df, string column, double testFraction, int? seed)
    {
        var col = df[column];
        var groups = new Dictionary<object, List<int>>();
        for (int i = 0; i < df.RowCount; i++)
        {
            var key = col.GetObject(i) ?? "__null__";
            if (!groups.TryGetValue(key, out var list))
            {
                list = new List<int>();
                groups[key] = list;
            }
            list.Add(i);
        }

        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        var trainIndices = new List<int>();
        var testIndices = new List<int>();

        foreach (var (_, indices) in groups)
        {
            // Shuffle within group
            for (int i = indices.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
            int testCount = Math.Max(1, (int)Math.Round(indices.Count * testFraction));
            testIndices.AddRange(indices[..testCount]);
            trainIndices.AddRange(indices[testCount..]);
        }

        return trainIndices.Concat(testIndices).ToArray();
    }
}
