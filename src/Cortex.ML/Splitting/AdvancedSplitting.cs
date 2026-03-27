using Cortex;

namespace Cortex.ML.Splitting;

public static class AdvancedSplitting
{
    /// <summary>Stratified K-Fold: maintains class distribution in each fold.</summary>
    public static IEnumerable<(int Fold, DataFrame Train, DataFrame Val)> StratifiedKFold(
        this DataFrame df, int k = 5, string column = "label", int? seed = null)
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

        // Shuffle within each group
        var rng = seed.HasValue ? new Random(seed.Value) : Random.Shared;
        foreach (var list in groups.Values)
        {
            for (int i = list.Count - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (list[i], list[j]) = (list[j], list[i]);
            }
        }

        // Assign each group's elements to folds round-robin
        var foldAssignments = new int[df.RowCount];
        foreach (var list in groups.Values)
        {
            for (int i = 0; i < list.Count; i++)
                foldAssignments[list[i]] = i % k;
        }

        for (int fold = 0; fold < k; fold++)
        {
            var trainIdx = new List<int>();
            var valIdx = new List<int>();
            for (int i = 0; i < df.RowCount; i++)
            {
                if (foldAssignments[i] == fold)
                    valIdx.Add(i);
                else
                    trainIdx.Add(i);
            }

            var tIdx = trainIdx.ToArray();
            var vIdx = valIdx.ToArray();
            yield return (fold,
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(tIdx))),
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(vIdx)))
            );
        }
    }

    /// <summary>Time series split: expanding window, no shuffling.</summary>
    public static IEnumerable<(int Fold, DataFrame Train, DataFrame Val)> TimeSeriesSplit(
        this DataFrame df, int nSplits = 5)
    {
        int n = df.RowCount;
        int minTrainSize = n / (nSplits + 1);

        for (int fold = 0; fold < nSplits; fold++)
        {
            int trainEnd = minTrainSize * (fold + 1);
            int valStart = trainEnd;
            int valEnd = Math.Min(trainEnd + minTrainSize, n);

            if (valStart >= n) break;

            var trainIdx = Enumerable.Range(0, trainEnd).ToArray();
            var valIdx = Enumerable.Range(valStart, valEnd - valStart).ToArray();

            yield return (fold,
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(trainIdx))),
                new DataFrame(df.ColumnNames.Select(name => df[name].TakeRows(valIdx)))
            );
        }
    }
}
