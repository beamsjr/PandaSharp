using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public enum RankMethod
{
    Average,
    Min,
    Max,
    First,
    Dense
}

public static class RankExtensions
{
    public static Column<double> Rank<T>(this Column<T> col, RankMethod method = RankMethod.Average, bool ascending = true)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        int n = col.Length;
        var result = new double?[n];

        // Build list of (index, value) for non-null entries, then sort
        var entries = new List<(int Index, T Value)>();
        for (int i = 0; i < n; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            entries.Add((i, col.Buffer.Span[i]));
        }

        entries.Sort((a, b) =>
        {
            int cmp = Comparer<T>.Default.Compare(a.Value, b.Value);
            return ascending ? cmp : -cmp;
        });

        if (method == RankMethod.Dense)
        {
            int rank = 1;
            for (int i = 0; i < entries.Count; i++)
            {
                if (i > 0 && Comparer<T>.Default.Compare(entries[i].Value, entries[i - 1].Value) != 0)
                    rank++;
                result[entries[i].Index] = rank;
            }
        }
        else
        {
            // Group ties
            int i = 0;
            while (i < entries.Count)
            {
                int j = i;
                while (j < entries.Count && Comparer<T>.Default.Compare(entries[j].Value, entries[i].Value) == 0)
                    j++;

                // Positions i..j-1 are ties, ranks are (i+1)..(j)
                double rankValue = method switch
                {
                    RankMethod.Average => (i + 1 + j) / 2.0,
                    RankMethod.Min => i + 1,
                    RankMethod.Max => j,
                    RankMethod.First => 0, // handled per-element below
                    _ => 0
                };

                for (int k = i; k < j; k++)
                {
                    result[entries[k].Index] = method == RankMethod.First ? k + 1 : rankValue;
                }

                i = j;
            }
        }

        return Column<double>.FromNullable(col.Name + "_rank", result);
    }
}
