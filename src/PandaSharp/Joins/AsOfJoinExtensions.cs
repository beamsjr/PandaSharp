using PandaSharp.Column;

namespace PandaSharp.Joins;

public enum AsOfDirection
{
    /// <summary>Find the nearest match that is less than or equal to the left key.</summary>
    Backward,
    /// <summary>Find the nearest match that is greater than or equal to the left key.</summary>
    Forward,
    /// <summary>Find the absolute nearest match.</summary>
    Nearest
}

public static class AsOfJoinExtensions
{
    /// <summary>
    /// Merge-as-of: for each row in left, find the nearest matching row in right
    /// based on the "on" column (typically a timestamp). Right must be sorted on the "on" column.
    /// </summary>
    public static DataFrame JoinAsOf(this DataFrame left, DataFrame right,
        string on, AsOfDirection direction = AsOfDirection.Backward, string? by = null)
    {
        var leftCol = left[on];
        var rightCol = right[on];

        // Build right-side values as comparable
        var rightValues = new IComparable?[right.RowCount];
        for (int i = 0; i < right.RowCount; i++)
            rightValues[i] = rightCol.GetObject(i) as IComparable;

        // For each left row, find the best match in right
        var rightMatches = new int?[left.RowCount];

        for (int l = 0; l < left.RowCount; l++)
        {
            var leftVal = leftCol.GetObject(l) as IComparable;
            if (leftVal is null) continue;

            // If 'by' is specified, only match within same by-group
            int? bestIdx = null;

            for (int r = 0; r < right.RowCount; r++)
            {
                if (rightValues[r] is null) continue;

                if (by is not null)
                {
                    var leftBy = left[by].GetObject(l);
                    var rightBy = right[by].GetObject(r);
                    if (!Equals(leftBy, rightBy)) continue;
                }

                int cmp = leftVal.CompareTo(rightValues[r]);

                bool valid = direction switch
                {
                    AsOfDirection.Backward => cmp >= 0,  // right <= left
                    AsOfDirection.Forward => cmp <= 0,    // right >= left
                    AsOfDirection.Nearest => true,
                    _ => false
                };

                if (!valid) continue;

                if (bestIdx is null)
                {
                    bestIdx = r;
                }
                else
                {
                    var bestVal = rightValues[bestIdx.Value]!;
                    bool isBetter = direction switch
                    {
                        AsOfDirection.Backward => rightValues[r]!.CompareTo(bestVal) > 0, // closer = larger
                        AsOfDirection.Forward => rightValues[r]!.CompareTo(bestVal) < 0,  // closer = smaller
                        AsOfDirection.Nearest => Math.Abs(Convert.ToDouble(leftVal) - Convert.ToDouble(rightValues[r]))
                            < Math.Abs(Convert.ToDouble(leftVal) - Convert.ToDouble(bestVal)),
                        _ => false
                    };
                    if (isBetter) bestIdx = r;
                }
            }

            rightMatches[l] = bestIdx;
        }

        // Build result: all left columns + non-key right columns
        var columns = new List<IColumn>();
        foreach (var name in left.ColumnNames)
            columns.Add(left[name]);

        foreach (var name in right.ColumnNames)
        {
            if (name == on) continue;
            if (by is not null && name == by) continue;

            var col = right[name];
            var values = new object?[left.RowCount];
            for (int i = 0; i < left.RowCount; i++)
                values[i] = rightMatches[i] is { } ri ? col.GetObject(ri) : null;
            columns.Add(BuildColumn(name, col.DataType, values));
        }

        return new DataFrame(columns);
    }

    private static IColumn BuildColumn(string name, Type type, object?[] values)
    {
        if (type == typeof(string))
            return new StringColumn(name, values.Select(v => v as string).ToArray());
        if (type == typeof(int)) return BuildTyped<int>(name, values);
        if (type == typeof(long)) return BuildTyped<long>(name, values);
        if (type == typeof(double)) return BuildTyped<double>(name, values);
        if (type == typeof(float)) return BuildTyped<float>(name, values);
        if (type == typeof(bool)) return BuildTyped<bool>(name, values);
        if (type == typeof(DateTime)) return BuildTyped<DateTime>(name, values);
        return new StringColumn(name, values.Select(v => v?.ToString()).ToArray());
    }

    private static Column<T> BuildTyped<T>(string name, object?[] values) where T : struct
    {
        var typed = new T?[values.Length];
        for (int i = 0; i < values.Length; i++)
            typed[i] = values[i] is null ? null : (T)Convert.ChangeType(values[i]!, typeof(T));
        return Column<T>.FromNullable(name, typed);
    }
}
