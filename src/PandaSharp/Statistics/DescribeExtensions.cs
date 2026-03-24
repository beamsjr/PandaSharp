using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class DescribeExtensions
{
    /// <summary>
    /// Returns a summary DataFrame with count, mean, std, min, 25%, 50%, 75%, max per numeric column.
    /// </summary>
    public static DataFrame Describe(this DataFrame df)
    {
        var statNames = new string[] { "count", "mean", "std", "min", "25%", "50%", "75%", "max" };
        var columns = new List<IColumn>();

        // First column: stat names
        columns.Add(new StringColumn("stat", statNames));

        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            var values = TryDescribeColumn(col);
            if (values is not null)
                columns.Add(new Column<double>(name, values));
        }

        return new DataFrame(columns);
    }

    private static double[]? TryDescribeColumn(IColumn col)
    {
        return col switch
        {
            Column<int> c => DescribeNumeric(c),
            Column<long> c => DescribeNumeric(c),
            Column<float> c => DescribeNumeric(c),
            Column<double> c => DescribeNumeric(c),
            _ => null
        };
    }

    private static double[] DescribeNumeric<T>(Column<T> col) where T : struct, INumber<T>, IComparisonOperators<T, T, bool>
    {
        return
        [
            col.Count(),
            col.Mean() ?? double.NaN,
            col.Std() ?? double.NaN,
            col.Min() is T min ? double.CreateChecked(min) : double.NaN,
            col.Quantile(0.25) ?? double.NaN,
            col.Quantile(0.50) ?? double.NaN,
            col.Quantile(0.75) ?? double.NaN,
            col.Max() is T max ? double.CreateChecked(max) : double.NaN,
        ];
    }
}
