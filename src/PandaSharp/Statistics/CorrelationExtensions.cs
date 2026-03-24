using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class CorrelationExtensions
{
    /// <summary>
    /// Returns a correlation matrix DataFrame for all numeric columns.
    /// </summary>
    public static DataFrame Corr(this DataFrame df)
    {
        var numericCols = GetNumericColumns(df);
        var names = numericCols.Select(c => c.Name).ToArray();
        var doubleArrays = numericCols.Select(ToDoubleArray).ToArray();

        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));

        for (int j = 0; j < names.Length; j++)
        {
            var values = new double[names.Length];
            for (int i = 0; i < names.Length; i++)
                values[i] = PearsonCorrelation(doubleArrays[i], doubleArrays[j]);
            columns.Add(new Column<double>(names[j], values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Returns a covariance matrix DataFrame for all numeric columns.
    /// </summary>
    public static DataFrame Cov(this DataFrame df, int ddof = 1)
    {
        var numericCols = GetNumericColumns(df);
        var names = numericCols.Select(c => c.Name).ToArray();
        var doubleArrays = numericCols.Select(ToDoubleArray).ToArray();

        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));

        for (int j = 0; j < names.Length; j++)
        {
            var values = new double[names.Length];
            for (int i = 0; i < names.Length; i++)
                values[i] = Covariance(doubleArrays[i], doubleArrays[j], ddof);
            columns.Add(new Column<double>(names[j], values));
        }

        return new DataFrame(columns);
    }

    private static List<IColumn> GetNumericColumns(DataFrame df)
    {
        var result = new List<IColumn>();
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            if (col.DataType == typeof(int) || col.DataType == typeof(long) ||
                col.DataType == typeof(float) || col.DataType == typeof(double))
                result.Add(col);
        }
        return result;
    }

    private static double?[] ToDoubleArray(IColumn col)
    {
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) { result[i] = null; continue; }
            result[i] = Convert.ToDouble(col.GetObject(i));
        }
        return result;
    }

    private static double PearsonCorrelation(double?[] x, double?[] y)
    {
        double cov = Covariance(x, y, 1);
        double stdX = StdDev(x);
        double stdY = StdDev(y);
        if (stdX == 0 || stdY == 0) return double.NaN;
        return cov / (stdX * stdY);
    }

    private static double Covariance(double?[] x, double?[] y, int ddof)
    {
        int n = 0;
        double sumX = 0, sumY = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i].HasValue && y[i].HasValue) { sumX += x[i]!.Value; sumY += y[i]!.Value; n++; }
        }
        if (n <= ddof) return double.NaN;
        double meanX = sumX / n, meanY = sumY / n;
        double cov = 0;
        for (int i = 0; i < x.Length; i++)
        {
            if (x[i].HasValue && y[i].HasValue)
                cov += (x[i]!.Value - meanX) * (y[i]!.Value - meanY);
        }
        return cov / (n - ddof);
    }

    private static double StdDev(double?[] vals)
    {
        int n = 0;
        double sum = 0;
        for (int i = 0; i < vals.Length; i++)
            if (vals[i].HasValue) { sum += vals[i]!.Value; n++; }
        if (n <= 1) return 0;
        double mean = sum / n;
        double sumSq = 0;
        for (int i = 0; i < vals.Length; i++)
            if (vals[i].HasValue) { double d = vals[i]!.Value - mean; sumSq += d * d; }
        return Math.Sqrt(sumSq / (n - 1));
    }
}
