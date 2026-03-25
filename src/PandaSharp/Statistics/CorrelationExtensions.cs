using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

public static class CorrelationExtensions
{
    /// <summary>
    /// Returns a correlation matrix DataFrame for all numeric columns.
    /// Uses a fast two-pass algorithm when all columns are non-null doubles.
    /// </summary>
    public static DataFrame Corr(this DataFrame df)
    {
        var numericCols = GetNumericColumns(df);
        var names = numericCols.Select(c => c.Name).ToArray();
        int k = names.Length;

        // Fast path: all Column<double> (with or without nulls — treat nulls as NaN)
        if (numericCols.All(c => c is Column<double>))
        {
            return FastCorr(numericCols.Cast<Column<double>>().ToArray(), names);
        }

        // Fallback: nullable-aware
        var doubleArrays = numericCols.Select(ToDoubleArray).ToArray();
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));

        for (int j = 0; j < k; j++)
        {
            var values = new double[k];
            for (int i = 0; i < k; i++)
                values[i] = PearsonCorrelation(doubleArrays[i], doubleArrays[j]);
            columns.Add(new Column<double>(names[j], values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Fast correlation matrix for non-null double columns.
    /// Pass 1: compute means. Pass 2: compute covariance matrix + variances.
    /// No boxing, no nullable overhead, SIMD-friendly inner loop.
    /// </summary>
    private static DataFrame FastCorr(Column<double>[] cols, string[] names)
    {
        int k = cols.Length;
        int n = cols[0].Length;

        // For wide matrices (many columns, few rows), use parallel matrix multiply approach
        if (k > 100 && n < k)
            return FastCorrWide(cols, names, k, n);

        // Standard pairwise approach for narrow matrices (few columns, many rows)
        var data = new double[k][];
        var means = new double[k];
        for (int c = 0; c < k; c++)
        {
            data[c] = cols[c].Values.ToArray();
            double sum = 0;
            var d = data[c];
            for (int i = 0; i < n; i++) sum += d[i];
            means[c] = sum / n;
        }

        var cov = new double[k, k];
        int vectorSize = Vector<double>.Count;

        for (int ci = 0; ci < k; ci++)
        {
            var di = data[ci];
            double mi = means[ci];
            var vmi = new Vector<double>(mi);

            for (int cj = ci; cj < k; cj++)
            {
                var dj = data[cj];
                double mj = means[cj];
                var vmj = new Vector<double>(mj);

                var vsum = Vector<double>.Zero;
                int i = 0;
                int limit = n - vectorSize;
                for (; i <= limit; i += vectorSize)
                {
                    var vi = new Vector<double>(di, i) - vmi;
                    var vj = new Vector<double>(dj, i) - vmj;
                    vsum += vi * vj;
                }

                double sum = 0;
                for (int s = 0; s < vectorSize; s++) sum += vsum[s];
                for (; i < n; i++) sum += (di[i] - mi) * (dj[i] - mj);

                cov[ci, cj] = sum / (n - 1);
                cov[cj, ci] = cov[ci, cj];
            }
        }

        var stds = new double[k];
        for (int c = 0; c < k; c++)
            stds[c] = Math.Sqrt(cov[c, c]);

        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));
        for (int j = 0; j < k; j++)
        {
            var values = new double[k];
            for (int i = 0; i < k; i++)
            {
                if (stds[i] == 0 || stds[j] == 0)
                    values[i] = double.NaN;
                else
                    values[i] = cov[i, j] / (stds[i] * stds[j]);
            }
            columns.Add(new Column<double>(names[j], values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Fast correlation for wide matrices (k >> n): uses row-major centered matrix
    /// and parallel blocked matrix multiply for X^T × X covariance computation.
    /// </summary>
    private static DataFrame FastCorrWide(Column<double>[] cols, string[] names, int k, int n)
    {
        // Build centered row-major matrix X[n, k]
        var X = new double[n * k];
        var stds = new double[k];

        // Parallel: compute means (excluding nulls/NaN), center data, compute stds
        Parallel.For(0, k, c =>
        {
            var span = cols[c].Values;
            bool hasNulls = cols[c].NullCount > 0;
            // Compute mean excluding nulls/NaN
            double sum = 0; int cnt = 0;
            for (int i = 0; i < n; i++)
            {
                double v = span[i];
                if (hasNulls && cols[c].IsNull(i)) continue;
                if (double.IsNaN(v)) continue;
                sum += v; cnt++;
            }
            double m = cnt > 0 ? sum / cnt : 0;
            // Center: replace null/NaN with 0 (neutral for dot product)
            double ss = 0;
            for (int i = 0; i < n; i++)
            {
                double v = span[i];
                if ((hasNulls && cols[c].IsNull(i)) || double.IsNaN(v))
                {
                    X[i * k + c] = 0; // neutral for dot product
                }
                else
                {
                    double cv = v - m;
                    X[i * k + c] = cv;
                    ss += cv * cv;
                }
            }
            stds[c] = cnt > 1 ? Math.Sqrt(ss / (cnt - 1)) : 0;
        });

        // Compute gram matrix C = X^T * X using BLAS dsyrk
        var C = new double[k * k];
        Native.NativeOps.GramMatrixUpper(X, n, k, C);

        // Convert to correlation + build DataFrame
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));
        for (int j = 0; j < k; j++)
        {
            var values = new double[k];
            for (int i = 0; i < k; i++)
            {
                int ii = Math.Min(i, j), jj = Math.Max(i, j);
                double si = stds[ii], sj = stds[jj];
                values[i] = (si == 0 || sj == 0) ? double.NaN : C[ii * k + jj] / ((n - 1) * si * sj);
            }
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
        // Typed fast path: avoid boxing
        if (col is Column<double> dc)
        {
            var span = dc.Values;
            var result = new double?[col.Length];
            for (int i = 0; i < col.Length; i++)
                result[i] = dc.IsNull(i) ? null : span[i];
            return result;
        }
        if (col is Column<int> ic)
        {
            var span = ic.Values;
            var result = new double?[col.Length];
            for (int i = 0; i < col.Length; i++)
                result[i] = ic.IsNull(i) ? null : (double)span[i];
            return result;
        }

        var fallback = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) { fallback[i] = null; continue; }
            fallback[i] = Convert.ToDouble(col.GetObject(i));
        }
        return fallback;
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
