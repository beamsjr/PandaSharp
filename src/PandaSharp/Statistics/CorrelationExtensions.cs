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

        if (k == 0)
        {
            return new DataFrame(new List<IColumn> { new StringColumn("column", Array.Empty<string>()) });
        }

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

        // For wide matrices (many columns, few rows), use parallel matrix multiply approach.
        // FastCorrWide is NaN-aware: it replaces NaN/null with 0 in centered data and
        // adjusts per-column valid counts, so it handles NaN correctly.
        if (k > 100 && n < k)
            return FastCorrWide(cols, names, k, n);

        // Check if any column contains NaN or null; if so, fall back to NaN-aware path
        bool hasAnyNaN = false;
        for (int c = 0; c < k && !hasAnyNaN; c++)
        {
            if (cols[c].NullCount > 0) { hasAnyNaN = true; break; }
            var span = cols[c].Values;
            for (int i = 0; i < n; i++)
            {
                if (double.IsNaN(span[i]))
                {
                    hasAnyNaN = true;
                    break;
                }
            }
        }

        if (hasAnyNaN)
        {
            // NaN-aware fallback: use nullable path which handles NaN correctly
            var doubleArrays = cols.Select(c =>
            {
                var span = c.Values;
                var result = new double?[n];
                for (int i = 0; i < n; i++)
                    result[i] = (c.IsNull(i) || double.IsNaN(span[i])) ? null : span[i];
                return result;
            }).ToArray();

            var fallbackColumns = new List<IColumn>();
            fallbackColumns.Add(new StringColumn("column", names));
            for (int j = 0; j < k; j++)
            {
                var values = new double[k];
                for (int i = 0; i < k; i++)
                    values[i] = PearsonCorrelation(doubleArrays[i], doubleArrays[j]);
                fallbackColumns.Add(new Column<double>(names[j], values));
            }
            return new DataFrame(fallbackColumns);
        }

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
        var validCounts = new int[k];
        // Track per-row validity as a bitmask-like bool array for pairwise count computation
        // valid[i * k + c] = true if row i, col c is valid
        var valid = new bool[n * k];

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
                valid[i * k + c] = true;
            }
            validCounts[c] = cnt;
            double m = cnt > 0 ? sum / cnt : 0;
            // Center: replace null/NaN with 0 (neutral for dot product)
            double ss = 0;
            for (int i = 0; i < n; i++)
            {
                if (!valid[i * k + c])
                {
                    X[i * k + c] = 0; // neutral for dot product
                }
                else
                {
                    double v = span[i];
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

        // Check if any column has missing values; if not, skip pairwise count computation
        bool anyMissing = false;
        for (int c = 0; c < k; c++)
            if (validCounts[c] < n) { anyMissing = true; break; }

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
                if (si == 0 || sj == 0)
                {
                    values[i] = double.NaN;
                }
                else
                {
                    // Use per-column valid count as denominator normalization
                    // For no-missing case, this equals (n-1) for all pairs
                    int denom = anyMissing ? Math.Min(validCounts[ii], validCounts[jj]) - 1 : n - 1;
                    values[i] = denom > 0 ? C[ii * k + jj] / (denom * si * sj) : double.NaN;
                }
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

    /// <summary>
    /// Returns a Spearman rank correlation matrix DataFrame for all numeric columns.
    /// Rank-transforms each column (average rank for ties), then applies Pearson on ranks.
    /// O(n log n) per column for ranking via IntroSort.
    /// </summary>
    public static DataFrame CorrSpearman(this DataFrame df)
    {
        var numericCols = GetNumericColumns(df);
        var names = numericCols.Select(c => c.Name).ToArray();
        int k = names.Length;
        if (k == 0) return new DataFrame(new List<IColumn> { new StringColumn("column", Array.Empty<string>()) });

        int n = numericCols[0].Length;
        if (n <= 1)
        {
            // Single row or empty: return NaN matrix
            return BuildNaNMatrix(names, k);
        }

        // Rank-transform each column into double[], handling nulls and ties with average rank
        var rankedData = new double[k][];
        for (int c = 0; c < k; c++)
            rankedData[c] = RankTransform(numericCols[c]);

        // Build Column<double> from ranked data and reuse FastCorr
        var rankedCols = new Column<double>[k];
        for (int c = 0; c < k; c++)
            rankedCols[c] = new Column<double>(names[c], rankedData[c]);

        return FastCorr(rankedCols, names);
    }

    /// <summary>
    /// Returns a Kendall tau-b correlation matrix DataFrame for all numeric columns.
    /// Uses Knight's O(n log n) merge-sort algorithm for inversion counting.
    /// Column pairs are computed in parallel.
    /// </summary>
    public static DataFrame CorrKendall(this DataFrame df)
    {
        var numericCols = GetNumericColumns(df);
        var names = numericCols.Select(c => c.Name).ToArray();
        int k = names.Length;
        if (k == 0) return new DataFrame(new List<IColumn> { new StringColumn("column", Array.Empty<string>()) });

        int n = numericCols[0].Length;
        if (n <= 1)
            return BuildNaNMatrix(names, k);

        // Convert all columns to double[], replacing nulls with NaN
        var data = new double[k][];
        for (int c = 0; c < k; c++)
            data[c] = ToDoubleArrayNaN(numericCols[c]);

        var result = new double[k, k];
        // Diagonal is always 1.0
        for (int i = 0; i < k; i++)
            result[i, i] = 1.0;

        // Compute upper triangle in parallel
        Parallel.For(0, k * (k - 1) / 2, idx =>
        {
            // Map linear index to (ci, cj) pair where ci < cj
            int ci = 0, cj = 0;
            int remaining = idx;
            for (ci = 0; ci < k - 1; ci++)
            {
                int pairsInRow = k - 1 - ci;
                if (remaining < pairsInRow)
                {
                    cj = ci + 1 + remaining;
                    break;
                }
                remaining -= pairsInRow;
            }

            double tau = KendallTauB(data[ci], data[cj], n);
            result[ci, cj] = tau;
            result[cj, ci] = tau;
        });

        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));
        for (int j = 0; j < k; j++)
        {
            var values = new double[k];
            for (int i = 0; i < k; i++)
                values[i] = result[i, j];
            columns.Add(new Column<double>(names[j], values));
        }

        return new DataFrame(columns);
    }

    /// <summary>
    /// Rank-transform a column's values using average rank for ties.
    /// Nulls get NaN rank. Uses IntroSort on index array: O(n log n).
    /// </summary>
    private static double[] RankTransform(IColumn col)
    {
        int n = col.Length;
        var ranks = new double[n];

        // Get values as doubles, track non-null indices
        var indices = new List<int>(n);
        var values = new double[n];

        if (col is Column<double> dc)
        {
            var span = dc.Values;
            for (int i = 0; i < n; i++)
            {
                if (dc.IsNull(i) || double.IsNaN(span[i]))
                {
                    ranks[i] = double.NaN;
                    continue;
                }
                values[i] = span[i];
                indices.Add(i);
            }
        }
        else if (col is Column<int> ic)
        {
            var span = ic.Values;
            for (int i = 0; i < n; i++)
            {
                if (ic.IsNull(i))
                {
                    ranks[i] = double.NaN;
                    continue;
                }
                values[i] = span[i];
                indices.Add(i);
            }
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                if (col.IsNull(i))
                {
                    ranks[i] = double.NaN;
                    continue;
                }
                var obj = col.GetObject(i);
                if (obj is null)
                {
                    ranks[i] = double.NaN;
                    continue;
                }
                values[i] = Convert.ToDouble(obj);
                indices.Add(i);
            }
        }

        // Sort indices by their corresponding values (IntroSort via Array.Sort)
        var sortedIndices = indices.ToArray();
        Array.Sort(sortedIndices, (a, b) => values[a].CompareTo(values[b]));

        // Assign average ranks, handling ties with single scan
        int m = sortedIndices.Length;
        int pos = 0;
        while (pos < m)
        {
            int tieStart = pos;
            double tieValue = values[sortedIndices[pos]];
            while (pos < m && values[sortedIndices[pos]] == tieValue)
                pos++;
            // Positions tieStart..pos-1 are tied; average rank = (tieStart+1 + pos) / 2
            double avgRank = (tieStart + 1 + pos) / 2.0;
            for (int t = tieStart; t < pos; t++)
                ranks[sortedIndices[t]] = avgRank;
        }

        return ranks;
    }

    /// <summary>
    /// Convert column to double[] with NaN for nulls.
    /// </summary>
    private static double[] ToDoubleArrayNaN(IColumn col)
    {
        int n = col.Length;
        var result = new double[n];

        if (col is Column<double> dc)
        {
            var span = dc.Values;
            for (int i = 0; i < n; i++)
                result[i] = dc.IsNull(i) ? double.NaN : span[i];
        }
        else if (col is Column<int> ic)
        {
            var span = ic.Values;
            for (int i = 0; i < n; i++)
                result[i] = ic.IsNull(i) ? double.NaN : span[i];
        }
        else if (col is Column<long> lc)
        {
            var span = lc.Values;
            for (int i = 0; i < n; i++)
                result[i] = lc.IsNull(i) ? double.NaN : span[i];
        }
        else if (col is Column<float> fc)
        {
            var span = fc.Values;
            for (int i = 0; i < n; i++)
                result[i] = fc.IsNull(i) ? double.NaN : span[i];
        }
        else
        {
            for (int i = 0; i < n; i++)
            {
                var obj = col.IsNull(i) ? null : col.GetObject(i);
                result[i] = obj is null ? double.NaN : Convert.ToDouble(obj);
            }
        }

        return result;
    }

    /// <summary>
    /// Compute Kendall tau-b for two double arrays using Knight's O(n log n) merge-sort algorithm.
    /// Handles ties via tau-b correction: tau_b = (C - D) / sqrt((C+D+T_x)*(C+D+T_y))
    /// where T_x = ties in x only, T_y = ties in y only.
    /// </summary>
    private static double KendallTauB(double[] x, double[] y, int n)
    {
        // Filter out pairs where either value is NaN
        var validIndices = new List<int>(n);
        for (int i = 0; i < n; i++)
        {
            if (!double.IsNaN(x[i]) && !double.IsNaN(y[i]))
                validIndices.Add(i);
        }

        int m = validIndices.Count;
        if (m <= 1) return double.NaN;

        // Create pairs sorted by (x, y)
        var pairs = new (double X, double Y)[m];
        for (int i = 0; i < m; i++)
        {
            int idx = validIndices[i];
            pairs[i] = (x[idx], y[idx]);
        }

        // Sort by x, then by y for ties in x
        Array.Sort(pairs, (a, b) =>
        {
            int cmp = a.X.CompareTo(b.X);
            return cmp != 0 ? cmp : a.Y.CompareTo(b.Y);
        });

        // Count ties in x (T_x) and joint ties (T_xy)
        long tiesX = 0;
        long tiesXY = 0;
        int i2 = 0;
        while (i2 < m)
        {
            int j = i2;
            while (j < m && pairs[j].X == pairs[i2].X)
                j++;
            long groupSize = j - i2;
            if (groupSize > 1)
                tiesX += groupSize * (groupSize - 1) / 2;
            // Count joint ties within this x-group
            int k2 = i2;
            while (k2 < j)
            {
                int l = k2;
                while (l < j && pairs[l].Y == pairs[k2].Y)
                    l++;
                long subGroup = l - k2;
                if (subGroup > 1)
                    tiesXY += subGroup * (subGroup - 1) / 2;
                k2 = l;
            }
            i2 = j;
        }

        // Extract y-values in x-sorted order, then use merge sort to count inversions
        var yArr = new double[m];
        for (int i = 0; i < m; i++)
            yArr[i] = pairs[i].Y;

        // Count ties in y
        long tiesY = 0;
        {
            var ySorted = (double[])yArr.Clone();
            Array.Sort(ySorted);
            int p = 0;
            while (p < m)
            {
                int q = p;
                while (q < m && ySorted[q] == ySorted[p])
                    q++;
                long gs = q - p;
                if (gs > 1)
                    tiesY += gs * (gs - 1) / 2;
                p = q;
            }
        }

        // Merge-sort inversion count on y-values gives number of discordant pairs
        // (among pairs that are not tied in x)
        long swaps = MergeSortCount(yArr, new double[m], 0, m - 1);

        long totalPairs = (long)m * (m - 1) / 2;
        long concordant = totalPairs - tiesX - tiesY + tiesXY - swaps;
        // swaps = discordant pairs (not counting ties)
        // C + D + T_x + T_y - T_xy = totalPairs
        // D = swaps, so C = totalPairs - T_x - T_y + T_xy - swaps... let's re-derive:
        // Actually with merge sort on sorted-by-x data:
        // swaps counts inversions in y = number of discordant pairs (where x_i < x_j but y_i > y_j)
        // But pairs tied in x are not discordant. Since we sorted by (x,y), ties in x with different y
        // are already in order, so they don't contribute inversions. Ties in both x and y also contribute 0.
        // So: discordant = swaps
        // concordant = totalPairs - discordant - tiesX - tiesY + tiesXY
        long discordant = swaps;
        concordant = totalPairs - discordant - tiesX - tiesY + tiesXY;

        double denominator = Math.Sqrt((double)(totalPairs - tiesX) * (totalPairs - tiesY));
        if (denominator == 0) return double.NaN;

        return (concordant - discordant) / denominator;
    }

    /// <summary>
    /// Merge sort that counts the number of inversions (swaps). O(n log n).
    /// </summary>
    private static long MergeSortCount(double[] arr, double[] temp, int left, int right)
    {
        if (left >= right) return 0;

        int mid = left + (right - left) / 2;
        long count = 0;
        count += MergeSortCount(arr, temp, left, mid);
        count += MergeSortCount(arr, temp, mid + 1, right);
        count += Merge(arr, temp, left, mid, right);
        return count;
    }

    private static long Merge(double[] arr, double[] temp, int left, int mid, int right)
    {
        for (int idx = left; idx <= right; idx++)
            temp[idx] = arr[idx];

        int i = left, j = mid + 1, k = left;
        long swaps = 0;

        while (i <= mid && j <= right)
        {
            if (temp[i] <= temp[j])
            {
                arr[k++] = temp[i++];
            }
            else
            {
                arr[k++] = temp[j++];
                swaps += (mid - i + 1); // all remaining in left half are inversions
            }
        }

        while (i <= mid) arr[k++] = temp[i++];
        while (j <= right) arr[k++] = temp[j++];

        return swaps;
    }

    private static DataFrame BuildNaNMatrix(string[] names, int k)
    {
        var columns = new List<IColumn>();
        columns.Add(new StringColumn("column", names));
        for (int j = 0; j < k; j++)
        {
            var values = new double[k];
            Array.Fill(values, double.NaN);
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
