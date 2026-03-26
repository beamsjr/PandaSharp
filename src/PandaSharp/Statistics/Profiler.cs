using System.Runtime.CompilerServices;
using PandaSharp.Column;

namespace PandaSharp.Statistics;

/// <summary>
/// Builds a DataProfile from a DataFrame.
/// </summary>
public static class Profiler
{
    /// <summary>
    /// Generate a comprehensive profile of the DataFrame.
    /// </summary>
    public static DataProfile Profile(this DataFrame df)
    {
        var columns = new ColumnProfile[df.ColumnCount];
        int totalMissing = 0;

        for (int c = 0; c < df.ColumnCount; c++)
        {
            var col = df[df.ColumnNames[c]];
            columns[c] = ProfileColumn(col);
            totalMissing += columns[c].NullCount;
        }

        int totalCells = df.RowCount * df.ColumnCount;

        // Correlation matrix for numeric columns
        var numericCols = df.ColumnNames
            .Where(n => IsNumericType(df[n].DataType))
            .ToArray();
        double[,]? corrMatrix = null;
        if (numericCols.Length >= 2)
            corrMatrix = ComputeCorrelationMatrix(df, numericCols);

        // Duplicate rows
        int dupes = CountDuplicateRows(df);

        // Memory estimate
        long memBytes = EstimateMemory(df);

        return new DataProfile
        {
            RowCount = df.RowCount,
            ColumnCount = df.ColumnCount,
            MemoryBytes = memBytes,
            DuplicateRowCount = dupes,
            DuplicateRowPercent = df.RowCount > 0 ? dupes * 100.0 / df.RowCount : 0,
            TotalMissingValues = totalMissing,
            MissingPercent = totalCells > 0 ? totalMissing * 100.0 / totalCells : 0,
            Columns = columns,
            CorrelationMatrix = corrMatrix,
            CorrelationColumns = numericCols
        };
    }

    private static ColumnProfile ProfileColumn(IColumn col)
    {
        int nonNull = col.Length - col.NullCount;
        bool isNumeric = IsNumericType(col.DataType);
        bool isString = col.DataType == typeof(string);

        // Top values (value counts)
        var valueCounts = new Dictionary<string, int>();
        int uniqueCount = 0;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) continue;
            var key = col.GetObject(i)?.ToString() ?? "";
            valueCounts[key] = valueCounts.GetValueOrDefault(key) + 1;
        }
        uniqueCount = valueCounts.Count;
        var topValues = valueCounts
            .OrderByDescending(kv => kv.Value)
            .Take(10)
            .Select(kv => new TopValue(kv.Key, kv.Value))
            .ToArray();

        var profile = new ColumnProfile
        {
            Name = col.Name,
            TypeName = col.DataType.Name,
            IsNumeric = isNumeric,
            IsString = isString,
            NonNullCount = nonNull,
            NullCount = col.NullCount,
            NullPercent = col.Length > 0 ? col.NullCount * 100.0 / col.Length : 0,
            UniqueCount = uniqueCount,
            TopValues = topValues
        };

        if (isNumeric && nonNull > 0)
            profile = ProfileNumeric(col, profile);

        if (isString)
            profile = ProfileString(col, profile);

        return profile;
    }

    private static ColumnProfile ProfileNumeric(IColumn col, ColumnProfile profile)
    {
        // Use typed extraction to avoid boxing
        var allValues = ExtractDoubleArray(col);
        var values = new List<double>();
        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) continue;
            double v = allValues[i];
            if (double.IsNaN(v)) continue; // skip NaN values
            values.Add(v);
        }

        if (values.Count == 0) return profile;

        double mean = values.Average();
        double sumSqDiff = 0;
        double sumCubDiff = 0;
        double sumQuadDiff = 0;
        int zeros = 0;

        foreach (var v in values)
        {
            double d = v - mean;
            sumSqDiff += d * d;
            sumCubDiff += d * d * d;
            sumQuadDiff += d * d * d * d;
            if (v == 0) zeros++;
        }

        double variance = values.Count > 1 ? sumSqDiff / (values.Count - 1) : 0;
        double std = Math.Sqrt(variance);
        double skew = values.Count > 2 && std > 0
            ? (values.Count * sumCubDiff) / ((values.Count - 1.0) * (values.Count - 2.0) * std * std * std)
            : 0;
        double kurtosis = values.Count > 3 && variance > 0
            ? ((values.Count * (values.Count + 1.0)) / ((values.Count - 1.0) * (values.Count - 2.0) * (values.Count - 3.0)))
              * (sumQuadDiff / (variance * variance))
              - (3.0 * (values.Count - 1.0) * (values.Count - 1.0)) / ((values.Count - 2.0) * (values.Count - 3.0))
            : 0;

        values.Sort();
        double q25 = Percentile(values, 0.25);
        double median = Percentile(values, 0.50);
        double q75 = Percentile(values, 0.75);

        return profile with
        {
            Mean = mean,
            Std = std,
            Min = values[0],
            Max = values[^1],
            Median = median,
            Q25 = q25,
            Q75 = q75,
            Skew = skew,
            Kurtosis = kurtosis,
            ZeroCount = zeros,
            ZeroPercent = values.Count > 0 ? zeros * 100.0 / values.Count : 0
        };
    }

    private static ColumnProfile ProfileString(IColumn col, ColumnProfile profile)
    {
        int minLen = int.MaxValue, maxLen = 0;
        long totalLen = 0;
        int emptyCount = 0;
        int validCount = 0;

        for (int i = 0; i < col.Length; i++)
        {
            if (col.IsNull(i)) continue;
            var s = col.GetObject(i)?.ToString() ?? "";
            int len = s.Length;
            if (len < minLen) minLen = len;
            if (len > maxLen) maxLen = len;
            totalLen += len;
            if (len == 0) emptyCount++;
            validCount++;
        }

        if (validCount == 0) minLen = 0;

        return profile with
        {
            MinLength = minLen,
            MaxLength = maxLen,
            MeanLength = validCount > 0 ? (double)totalLen / validCount : 0,
            EmptyStringCount = emptyCount
        };
    }

    private static double Percentile(List<double> sorted, double p)
    {
        if (sorted.Count == 0) return 0;
        double idx = p * (sorted.Count - 1);
        int lo = (int)idx;
        int hi = Math.Min(lo + 1, sorted.Count - 1);
        double frac = idx - lo;
        return sorted[lo] * (1 - frac) + sorted[hi] * frac;
    }

    private static double[,] ComputeCorrelationMatrix(DataFrame df, string[] numericCols)
    {
        int n = numericCols.Length;
        int rows = df.RowCount;
        var matrix = new double[n, n];
        var colArrays = new double[n][];
        var means = new double[n];

        // Extract typed arrays — avoid GetObject() boxing
        for (int i = 0; i < n; i++)
        {
            var col = df[numericCols[i]];
            colArrays[i] = ExtractDoubleArray(col);
            double sum = 0;
            var arr = colArrays[i];
            for (int r = 0; r < rows; r++) sum += arr[r];
            means[i] = sum / rows;
        }

        for (int i = 0; i < n; i++)
        {
            matrix[i, i] = 1.0;
            for (int j = i + 1; j < n; j++)
            {
                double sumXY = 0, sumX2 = 0, sumY2 = 0;
                var ai = colArrays[i];
                var aj = colArrays[j];
                double mi = means[i], mj = means[j];
                for (int r = 0; r < rows; r++)
                {
                    double dx = ai[r] - mi;
                    double dy = aj[r] - mj;
                    sumXY += dx * dy;
                    sumX2 += dx * dx;
                    sumY2 += dy * dy;
                }
                double denom = Math.Sqrt(sumX2 * sumY2);
                double corr = denom > 0 ? sumXY / denom : 0;
                matrix[i, j] = corr;
                matrix[j, i] = corr;
            }
        }

        return matrix;
    }

    /// <summary>Extract column data as double[] without boxing — typed fast paths.</summary>
    private static double[] ExtractDoubleArray(IColumn col)
    {
        int len = col.Length;
        var result = new double[len];

        if (col is Column.Column<double> dc)
        {
            var span = dc.Buffer.Span;
            for (int i = 0; i < len; i++)
                result[i] = dc.Nulls.IsNull(i) ? 0 : span[i];
        }
        else if (col is Column.Column<int> ic)
        {
            var span = ic.Buffer.Span;
            for (int i = 0; i < len; i++)
                result[i] = ic.Nulls.IsNull(i) ? 0 : span[i];
        }
        else if (col is Column.Column<long> lc)
        {
            var span = lc.Buffer.Span;
            for (int i = 0; i < len; i++)
                result[i] = lc.Nulls.IsNull(i) ? 0 : span[i];
        }
        else if (col is Column.Column<float> fc)
        {
            var span = fc.Buffer.Span;
            for (int i = 0; i < len; i++)
                result[i] = fc.Nulls.IsNull(i) ? 0 : span[i];
        }
        else
        {
            for (int i = 0; i < len; i++)
                result[i] = col.IsNull(i) ? 0 : Convert.ToDouble(col.GetObject(i));
        }

        return result;
    }

    private static int CountDuplicateRows(DataFrame df)
    {
        if (df.RowCount <= 1) return 0;

        // Hash-based dedup: compute a combined hash per row instead of string concatenation
        var hashes = new HashSet<long>();
        int dupes = 0;
        var cols = df.ColumnNames.Select(n => df[n]).ToArray();

        for (int r = 0; r < df.RowCount; r++)
        {
            long hash = 17;
            for (int c = 0; c < cols.Length; c++)
            {
                var val = cols[c].GetObject(r);
                hash = hash * 31 + (val?.GetHashCode() ?? 0);
            }
            if (!hashes.Add(hash)) dupes++;
        }

        return dupes;
    }

    private static long EstimateMemory(DataFrame df)
    {
        long total = 0;
        foreach (var name in df.ColumnNames)
        {
            var col = df[name];
            if (col.DataType == typeof(int)) total += col.Length * 4L;
            else if (col.DataType == typeof(long)) total += col.Length * 8L;
            else if (col.DataType == typeof(double)) total += col.Length * 8L;
            else if (col.DataType == typeof(float)) total += col.Length * 4L;
            else if (col.DataType == typeof(bool)) total += col.Length;
            else total += col.Length * 40L; // rough estimate for strings
        }
        return total;
    }

    private static bool IsNumericType(Type t) =>
        t == typeof(int) || t == typeof(long) || t == typeof(double) || t == typeof(float);
}
