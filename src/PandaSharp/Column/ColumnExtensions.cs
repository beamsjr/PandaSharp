using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PandaSharp.Column;

/// <summary>
/// Comparison and aggregate extension methods for typed columns.
/// These are extensions because C# doesn't allow further constraining a class's
/// type parameter T in non-generic methods.
/// </summary>
public static class ColumnExtensions
{
    // -- Comparisons --

    public static bool[] Gt<T>(this Column<T> col, T value)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] > value;
        return result;
    }

    public static bool[] Gte<T>(this Column<T> col, T value)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] >= value;
        return result;
    }

    public static bool[] Lt<T>(this Column<T> col, T value)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] < value;
        return result;
    }

    public static bool[] Lte<T>(this Column<T> col, T value)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] <= value;
        return result;
    }

    public static bool[] Eq<T>(this Column<T> col, T value)
        where T : struct, IEqualityOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] == value;
        return result;
    }

    /// <summary>
    /// Returns a mask where values are between lower and upper (inclusive).
    /// </summary>
    public static bool[] Between<T>(this Column<T> col, T lower, T upper)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && span[i] >= lower && span[i] <= upper;
        return result;
    }

    /// <summary>
    /// Returns a mask where value is in the given set.
    /// </summary>
    public static bool[] IsIn<T>(this Column<T> col, params T[] values)
        where T : struct, IEqualityOperators<T, T, bool>
    {
        var set = new HashSet<T>(values);
        var span = col.Buffer.Span;
        var result = new bool[col.Length];
        for (int i = 0; i < span.Length; i++)
            result[i] = !col.Nulls.IsNull(i) && set.Contains(span[i]);
        return result;
    }

    /// <summary>
    /// Returns a mask where string value is in the given set.
    /// </summary>
    public static bool[] IsIn(this StringColumn col, params string[] values)
    {
        var set = new HashSet<string>(values);
        var result = new bool[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col[i] is not null && set.Contains(col[i]!);
        return result;
    }

    /// <summary>
    /// Clamp values to a range. Values below lower become lower, above upper become upper.
    /// </summary>
    public static Column<T> Clip<T>(this Column<T> col, T lower, T upper)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        var span = col.Buffer.Span;
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            var val = span[i];
            if (val < lower) result[i] = lower;
            else if (val > upper) result[i] = upper;
            else result[i] = val;
        }
        return Column<T>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Replace values matching a condition with a new value.
    /// </summary>
    public static Column<T> Where<T>(this Column<T> col, Func<T, bool> condition, T replacement)
        where T : struct
    {
        var span = col.Buffer.Span;
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            result[i] = condition(span[i]) ? replacement : span[i];
        }
        return Column<T>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Map each value through a function, producing a new column.
    /// </summary>
    public static Column<TResult> Map<T, TResult>(this Column<T> col, Func<T, TResult> func)
        where T : struct
        where TResult : struct
    {
        var span = col.Buffer.Span;
        var result = new TResult?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : func(span[i]);
        return Column<TResult>.FromNullable(col.Name, result);
    }

    // -- Zip --

    /// <summary>
    /// Combine two columns element-wise using a function.
    /// Usage: col1.Zip(col2, (a, b) => a + b, "Sum")
    /// </summary>
    public static Column<TResult> Zip<T1, T2, TResult>(
        this Column<T1> left, Column<T2> right, Func<T1, T2, TResult> func, string? name = null)
        where T1 : struct where T2 : struct where TResult : struct
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");
        var result = new TResult?[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            if (left.Nulls.IsNull(i) || right.Nulls.IsNull(i))
                result[i] = null;
            else
                result[i] = func(left.Buffer.Span[i], right.Buffer.Span[i]);
        }
        return Column<TResult>.FromNullable(name ?? left.Name, result);
    }

    // -- Replace --

    /// <summary>
    /// Replace specific values using a mapping dictionary.
    /// Usage: col.Replace(new Dictionary&lt;int,int&gt; { [1] = 10, [2] = 20 })
    /// </summary>
    public static Column<T> Replace<T>(this Column<T> col, Dictionary<T, T> mapping)
        where T : struct
    {
        var span = col.Buffer.Span;
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            result[i] = mapping.TryGetValue(span[i], out var replacement) ? replacement : span[i];
        }
        return Column<T>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Replace string values using a mapping dictionary.
    /// </summary>
    public static StringColumn Replace(this StringColumn col, Dictionary<string, string> mapping)
    {
        var result = new string?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col[i] is null) { result[i] = null; continue; }
            result[i] = mapping.TryGetValue(col[i]!, out var replacement) ? replacement : col[i];
        }
        return new StringColumn(col.Name, result);
    }

    // -- ArgMin / ArgMax --

    /// <summary>
    /// Returns the index of the minimum non-null value.
    /// </summary>
    public static int? ArgMin<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        int? bestIdx = null;
        T? bestVal = null;
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            if (bestVal is null || span[i] < bestVal.Value)
            {
                bestVal = span[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    /// <summary>
    /// Returns the index of the maximum non-null value.
    /// </summary>
    public static int? ArgMax<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        int? bestIdx = null;
        T? bestVal = null;
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            if (bestVal is null || span[i] > bestVal.Value)
            {
                bestVal = span[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // -- Normalize --

    /// <summary>
    /// Min-max normalization: scales values to [0, 1].
    /// </summary>
    public static Column<double> NormalizeMinMax<T>(this Column<T> col)
        where T : struct, INumber<T>, IComparisonOperators<T, T, bool>
    {
        var min = col.Min();
        var max = col.Max();
        if (min is null || max is null) return Column<double>.FromNullable(col.Name, new double?[col.Length]);
        double dMin = double.CreateChecked(min.Value);
        double dMax = double.CreateChecked(max.Value);
        double range = dMax - dMin;

        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            result[i] = range == 0 ? 0.5 : (double.CreateChecked(span[i]) - dMin) / range;
        }
        return Column<double>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Z-score normalization: (x - mean) / std.
    /// </summary>
    public static Column<double> NormalizeZScore<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var mean = col.Mean();
        var std = col.Std();
        if (mean is null || std is null || std == 0)
            return Column<double>.FromNullable(col.Name, new double?[col.Length]);

        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            result[i] = (double.CreateChecked(span[i]) - mean.Value) / std.Value;
        }
        return Column<double>.FromNullable(col.Name, result);
    }

    // -- Bool aggregates --

    /// <summary>Returns true if any non-null value is true.</summary>
    public static bool Any(this Column<bool> col)
    {
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
            if (!col.Nulls.IsNull(i) && span[i]) return true;
        return false;
    }

    /// <summary>Returns true if all non-null values are true.</summary>
    public static bool All(this Column<bool> col)
    {
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
            if (!col.Nulls.IsNull(i) && !span[i]) return false;
        return true;
    }

    /// <summary>Count number of true values.</summary>
    public static int SumTrue(this Column<bool> col)
    {
        int count = 0;
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
            if (!col.Nulls.IsNull(i) && span[i]) count++;
        return count;
    }

    // -- Cumcount --

    /// <summary>
    /// Returns a DataFrame of unique values and their counts, sorted by count descending.
    /// </summary>
    public static DataFrame ValueCounts<T>(this Column<T> col) where T : struct
    {
        var counts = new Dictionary<T, int>();
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            var val = span[i];
            counts[val] = counts.GetValueOrDefault(val) + 1;
        }
        var sorted = counts.OrderByDescending(kv => kv.Value).ToList();
        return new DataFrame(
            new Column<T>(col.Name, sorted.Select(kv => kv.Key).ToArray()),
            new Column<int>("count", sorted.Select(kv => kv.Value).ToArray())
        );
    }

    /// <summary>
    /// Returns the cumulative count of non-null values seen so far (0-based).
    /// </summary>
    public static Column<int> Cumcount<T>(this Column<T> col) where T : struct
    {
        var result = new int?[col.Length];
        int count = 0;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }
            result[i] = count;
            count++;
        }
        return Column<int>.FromNullable(col.Name, result);
    }

    // -- Unique --

    /// <summary>
    /// Return a column of distinct non-null values.
    /// </summary>
    public static Column<T> Unique<T>(this Column<T> col)
        where T : struct
    {
        var seen = new HashSet<T>();
        var span = col.Buffer.Span;
        for (int i = 0; i < col.Length; i++)
        {
            if (!col.Nulls.IsNull(i))
                seen.Add(span[i]);
        }
        return new Column<T>(col.Name, seen.ToArray());
    }

    /// <summary>
    /// Return a StringColumn of distinct non-null values.
    /// </summary>
    public static StringColumn Unique(this StringColumn col)
    {
        var seen = new HashSet<string>();
        for (int i = 0; i < col.Length; i++)
        {
            if (col[i] is { } s)
                seen.Add(s);
        }
        return new StringColumn(col.Name, seen.ToArray());
    }

    // -- Shift (lag/lead) --

    /// <summary>
    /// Shift values by n positions. Positive n shifts down (lag), negative shifts up (lead).
    /// Vacated positions are filled with null.
    /// </summary>
    public static Column<T> Shift<T>(this Column<T> col, int periods = 1) where T : struct
    {
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            int srcIdx = i - periods;
            if (srcIdx >= 0 && srcIdx < col.Length)
                result[i] = col[srcIdx];
            // else stays null
        }
        return Column<T>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Shift string column by n positions.
    /// </summary>
    public static StringColumn Shift(this StringColumn col, int periods = 1)
    {
        var result = new string?[col.Length];
        for (int i = 0; i < col.Length; i++)
        {
            int srcIdx = i - periods;
            if (srcIdx >= 0 && srcIdx < col.Length)
                result[i] = col[srcIdx];
        }
        return new StringColumn(col.Name, result);
    }

    // -- Math operations --

    /// <summary>
    /// Absolute value of each element.
    /// </summary>
    public static Column<T> Abs<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : T.Abs(span[i]);
        return Column<T>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Round each element to specified decimal places.
    /// </summary>
    public static Column<double> Round(this Column<double> col, int decimals = 0)
    {
        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : Math.Round(span[i], decimals);
        return Column<double>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Square root of each element.
    /// </summary>
    public static Column<double> Sqrt<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : Math.Sqrt(double.CreateChecked(span[i]));
        return Column<double>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Natural log of each element.
    /// </summary>
    public static Column<double> Log<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : Math.Log(double.CreateChecked(span[i]));
        return Column<double>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Log base 10 of each element.
    /// </summary>
    public static Column<double> Log10<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : Math.Log10(double.CreateChecked(span[i]));
        return Column<double>.FromNullable(col.Name, result);
    }

    /// <summary>
    /// Power of each element.
    /// </summary>
    public static Column<double> Pow<T>(this Column<T> col, double exponent)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var result = new double?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : Math.Pow(double.CreateChecked(span[i]), exponent);
        return Column<double>.FromNullable(col.Name, result);
    }

    // -- Aggregates (SIMD-accelerated where possible) --

    public static T? Sum<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        if (col.Length == 0) return null;
        var span = col.Buffer.Span;

        // Fast path: no nulls → SIMD for double
        if (col.NullCount == 0)
        {
            if (typeof(T) == typeof(double))
            {
                var dSpan = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(span);
                return (T)(object)Storage.SimdOps.SumDouble(dSpan);
            }
        }

        // Fallback: scalar loop with null checks
        T sum = T.Zero;
        for (int i = 0; i < span.Length; i++)
            if (!col.Nulls.IsNull(i)) sum += span[i];
        return sum;
    }

    public static double? Mean<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        if (col.Length == 0) return null;
        int validCount = col.Count();
        if (validCount == 0) return null;
        var sum = col.Sum()!.Value;
        return double.CreateChecked(sum) / validCount;
    }

    public static T? Min<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        if (col.Length == 0) return null;
        var span = col.Buffer.Span;
        T? min = null;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            if (min is null || span[i] < min.Value)
                min = span[i];
        }
        return min;
    }

    public static T? Max<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        if (col.Length == 0) return null;
        var span = col.Buffer.Span;
        T? max = null;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            if (max is null || span[i] > max.Value)
                max = span[i];
        }
        return max;
    }

    // -- Extended Aggregates --

    public static double? Median<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var values = GetNonNullCopy(col);
        if (values.Length == 0) return null;
        int mid = values.Length / 2;
        if (values.Length % 2 == 0)
        {
            double a = double.CreateChecked(QuickSelect(values, mid - 1));
            double b = double.CreateChecked(QuickSelect(values, mid));
            return (a + b) / 2.0;
        }
        return double.CreateChecked(QuickSelect(values, mid));
    }

    /// <summary>
    /// O(n) average-case selection of the k-th smallest element (Quickselect).
    /// Modifies the array in-place.
    /// </summary>
    private static T QuickSelect<T>(T[] arr, int k) where T : struct, INumber<T>
    {
        int lo = 0, hi = arr.Length - 1;
        var rng = new Random(42); // deterministic for reproducibility
        while (lo < hi)
        {
            // Median-of-3 pivot
            int pivotIdx = MedianOfThree(arr, lo, hi);
            (arr[pivotIdx], arr[hi]) = (arr[hi], arr[pivotIdx]);
            T pivot = arr[hi];

            int store = lo;
            for (int i = lo; i < hi; i++)
            {
                if (Comparer<T>.Default.Compare(arr[i], pivot) < 0)
                {
                    (arr[store], arr[i]) = (arr[i], arr[store]);
                    store++;
                }
            }
            (arr[store], arr[hi]) = (arr[hi], arr[store]);

            if (store == k) return arr[store];
            else if (store < k) lo = store + 1;
            else hi = store - 1;
        }
        return arr[lo];
    }

    private static int MedianOfThree<T>(T[] arr, int lo, int hi) where T : struct, INumber<T>
    {
        int mid = lo + (hi - lo) / 2;
        if (Comparer<T>.Default.Compare(arr[lo], arr[mid]) > 0) (arr[lo], arr[mid]) = (arr[mid], arr[lo]);
        if (Comparer<T>.Default.Compare(arr[lo], arr[hi]) > 0) (arr[lo], arr[hi]) = (arr[hi], arr[lo]);
        if (Comparer<T>.Default.Compare(arr[mid], arr[hi]) > 0) (arr[mid], arr[hi]) = (arr[hi], arr[mid]);
        return mid;
    }

    private static T[] GetNonNullCopy<T>(Column<T> col) where T : struct, INumber<T>
    {
        // Fast path: no nulls → direct span copy (avoids List intermediate)
        if (col.NullCount == 0)
            return col.Buffer.Span.ToArray();

        var span = col.Buffer.Span;
        var result = new T[col.Count()];
        int j = 0;
        for (int i = 0; i < span.Length; i++)
            if (!col.Nulls.IsNull(i)) result[j++] = span[i];
        return result;
    }

    public static double? Quantile<T>(this Column<T> col, double p)
        where T : struct, INumber<T>
    {
        if (p < 0 || p > 1) throw new ArgumentOutOfRangeException(nameof(p), "Must be between 0 and 1.");
        var values = GetNonNullCopy(col);
        if (values.Length == 0) return null;

        double idx = p * (values.Length - 1);
        int lo = (int)idx;
        int hi = Math.Min(lo + 1, values.Length - 1);

        if (lo == hi)
        {
            // Exact index — use O(n) quickselect
            return double.CreateChecked(QuickSelect(values, lo));
        }

        // Interpolated — quickselect for lo, then partial sort gives us hi for free
        // (after QuickSelect(k), elements at indices < k are all <= arr[k])
        double loVal = double.CreateChecked(QuickSelect(values, lo));
        // After quickselect for lo, arr[lo+1..n] are all >= arr[lo]
        // Find minimum of arr[lo+1..n] for arr[hi]
        T hiVal = values[lo + 1];
        for (int i = lo + 2; i < values.Length; i++)
            if (Comparer<T>.Default.Compare(values[i], hiVal) < 0)
                hiVal = values[i];
        double frac = idx - lo;
        return double.CreateChecked(loVal) * (1 - frac) + double.CreateChecked(hiVal) * frac;
    }

    public static double? Std<T>(this Column<T> col, int ddof = 1)
        where T : struct, INumber<T>
    {
        var variance = col.Var(ddof);
        return variance.HasValue ? Math.Sqrt(variance.Value) : null;
    }

    public static double? Var<T>(this Column<T> col, int ddof = 1)
        where T : struct, INumber<T>
    {
        int n = col.Count();
        if (n <= ddof) return null;
        var mean = col.Mean()!.Value;
        var span = col.Buffer.Span;
        double sumSq = 0;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            double diff = double.CreateChecked(span[i]) - mean;
            sumSq += diff * diff;
        }
        return sumSq / (n - ddof);
    }

    public static T? Mode<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        if (col.Count() == 0) return null;
        var counts = new Dictionary<T, int>();
        var span = col.Buffer.Span;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            var val = span[i];
            counts[val] = counts.GetValueOrDefault(val) + 1;
        }
        return counts.MaxBy(kv => kv.Value).Key;
    }

    public static double? Skew<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        int n = col.Count();
        if (n < 3) return null;
        var mean = col.Mean()!.Value;
        var std = col.Std()!.Value;
        if (std == 0) return 0;

        var span = col.Buffer.Span;
        double sum3 = 0;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            double diff = (double.CreateChecked(span[i]) - mean) / std;
            sum3 += diff * diff * diff;
        }
        // Adjusted Fisher-Pearson standardized moment coefficient
        return (double)n / ((n - 1) * (n - 2)) * sum3;
    }

    public static double? Kurtosis<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        int n = col.Count();
        if (n < 4) return null;
        var mean = col.Mean()!.Value;
        var std = col.Std()!.Value;
        if (std == 0) return 0;

        var span = col.Buffer.Span;
        double sum4 = 0;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            double diff = (double.CreateChecked(span[i]) - mean) / std;
            sum4 += diff * diff * diff * diff;
        }
        // Excess kurtosis (Fisher's definition)
        double k = (double)n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * sum4;
        return k - 3.0 * (n - 1) * (n - 1) / ((n - 2) * (n - 3));
    }

    public static double? Sem<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        int n = col.Count();
        if (n == 0) return null;
        var std = col.Std();
        return std.HasValue ? std.Value / Math.Sqrt(n) : null;
    }

    // -- Cumulative Operations --

    public static Column<T> CumSum<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        // Fast path: no nulls → write directly to Arrow byte buffer
        if (col.NullCount == 0)
        {
            int n = col.Length;
            var bytes = new byte[n * Unsafe.SizeOf<T>()];
            var result = MemoryMarshal.Cast<byte, T>(bytes.AsSpan());
            var span = col.Buffer.Span;
            T running = T.Zero;
            for (int i = 0; i < n; i++) { running += span[i]; result[i] = running; }
            return Column<T>.WrapResult(col.Name, bytes, n);
        }
        var values = new T?[col.Length];
        T running2 = T.Zero;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { values[i] = null; continue; }
            running2 += col.Buffer.Span[i];
            values[i] = running2;
        }
        return Column<T>.FromNullable(col.Name, values);
    }

    public static Column<T> CumProd<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var values = new T?[col.Length];
        T running = T.One;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { values[i] = null; continue; }
            running *= col.Buffer.Span[i];
            values[i] = running;
        }
        return Column<T>.FromNullable(col.Name, values);
    }

    public static Column<T> CumMin<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        if (col.NullCount == 0)
        {
            int n = col.Length;
            if (n == 0) return Column<T>.WrapResult(col.Name, Array.Empty<byte>(), 0);
            var bytes = new byte[n * Unsafe.SizeOf<T>()];
            var result = MemoryMarshal.Cast<byte, T>(bytes.AsSpan());
            var span = col.Buffer.Span;
            T running = span[0];
            result[0] = running;
            for (int i = 1; i < n; i++) { if (span[i] < running) running = span[i]; result[i] = running; }
            return Column<T>.WrapResult(col.Name, bytes, n);
        }
        var values = new T?[col.Length];
        T? running2 = null;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { values[i] = null; continue; }
            var val = col.Buffer.Span[i];
            running2 = running2 is null || val < running2.Value ? val : running2;
            values[i] = running2;
        }
        return Column<T>.FromNullable(col.Name, values);
    }

    public static Column<T> CumMax<T>(this Column<T> col)
        where T : struct, IComparisonOperators<T, T, bool>
    {
        if (col.NullCount == 0)
        {
            int n = col.Length;
            if (n == 0) return Column<T>.WrapResult(col.Name, Array.Empty<byte>(), 0);
            var bytes = new byte[n * Unsafe.SizeOf<T>()];
            var result = MemoryMarshal.Cast<byte, T>(bytes.AsSpan());
            var span = col.Buffer.Span;
            T running = span[0];
            result[0] = running;
            for (int i = 1; i < n; i++) { if (span[i] > running) running = span[i]; result[i] = running; }
            return Column<T>.WrapResult(col.Name, bytes, n);
        }
        var values = new T?[col.Length];
        T? running2 = null;
        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { values[i] = null; continue; }
            var val = col.Buffer.Span[i];
            running2 = running2 is null || val > running2.Value ? val : running2;
            values[i] = running2;
        }
        return Column<T>.FromNullable(col.Name, values);
    }

    // -- Diff and PctChange --

    public static Column<double> PctChange<T>(this Column<T> col, int periods = 1)
        where T : struct, INumber<T>
    {
        // Fast path for Column<double> with no nulls — uses nullable for first 'periods' elements
        if (col is Column<double> dc && dc.NullCount == 0)
        {
            int n = dc.Length;
            var result = new double?[n];
            var span = dc.Buffer.Span;
            for (int i = periods; i < n; i++)
                result[i] = span[i - periods] != 0 ? (span[i] - span[i - periods]) / span[i - periods] : null;
            return Column<double>.FromNullable(dc.Name, result);
        }
        var values = new double?[col.Length];
        for (int i = periods; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i) || col.Nulls.IsNull(i - periods))
            {
                values[i] = null;
                continue;
            }
            double prev = double.CreateChecked(col.Buffer.Span[i - periods]);
            if (prev == 0) { values[i] = null; continue; }
            double curr = double.CreateChecked(col.Buffer.Span[i]);
            values[i] = (curr - prev) / prev;
        }
        return Column<double>.FromNullable(col.Name, values);
    }

    public static Column<double> Diff<T>(this Column<T> col, int periods = 1)
        where T : struct, INumber<T>
    {
        var values = new double?[col.Length];
        for (int i = periods; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i) || col.Nulls.IsNull(i - periods))
            {
                values[i] = null;
                continue;
            }
            double prev = double.CreateChecked(col.Buffer.Span[i - periods]);
            double curr = double.CreateChecked(col.Buffer.Span[i]);
            values[i] = curr - prev;
        }
        return Column<double>.FromNullable(col.Name, values);
    }

    // -- Prod (product of all non-null values) --

    public static T? Prod<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        if (col.Length == 0 || col.Count() == 0) return null;
        var span = col.Buffer.Span;
        T product = T.One;
        for (int i = 0; i < span.Length; i++)
            if (!col.Nulls.IsNull(i)) product *= span[i];
        return product;
    }

    // -- CastColumn: convert column type --

    /// <summary>
    /// Cast a numeric column to a different numeric type.
    /// Usage: df.CastColumn&lt;double&gt;("IntColumn")
    /// </summary>
    public static Column<TTarget> Cast<TSource, TTarget>(this Column<TSource> col)
        where TSource : struct, INumber<TSource>
        where TTarget : struct, INumber<TTarget>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
        {
            var result = new TTarget[col.Length];
            for (int i = 0; i < col.Length; i++)
                result[i] = TTarget.CreateChecked(span[i]);
            return new Column<TTarget>(col.Name, result);
        }

        var nullable = new TTarget?[col.Length];
        for (int i = 0; i < col.Length; i++)
            nullable[i] = col.Nulls.IsNull(i) ? null : TTarget.CreateChecked(span[i]);
        return Column<TTarget>.FromNullable(col.Name, nullable);
    }

    // -- Cut / QCut (binning) --

    /// <summary>
    /// Bin values into equal-width bins (like pandas pd.cut).
    /// Returns a StringColumn with bin labels like "(0.0, 10.0]".
    /// </summary>
    /// <param name="numBins">Number of equal-width bins.</param>
    /// <param name="labels">Custom labels (length must match numBins). Null = auto-generate.</param>
    public static StringColumn Cut(this Column<double> col, int numBins, string[]? labels = null)
    {
        if (numBins < 1) throw new ArgumentException("numBins must be >= 1.");
        if (labels is not null && labels.Length != numBins)
            throw new ArgumentException($"labels length ({labels.Length}) must match numBins ({numBins}).");

        var span = col.Buffer.Span;
        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < span.Length; i++)
        {
            if (col.Nulls.IsNull(i)) continue;
            if (span[i] < min) min = span[i];
            if (span[i] > max) max = span[i];
        }

        if (min == max) max = min + 1; // avoid zero-width bins

        var edges = new double[numBins + 1];
        double width = (max - min) / numBins;
        for (int b = 0; b <= numBins; b++)
            edges[b] = min + b * width;
        edges[0] = min - width * 0.001; // slightly widen first bin to include min

        return BinByEdges(col, edges, labels);
    }

    /// <summary>
    /// Bin values into equal-width bins using explicit edge values.
    /// </summary>
    public static StringColumn Cut(this Column<double> col, double[] edges, string[]? labels = null)
    {
        if (edges.Length < 2) throw new ArgumentException("Need at least 2 edges.");
        if (labels is not null && labels.Length != edges.Length - 1)
            throw new ArgumentException($"labels length ({labels.Length}) must be edges.Length - 1 ({edges.Length - 1}).");
        return BinByEdges(col, edges, labels);
    }

    /// <summary>
    /// Bin values into quantile-based bins (like pandas pd.qcut).
    /// Each bin contains approximately the same number of values.
    /// </summary>
    /// <param name="numQuantiles">Number of quantile bins (e.g., 4 for quartiles).</param>
    /// <param name="labels">Custom labels (length must match numQuantiles). Null = auto-generate.</param>
    public static StringColumn QCut(this Column<double> col, int numQuantiles, string[]? labels = null)
    {
        if (numQuantiles < 1) throw new ArgumentException("numQuantiles must be >= 1.");
        if (labels is not null && labels.Length != numQuantiles)
            throw new ArgumentException($"labels length ({labels.Length}) must match numQuantiles ({numQuantiles}).");

        var sorted = GetSortedNonNull(col);
        if (sorted.Length == 0)
        {
            var empty = new string?[col.Length];
            return new StringColumn($"{col.Name}_qbin", empty);
        }

        // Compute quantile edges
        var edges = new double[numQuantiles + 1];
        edges[0] = sorted[0] - 0.001; // include min
        edges[numQuantiles] = sorted[^1];
        for (int q = 1; q < numQuantiles; q++)
        {
            double frac = (double)q / numQuantiles;
            double idx = frac * (sorted.Length - 1);
            int lo = (int)idx;
            int hi = Math.Min(lo + 1, sorted.Length - 1);
            double t = idx - lo;
            edges[q] = sorted[lo] * (1 - t) + sorted[hi] * t;
        }

        return BinByEdges(col, edges, labels);
    }

    private static StringColumn BinByEdges(Column<double> col, double[] edges, string[]? labels)
    {
        int numBins = edges.Length - 1;
        var binLabels = labels ?? new string[numBins];
        if (labels is null)
        {
            for (int b = 0; b < numBins; b++)
                binLabels[b] = $"({edges[b]:G6}, {edges[b + 1]:G6}]";
        }

        var span = col.Buffer.Span;
        var result = new string?[col.Length];

        for (int i = 0; i < col.Length; i++)
        {
            if (col.Nulls.IsNull(i)) { result[i] = null; continue; }

            double val = span[i];
            // Binary search for the right bin
            int bin = -1;
            for (int b = 0; b < numBins; b++)
            {
                if (val > edges[b] && val <= edges[b + 1])
                {
                    bin = b;
                    break;
                }
            }

            // Edge case: value equals last edge → last bin
            if (bin == -1 && val == edges[numBins])
                bin = numBins - 1;
            // Edge case: value equals first edge → first bin
            if (bin == -1 && val <= edges[0])
                bin = 0;

            result[i] = bin >= 0 ? binLabels[bin] : null;
        }

        return new StringColumn($"{col.Name}_bin", result);
    }

    // -- Helpers --

    private static T[] GetSortedNonNull<T>(Column<T> col) where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        var list = new List<T>(col.Count());
        for (int i = 0; i < span.Length; i++)
            if (!col.Nulls.IsNull(i)) list.Add(span[i]);
        var arr = list.ToArray();
        Array.Sort(arr);
        return arr;
    }
}
