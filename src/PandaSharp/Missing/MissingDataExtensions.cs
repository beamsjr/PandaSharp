using System.Numerics;
using System.Runtime.CompilerServices;
using PandaSharp.Column;

namespace PandaSharp.Missing;

public static class MissingDataExtensions
{
    /// <summary>
    /// Check if a value is effectively missing: null in bitmask OR NaN for double/float.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsEffectivelyMissing<T>(Column<T> col, int index) where T : struct
    {
        if (col.IsNull(index)) return true;
        if (typeof(T) == typeof(double))
            return double.IsNaN(Unsafe.As<T, double>(ref Unsafe.AsRef(in col.Buffer.Span[index])));
        if (typeof(T) == typeof(float))
            return float.IsNaN(Unsafe.As<T, float>(ref Unsafe.AsRef(in col.Buffer.Span[index])));
        return false;
    }

    /// <summary>
    /// Check if a value in an IColumn is missing: null in bitmask OR NaN for floating-point columns.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static bool IsEffectivelyMissing(IColumn column, int index)
    {
        if (column.IsNull(index)) return true;
        if (column is Column<double> dc)
            return double.IsNaN(dc.Buffer.Span[index]);
        if (column is Column<float> fc)
            return float.IsNaN(fc.Buffer.Span[index]);
        return false;
    }

    // -- Column-level operations --

    public static bool[] IsNa(this IColumn column)
    {
        var result = new bool[column.Length];
        for (int i = 0; i < column.Length; i++)
            result[i] = IsEffectivelyMissing(column, i);
        return result;
    }

    public static bool[] NotNa(this IColumn column)
    {
        var result = new bool[column.Length];
        for (int i = 0; i < column.Length; i++)
            result[i] = !IsEffectivelyMissing(column, i);
        return result;
    }

    public static Column<T> FillNa<T>(this Column<T> column, T value) where T : struct
    {
        // Fast path: if no nulls and no NaN, nothing to fill
        bool isFloating = typeof(T) == typeof(double) || typeof(T) == typeof(float);
        bool hasNaN = false;
        if (isFloating && column.NullCount == 0)
        {
            var checkSpan = column.Buffer.Span;
            for (int i = 0; i < checkSpan.Length; i++)
            {
                if (typeof(T) == typeof(double) && double.IsNaN(Unsafe.As<T, double>(ref Unsafe.AsRef(in checkSpan[i]))))
                { hasNaN = true; break; }
                if (typeof(T) == typeof(float) && float.IsNaN(Unsafe.As<T, float>(ref Unsafe.AsRef(in checkSpan[i]))))
                { hasNaN = true; break; }
            }
            if (!hasNaN) return column; // nothing to fill
        }

        var values = new T?[column.Length];
        if (!isFloating || (!hasNaN && column.NullCount > 0))
        {
            // No NaN possible or confirmed: use simpler IsNull check
            for (int i = 0; i < column.Length; i++)
                values[i] = column.IsNull(i) ? value : column[i];
        }
        else
        {
            for (int i = 0; i < column.Length; i++)
                values[i] = IsEffectivelyMissing(column, i) ? value : column[i];
        }
        return Column<T>.FromNullable(column.Name, values);
    }

    public static Column<T> FillNa<T>(this Column<T> column, FillStrategy strategy) where T : struct
    {
        return strategy switch
        {
            FillStrategy.Forward => FillForward(column),
            FillStrategy.Backward => FillBackward(column),
            _ => throw new ArgumentException($"Use FillNa(value) for scalar fill strategy.")
        };
    }

    public static StringColumn FillNa(this StringColumn column, string value)
    {
        var values = new string?[column.Length];
        for (int i = 0; i < column.Length; i++)
            values[i] = column.IsNull(i) ? value : column[i];
        return new StringColumn(column.Name, values);
    }

    public static StringColumn FillNa(this StringColumn column, FillStrategy strategy)
    {
        var values = new string?[column.Length];

        if (strategy == FillStrategy.Forward)
        {
            string? last = null;
            for (int i = 0; i < column.Length; i++)
            {
                if (!column.IsNull(i)) last = column[i];
                values[i] = column.IsNull(i) ? last : column[i];
            }
        }
        else if (strategy == FillStrategy.Backward)
        {
            string? next = null;
            for (int i = column.Length - 1; i >= 0; i--)
            {
                if (!column.IsNull(i)) next = column[i];
                values[i] = column.IsNull(i) ? next : column[i];
            }
        }
        else
        {
            throw new ArgumentException("Use FillNa(value) for scalar fill strategy.");
        }

        return new StringColumn(column.Name, values);
    }

    /// <summary>
    /// Quick check if a floating-point column has any NaN values in its buffer.
    /// </summary>
    private static bool ColumnHasNaN<T>(Column<T> column) where T : struct
    {
        if (typeof(T) == typeof(double))
        {
            var span = column.Buffer.Span;
            for (int i = 0; i < span.Length; i++)
                if (double.IsNaN(Unsafe.As<T, double>(ref Unsafe.AsRef(in span[i]))))
                    return true;
        }
        else if (typeof(T) == typeof(float))
        {
            var span = column.Buffer.Span;
            for (int i = 0; i < span.Length; i++)
                if (float.IsNaN(Unsafe.As<T, float>(ref Unsafe.AsRef(in span[i]))))
                    return true;
        }
        return false;
    }

    private static Column<T> FillForward<T>(Column<T> column) where T : struct
    {
        bool isFloating = typeof(T) == typeof(double) || typeof(T) == typeof(float);
        bool useSimple = !isFloating || !ColumnHasNaN(column);
        var values = new T?[column.Length];
        T? last = null;
        for (int i = 0; i < column.Length; i++)
        {
            bool missing = useSimple ? column.IsNull(i) : IsEffectivelyMissing(column, i);
            if (!missing) last = column[i];
            values[i] = missing ? last : column[i];
        }
        return Column<T>.FromNullable(column.Name, values);
    }

    private static Column<T> FillBackward<T>(Column<T> column) where T : struct
    {
        bool isFloating = typeof(T) == typeof(double) || typeof(T) == typeof(float);
        bool useSimple = !isFloating || !ColumnHasNaN(column);
        var values = new T?[column.Length];
        T? next = null;
        for (int i = column.Length - 1; i >= 0; i--)
        {
            bool missing = useSimple ? column.IsNull(i) : IsEffectivelyMissing(column, i);
            if (!missing) next = column[i];
            values[i] = missing ? next : column[i];
        }
        return Column<T>.FromNullable(column.Name, values);
    }

    // -- DataFrame-level operations --

    public static DataFrame DropNa(this DataFrame df, int axis = 0, int? threshold = null)
    {
        if (axis == 0)
        {
            // Drop rows with any nulls/NaN (or more than threshold nulls/NaN)
            var mask = new bool[df.RowCount];
            for (int r = 0; r < df.RowCount; r++)
            {
                int nullCount = 0;
                for (int c = 0; c < df.ColumnCount; c++)
                    if (IsEffectivelyMissing(df[df.ColumnNames[c]], r)) nullCount++;

                if (threshold.HasValue)
                    mask[r] = (df.ColumnCount - nullCount) >= threshold.Value;
                else
                    mask[r] = nullCount == 0;
            }
            return df.Filter(mask);
        }
        else
        {
            // Drop columns with any nulls/NaN
            var cols = new List<IColumn>();
            foreach (var name in df.ColumnNames)
            {
                var col = df[name];
                int missingCount = 0;
                for (int i = 0; i < col.Length; i++)
                    if (IsEffectivelyMissing(col, i)) missingCount++;

                if (threshold.HasValue)
                {
                    if ((col.Length - missingCount) >= threshold.Value)
                        cols.Add(col);
                }
                else
                {
                    if (missingCount == 0)
                        cols.Add(col);
                }
            }
            return new DataFrame(cols);
        }
    }

    // -- Interpolation for numeric columns --

    public static Column<double> Interpolate(this Column<double> column, InterpolationMethod method = InterpolationMethod.Linear)
    {
        // Check for NaN values in addition to null bitmask
        bool hasNaN = false;
        if (column.NullCount == 0)
        {
            var checkSpan = column.Buffer.Span;
            for (int i = 0; i < checkSpan.Length; i++)
                if (double.IsNaN(checkSpan[i])) { hasNaN = true; break; }
            if (!hasNaN)
                return column; // nothing to interpolate
        }

        return method switch
        {
            InterpolationMethod.Linear => InterpolateLinear(column),
            InterpolationMethod.Polynomial => InterpolatePolynomial(column),
            InterpolationMethod.Cubic or InterpolationMethod.Spline or InterpolationMethod.Pchip =>
                InterpolateCubicSpline(column),
            InterpolationMethod.Index => InterpolateLinear(column), // index-based = linear by position
            InterpolationMethod.Time => InterpolateLinear(column),  // time-based = linear by position
            _ => throw new NotSupportedException($"Interpolation method '{method}' is not supported.")
        };
    }

    private static Column<double> InterpolateLinear(Column<double> column)
    {
        int n = column.Length;
        var span = column.Buffer.Span;
        var result = new double?[n];

        // Copy non-null, non-NaN values
        for (int i = 0; i < n; i++)
            result[i] = (column.Nulls.IsNull(i) || double.IsNaN(span[i])) ? null : span[i];

        for (int i = 0; i < n; i++)
        {
            if (result[i].HasValue) continue;

            int prev = -1;
            for (int j = i - 1; j >= 0; j--)
                if (result[j].HasValue) { prev = j; break; }

            int next = -1;
            for (int j = i + 1; j < n; j++)
                if (result[j].HasValue) { next = j; break; }

            if (prev >= 0 && next >= 0)
                result[i] = result[prev]!.Value + (double)(i - prev) / (next - prev) * (result[next]!.Value - result[prev]!.Value);
            else if (prev >= 0)
                result[i] = result[prev]!.Value;
            else if (next >= 0)
                result[i] = result[next]!.Value;
            // else: no known values at all — leave as null
        }

        return Column<double>.FromNullable(column.Name, result);
    }

    /// <summary>
    /// Polynomial interpolation using Neville's algorithm with up to 3 surrounding known points.
    /// Falls back to linear when fewer than 3 points are available.
    /// </summary>
    private static Column<double> InterpolatePolynomial(Column<double> column)
    {
        int n = column.Length;
        var span = column.Buffer.Span;
        var result = new double?[n];

        // Collect known (index, value) pairs — skip both null and NaN
        var known = new List<(int Idx, double Val)>();
        for (int i = 0; i < n; i++)
        {
            if (!column.Nulls.IsNull(i) && !double.IsNaN(span[i]))
            {
                known.Add((i, span[i]));
                result[i] = span[i];
            }
        }

        if (known.Count < 2)
        {
            // Not enough points — fill with the single known value if available
            if (known.Count == 1)
                for (int i = 0; i < n; i++)
                    result[i] = known[0].Val;
            // else: all null — leave as null
            return Column<double>.FromNullable(column.Name, result);
        }

        for (int i = 0; i < n; i++)
        {
            if (result[i].HasValue) continue;

            // Find the 3 closest known points (or fewer if not available)
            var nearest = GetNearestKnown(known, i, 3);
            result[i] = NevilleInterpolate(nearest, i);
        }

        return Column<double>.FromNullable(column.Name, result);
    }

    /// <summary>
    /// Natural cubic spline interpolation through all known points.
    /// Produces smooth C2-continuous curves.
    /// </summary>
    private static Column<double> InterpolateCubicSpline(Column<double> column)
    {
        int n = column.Length;
        var span = column.Buffer.Span;
        var result = new double?[n];

        // Collect known (index, value) pairs — skip both null and NaN
        var knownIdx = new List<double>();
        var knownVal = new List<double>();
        for (int i = 0; i < n; i++)
        {
            if (!column.Nulls.IsNull(i) && !double.IsNaN(span[i]))
            {
                knownIdx.Add(i);
                knownVal.Add(span[i]);
                result[i] = span[i];
            }
        }

        if (knownIdx.Count < 2)
        {
            if (knownIdx.Count == 1)
                for (int i = 0; i < n; i++)
                    result[i] = knownVal[0];
            // else: all null — leave as null
            return Column<double>.FromNullable(column.Name, result);
        }

        // Compute natural cubic spline coefficients
        var x = knownIdx.ToArray();
        var y = knownVal.ToArray();
        int m = x.Length;
        var (a, b, c, d) = ComputeSplineCoefficients(x, y, m);

        // Evaluate spline at each null/NaN position
        for (int i = 0; i < n; i++)
        {
            if (result[i].HasValue) continue;

            // Find the spline segment
            double xi = i;
            int seg = 0;
            for (int j = 1; j < m; j++)
            {
                if (xi <= x[j]) { seg = j - 1; break; }
                seg = j - 1;
            }

            // Clamp to edges
            if (xi < x[0]) seg = 0;
            else if (xi > x[m - 1]) seg = m - 2;

            double dx = xi - x[seg];
            result[i] = a[seg] + b[seg] * dx + c[seg] * dx * dx + d[seg] * dx * dx * dx;
        }

        return Column<double>.FromNullable(column.Name, result);
    }

    /// <summary>Natural cubic spline: compute coefficients a, b, c, d for each segment.</summary>
    private static (double[] a, double[] b, double[] c, double[] d) ComputeSplineCoefficients(
        double[] x, double[] y, int n)
    {
        var a = new double[n];
        var b = new double[n - 1];
        var c = new double[n];
        var d = new double[n - 1];
        var h = new double[n - 1];
        var alpha = new double[n];

        for (int i = 0; i < n; i++) a[i] = y[i];
        for (int i = 0; i < n - 1; i++) h[i] = x[i + 1] - x[i];

        // Natural spline boundary: c[0] = c[n-1] = 0
        for (int i = 1; i < n - 1; i++)
            alpha[i] = 3.0 / h[i] * (a[i + 1] - a[i]) - 3.0 / h[i - 1] * (a[i] - a[i - 1]);

        // Solve tridiagonal system for c[]
        var l = new double[n];
        var mu = new double[n];
        var z = new double[n];

        l[0] = 1; mu[0] = 0; z[0] = 0;
        for (int i = 1; i < n - 1; i++)
        {
            l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1];
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[n - 1] = 1; z[n - 1] = 0; c[n - 1] = 0;

        for (int j = n - 2; j >= 0; j--)
        {
            c[j] = z[j] - mu[j] * c[j + 1];
            b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
            d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
        }

        return (a, b, c, d);
    }

    /// <summary>Get the k nearest known points to target index.</summary>
    private static List<(int Idx, double Val)> GetNearestKnown(List<(int Idx, double Val)> known, int target, int k)
    {
        // Binary search for closest position
        int lo = 0, hi = known.Count - 1;
        while (lo < hi)
        {
            int mid = (lo + hi) / 2;
            if (known[mid].Idx < target) lo = mid + 1;
            else hi = mid;
        }

        // Expand window around lo to get k points
        int left = lo - 1, right = lo;
        var result = new List<(int, double)>(k);
        while (result.Count < k && (left >= 0 || right < known.Count))
        {
            if (left < 0)
                result.Add(known[right++]);
            else if (right >= known.Count)
                result.Add(known[left--]);
            else if (Math.Abs(known[left].Idx - target) <= Math.Abs(known[right].Idx - target))
                result.Add(known[left--]);
            else
                result.Add(known[right++]);
        }
        return result;
    }

    /// <summary>Neville's algorithm for polynomial interpolation.</summary>
    private static double NevilleInterpolate(List<(int Idx, double Val)> points, double target)
    {
        int n = points.Count;
        var p = new double[n];
        for (int i = 0; i < n; i++) p[i] = points[i].Val;

        for (int k = 1; k < n; k++)
            for (int i = n - 1; i >= k; i--)
            {
                double xi = points[i].Idx;
                double xik = points[i - k].Idx;
                p[i] = ((target - xik) * p[i] - (target - xi) * p[i - 1]) / (xi - xik);
            }

        return p[n - 1];
    }

    public static Column<T> Interpolate<T>(this Column<T> column, InterpolationMethod method = InterpolationMethod.Linear)
        where T : struct, INumber<T>
    {
        // Check for NaN values in addition to null bitmask (same as Column<double> overload)
        bool hasMissing = column.NullCount > 0;
        if (!hasMissing && (typeof(T) == typeof(float) || typeof(T) == typeof(double)))
        {
            var checkSpan = column.Buffer.Span;
            for (int i = 0; i < checkSpan.Length; i++)
            {
                if (typeof(T) == typeof(float) && float.IsNaN(Unsafe.As<T, float>(ref Unsafe.AsRef(in checkSpan[i]))))
                { hasMissing = true; break; }
                if (typeof(T) == typeof(double) && double.IsNaN(Unsafe.As<T, double>(ref Unsafe.AsRef(in checkSpan[i]))))
                { hasMissing = true; break; }
            }
        }
        if (!hasMissing)
            return column;

        // For non-double types, convert to double, interpolate, convert back
        if (typeof(T) != typeof(double) && method != InterpolationMethod.Linear)
        {
            var asDouble = ConvertToDoubleColumn(column);
            var interpolated = asDouble.Interpolate(method);
            return ConvertFromDoubleColumn<T>(interpolated, column.Name);
        }

        // Single pass: read from typed buffer, interpolate as double, convert back
        int n = column.Length;
        var span = column.Buffer.Span;
        var result = new T?[n];

        // Copy non-null, non-NaN values
        for (int i = 0; i < n; i++)
            result[i] = IsEffectivelyMissing(column, i) ? null : span[i];

        // Interpolate nulls
        for (int i = 0; i < n; i++)
        {
            if (result[i].HasValue) continue;

            int prev = -1;
            for (int j = i - 1; j >= 0; j--)
                if (result[j].HasValue) { prev = j; break; }

            int next = -1;
            for (int j = i + 1; j < n; j++)
                if (result[j].HasValue) { next = j; break; }

            if (prev >= 0 && next >= 0)
            {
                double ratio = (double)(i - prev) / (next - prev);
                double val = double.CreateChecked(result[prev]!.Value) + ratio * (double.CreateChecked(result[next]!.Value) - double.CreateChecked(result[prev]!.Value));
                result[i] = T.CreateChecked(val);
            }
            else if (prev >= 0)
                result[i] = result[prev];
            else if (next >= 0)
                result[i] = result[next];
        }

        return Column<T>.FromNullable(column.Name, result);
    }

    private static Column<double> ConvertToDoubleColumn<T>(Column<T> column) where T : struct, INumber<T>
    {
        var result = new double?[column.Length];
        var span = column.Buffer.Span;
        for (int i = 0; i < column.Length; i++)
            result[i] = column.Nulls.IsNull(i) ? null : double.CreateChecked(span[i]);
        return Column<double>.FromNullable(column.Name, result);
    }

    private static Column<T> ConvertFromDoubleColumn<T>(Column<double> column, string name) where T : struct, INumber<T>
    {
        var span = column.Buffer.Span;
        var result = new T?[column.Length];
        for (int i = 0; i < column.Length; i++)
        {
            if (column.IsNull(i))
                result[i] = null;
            else
                result[i] = T.CreateChecked(span[i]);
        }
        return Column<T>.FromNullable(name, result);
    }
}
