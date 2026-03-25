using System.Numerics;
using PandaSharp.Column;

namespace PandaSharp.Window;

public class RollingWindow<T> where T : struct, INumber<T>
{
    private readonly Column<T> _column;
    private readonly int _windowSize;
    private readonly int _minPeriods;
    private readonly bool _center;

    internal RollingWindow(Column<T> column, int windowSize, int minPeriods = -1, bool center = false)
    {
        _column = column;
        _windowSize = windowSize;
        _minPeriods = minPeriods >= 0 ? minPeriods : windowSize;
        _center = center;
    }

    // Typed fast path for Mean — O(n) sliding window with running sum
    public Column<double> Mean()
    {
        if (_column.NullCount == 0 && !_center)
            return RollingMeanFast();
        return Apply(vals => vals.Count > 0 ? vals.Average() : double.NaN);
    }

    public Column<double> Sum() => Apply(vals => vals.Count > 0 ? vals.Sum() : double.NaN);
    public Column<double> Min() => Apply(vals => vals.Count > 0 ? vals.Min() : double.NaN);
    public Column<double> Max() => Apply(vals => vals.Count > 0 ? vals.Max() : double.NaN);

    public Column<double> Std() => Apply(vals =>
    {
        if (vals.Count < 2) return double.NaN;
        double mean = vals.Average();
        return Math.Sqrt(vals.Sum(v => (v - mean) * (v - mean)) / (vals.Count - 1));
    });

    public Column<double> Var() => Apply(vals =>
    {
        if (vals.Count < 2) return double.NaN;
        double mean = vals.Average();
        return vals.Sum(v => (v - mean) * (v - mean)) / (vals.Count - 1);
    });

    /// <summary>O(n) sliding window mean for null-free, non-centered case.</summary>
    private Column<double> RollingMeanFast()
    {
        int n = _column.Length;

        // Native C fast path for double columns (only when minPeriods == windowSize, the default)
        if (typeof(T) == typeof(double) && Native.NativeOps.IsAvailable && _minPeriods == _windowSize)
        {
            var span = System.Runtime.InteropServices.MemoryMarshal.Cast<T, double>(_column.Buffer.Span);
            var nativeResult = Native.NativeOps.RollingMean(span, _windowSize);
            // Convert NaN to null for the first (window-1) elements
            var nullable = new double?[n];
            for (int i = 0; i < n; i++)
                nullable[i] = double.IsNaN(nativeResult[i]) ? null : nativeResult[i];
            return Column<double>.FromNullable(_column.Name, nullable);
        }

        var result = new double?[n];
        var src = _column.Buffer.Span;
        double sum = 0;

        for (int i = 0; i < n; i++)
        {
            sum += double.CreateChecked(src[i]);
            if (i >= _windowSize) sum -= double.CreateChecked(src[i - _windowSize]);
            int windowLen = Math.Min(i + 1, _windowSize);
            result[i] = windowLen >= _minPeriods ? sum / windowLen : null;
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    public Column<double> Apply(Func<List<double>, double> func)
    {
        int n = _column.Length;
        var result = new double?[n];
        // Reuse a single list, clear between rows
        var windowValues = new List<double>(_windowSize);

        for (int i = 0; i < n; i++)
        {
            int start, end;
            if (_center)
            {
                int half = _windowSize / 2;
                start = i - half;
                end = start + _windowSize;
            }
            else
            {
                end = i + 1;
                start = end - _windowSize;
            }

            start = Math.Max(0, start);
            end = Math.Min(n, end);

            windowValues.Clear();
            for (int j = start; j < end; j++)
            {
                if (!_column.Nulls.IsNull(j))
                    windowValues.Add(double.CreateChecked(_column.Buffer.Span[j]));
            }

            result[i] = windowValues.Count >= _minPeriods ? func(windowValues) : null;
        }

        return Column<double>.FromNullable(_column.Name, result);
    }
}

public class ExpandingWindow<T> where T : struct, INumber<T>
{
    private readonly Column<T> _column;
    private readonly int _minPeriods;

    internal ExpandingWindow(Column<T> column, int minPeriods = 1)
    {
        _column = column;
        _minPeriods = minPeriods;
    }

    // Typed fast paths — O(n) single pass, no copying
    public Column<double> Mean()
    {
        int n = _column.Length;
        var result = new double?[n];
        double sum = 0; int count = 0;
        var span = _column.Buffer.Span;
        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i)) { sum += double.CreateChecked(span[i]); count++; }
            result[i] = count >= _minPeriods ? sum / count : null;
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    public Column<double> Sum()
    {
        int n = _column.Length;
        var result = new double?[n];
        double sum = 0; int count = 0;
        var span = _column.Buffer.Span;
        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i)) { sum += double.CreateChecked(span[i]); count++; }
            result[i] = count >= _minPeriods ? sum : null;
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    public Column<double> Min()
    {
        int n = _column.Length;
        var result = new double?[n];
        double min = double.MaxValue; int count = 0;
        var span = _column.Buffer.Span;
        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i)) { double v = double.CreateChecked(span[i]); if (v < min) min = v; count++; }
            result[i] = count >= _minPeriods ? min : null;
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    public Column<double> Max()
    {
        int n = _column.Length;
        var result = new double?[n];
        double max = double.MinValue; int count = 0;
        var span = _column.Buffer.Span;
        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i)) { double v = double.CreateChecked(span[i]); if (v > max) max = v; count++; }
            result[i] = count >= _minPeriods ? max : null;
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    public Column<double> Std()
    {
        int n = _column.Length;
        var result = new double?[n];
        double sum = 0, sumSq = 0; int count = 0;
        var span = _column.Buffer.Span;
        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i)) { double v = double.CreateChecked(span[i]); sum += v; sumSq += v * v; count++; }
            if (count >= _minPeriods)
            {
                if (count > 1)
                {
                    double mean = sum / count;
                    result[i] = Math.Sqrt((sumSq - count * mean * mean) / (count - 1));
                }
                else
                    result[i] = double.NaN; // std of single value is NaN
            }
        }
        return Column<double>.FromNullable(_column.Name, result);
    }

    // Generic fallback for custom functions
    public Column<double> Apply(Func<List<double>, double> func)
    {
        int n = _column.Length;
        var result = new double?[n];
        var accumulated = new List<double>();

        for (int i = 0; i < n; i++)
        {
            if (!_column.Nulls.IsNull(i))
                accumulated.Add(double.CreateChecked(_column.Buffer.Span[i]));

            result[i] = accumulated.Count >= _minPeriods ? func(accumulated) : null;
        }

        return Column<double>.FromNullable(_column.Name, result);
    }
}

public class EwmWindow<T> where T : struct, INumber<T>
{
    private readonly Column<T> _column;
    private readonly double _alpha;

    internal EwmWindow(Column<T> column, double? span = null, double? alpha = null)
    {
        _column = column;
        if (alpha.HasValue)
            _alpha = alpha.Value;
        else if (span.HasValue)
            _alpha = 2.0 / (span.Value + 1);
        else
            throw new ArgumentException("Either span or alpha must be provided.");
    }

    public Column<double> Mean()
    {
        int n = _column.Length;
        var result = new double?[n];
        double? ewm = null;

        for (int i = 0; i < n; i++)
        {
            if (_column.Nulls.IsNull(i)) { result[i] = null; continue; }
            double val = double.CreateChecked(_column.Buffer.Span[i]);
            ewm = ewm is null ? val : _alpha * val + (1 - _alpha) * ewm.Value;
            result[i] = ewm;
        }

        return Column<double>.FromNullable(_column.Name, result);
    }
}
