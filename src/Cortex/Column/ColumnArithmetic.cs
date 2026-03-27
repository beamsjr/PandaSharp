using System.Numerics;
using Cortex.Storage;

namespace Cortex.Column;

/// <summary>
/// Arithmetic operators for typed columns: col + col, col * scalar, etc.
/// </summary>
public static class ColumnArithmetic
{
    // -- Column + Column --

    public static Column<T> Add<T>(this Column<T> left, Column<T> right)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");
        var ls = left.Buffer.Span;
        var rs = right.Buffer.Span;
        // Fast path: no nulls → SIMD directly into Arrow byte buffer, zero intermediate copy
        if (left.NullCount == 0 && right.NullCount == 0)
            return SimdArithmetic.AddToColumn(left.Name, ls, rs);
        var result = new T?[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left.Nulls.IsNull(i) || right.Nulls.IsNull(i) ? null : ls[i] + rs[i];
        return Column<T>.FromNullable(left.Name, result);
    }

    public static Column<T> Subtract<T>(this Column<T> left, Column<T> right)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");
        var ls = left.Buffer.Span;
        var rs = right.Buffer.Span;
        if (left.NullCount == 0 && right.NullCount == 0)
            return SimdArithmetic.SubtractToColumn(left.Name, ls, rs);
        var result = new T?[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left.Nulls.IsNull(i) || right.Nulls.IsNull(i) ? null : ls[i] - rs[i];
        return Column<T>.FromNullable(left.Name, result);
    }

    public static Column<T> Multiply<T>(this Column<T> left, Column<T> right)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");
        var ls = left.Buffer.Span;
        var rs = right.Buffer.Span;
        if (left.NullCount == 0 && right.NullCount == 0)
        {
            return SimdArithmetic.MultiplyToColumn(left.Name, ls, rs);
        }
        var result = new T?[left.Length];
        for (int i = 0; i < left.Length; i++)
            result[i] = left.Nulls.IsNull(i) || right.Nulls.IsNull(i) ? null : ls[i] * rs[i];
        return Column<T>.FromNullable(left.Name, result);
    }

    public static Column<T> Divide<T>(this Column<T> left, Column<T> right)
        where T : struct, INumber<T>
    {
        if (left.Length != right.Length)
            throw new ArgumentException("Column lengths must match.");
        var ls = left.Buffer.Span;
        var rs = right.Buffer.Span;
        bool isFloatingPoint = typeof(T) == typeof(double) || typeof(T) == typeof(float);
        var result = new T?[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            if (left.Nulls.IsNull(i) || right.Nulls.IsNull(i))
                result[i] = null;
            else if (rs[i] == T.Zero && !isFloatingPoint)
                result[i] = null; // integer division by zero → null
            else
                result[i] = ls[i] / rs[i]; // floating-point: IEEE 754 handles ±Inf/NaN
        }
        return Column<T>.FromNullable(left.Name, result);
    }

    // -- Column + Scalar --

    public static Column<T> Add<T>(this Column<T> col, T scalar)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
            return SimdArithmetic.AddScalarToColumn(col.Name, span, scalar);
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : span[i] + scalar;
        return Column<T>.FromNullable(col.Name, result);
    }

    public static Column<T> Subtract<T>(this Column<T> col, T scalar)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
        {
            var vals = new T[col.Length];
            SimdArithmetic.SubtractScalar(span, scalar, vals);
            return new Column<T>(col.Name, vals);
        }
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : span[i] - scalar;
        return Column<T>.FromNullable(col.Name, result);
    }

    public static Column<T> Multiply<T>(this Column<T> col, T scalar)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
            return SimdArithmetic.MultiplyScalarToColumn(col.Name, span, scalar);
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : span[i] * scalar;
        return Column<T>.FromNullable(col.Name, result);
    }

    public static Column<T> Divide<T>(this Column<T> col, T scalar)
        where T : struct, INumber<T>
    {
        if (scalar == T.Zero) throw new DivideByZeroException();
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
        {
            var vals = new T[col.Length];
            SimdArithmetic.DivideScalar(span, scalar, vals);
            return new Column<T>(col.Name, vals);
        }
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : span[i] / scalar;
        return Column<T>.FromNullable(col.Name, result);
    }

    // -- Negate --

    public static Column<T> Negate<T>(this Column<T> col)
        where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
        {
            var vals = new T[col.Length];
            SimdArithmetic.Negate(span, vals);
            return new Column<T>(col.Name, vals);
        }
        var result = new T?[col.Length];
        for (int i = 0; i < col.Length; i++)
            result[i] = col.Nulls.IsNull(i) ? null : -span[i];
        return Column<T>.FromNullable(col.Name, result);
    }

    // -- Rename convenience --

    public static Column<T> Rename<T>(this Column<T> col, string newName) where T : struct =>
        (Column<T>)col.Clone(newName);

    public static StringColumn Rename(this StringColumn col, string newName) =>
        (StringColumn)col.Clone(newName);

    // -- Type widening --

    /// <summary>
    /// Convert a numeric column to Column&lt;double&gt;.
    /// Supports int, long, float, double, byte, short, decimal.
    /// </summary>
    public static Column<double> AsDouble<T>(this Column<T> col) where T : struct, INumber<T>
    {
        var span = col.Buffer.Span;
        if (col.NullCount == 0)
        {
            var result = new double[col.Length];
            for (int i = 0; i < span.Length; i++)
                result[i] = double.CreateChecked(span[i]);
            return new Column<double>(col.Name, result);
        }
        var nullable = new double?[col.Length];
        for (int i = 0; i < span.Length; i++)
            nullable[i] = col.Nulls.IsNull(i) ? null : double.CreateChecked(span[i]);
        return Column<double>.FromNullable(col.Name, nullable);
    }

    // -- Mixed-type arithmetic (int + double, float + double, etc.) --

    public static Column<double> Add(this Column<int> left, Column<double> right)
        => left.AsDouble().Add(right);

    public static Column<double> Add(this Column<double> left, Column<int> right)
        => left.Add(right.AsDouble());

    public static Column<double> Subtract(this Column<int> left, Column<double> right)
        => left.AsDouble().Subtract(right);

    public static Column<double> Subtract(this Column<double> left, Column<int> right)
        => left.Subtract(right.AsDouble());

    public static Column<double> Multiply(this Column<int> left, Column<double> right)
        => left.AsDouble().Multiply(right);

    public static Column<double> Multiply(this Column<double> left, Column<int> right)
        => left.Multiply(right.AsDouble());

    public static Column<double> Divide(this Column<int> left, Column<double> right)
        => left.AsDouble().Divide(right);

    public static Column<double> Divide(this Column<double> left, Column<int> right)
        => left.Divide(right.AsDouble());

    // float + double
    public static Column<double> Add(this Column<float> left, Column<double> right)
        => left.AsDouble().Add(right);

    public static Column<double> Add(this Column<double> left, Column<float> right)
        => left.Add(right.AsDouble());

    public static Column<double> Subtract(this Column<float> left, Column<double> right)
        => left.AsDouble().Subtract(right);

    public static Column<double> Subtract(this Column<double> left, Column<float> right)
        => left.Subtract(right.AsDouble());

    public static Column<double> Multiply(this Column<float> left, Column<double> right)
        => left.AsDouble().Multiply(right);

    public static Column<double> Multiply(this Column<double> left, Column<float> right)
        => left.Multiply(right.AsDouble());

    public static Column<double> Divide(this Column<float> left, Column<double> right)
        => left.AsDouble().Divide(right);

    public static Column<double> Divide(this Column<double> left, Column<float> right)
        => left.Divide(right.AsDouble());

    // long + double
    public static Column<double> Add(this Column<long> left, Column<double> right)
        => left.AsDouble().Add(right);

    public static Column<double> Add(this Column<double> left, Column<long> right)
        => left.Add(right.AsDouble());

    public static Column<double> Subtract(this Column<long> left, Column<double> right)
        => left.AsDouble().Subtract(right);

    public static Column<double> Subtract(this Column<double> left, Column<long> right)
        => left.Subtract(right.AsDouble());

    public static Column<double> Multiply(this Column<long> left, Column<double> right)
        => left.AsDouble().Multiply(right);

    public static Column<double> Multiply(this Column<double> left, Column<long> right)
        => left.Multiply(right.AsDouble());

    public static Column<double> Divide(this Column<long> left, Column<double> right)
        => left.AsDouble().Divide(right);

    public static Column<double> Divide(this Column<double> left, Column<long> right)
        => left.Divide(right.AsDouble());
}
