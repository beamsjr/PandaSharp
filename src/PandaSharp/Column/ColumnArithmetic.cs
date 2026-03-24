using System.Numerics;
using PandaSharp.Storage;

namespace PandaSharp.Column;

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
        // No null-free fast path for divide — need to check for zero
        var result = new T?[left.Length];
        for (int i = 0; i < left.Length; i++)
        {
            if (left.Nulls.IsNull(i) || right.Nulls.IsNull(i) || rs[i] == T.Zero)
                result[i] = null;
            else
                result[i] = ls[i] / rs[i];
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
}
