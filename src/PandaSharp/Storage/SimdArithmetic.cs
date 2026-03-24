using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace PandaSharp.Storage;

/// <summary>
/// SIMD-accelerated element-wise arithmetic using Vector&lt;T&gt;.
/// Falls back to scalar on unsupported types or short arrays.
/// </summary>
internal static class SimdArithmetic
{
    public static void Add<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && left.Length >= Vector<T>.Count)
        {
            var vl = MemoryMarshal.Cast<T, Vector<T>>(left);
            var vr = MemoryMarshal.Cast<T, Vector<T>>(right);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vl.Length; v++)
                vo[v] = vl[v] + vr[v];
            i = vl.Length * Vector<T>.Count;
        }
        for (; i < left.Length; i++)
            result[i] = left[i] + right[i];
    }

    public static void Subtract<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && left.Length >= Vector<T>.Count)
        {
            var vl = MemoryMarshal.Cast<T, Vector<T>>(left);
            var vr = MemoryMarshal.Cast<T, Vector<T>>(right);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vl.Length; v++)
                vo[v] = vl[v] - vr[v];
            i = vl.Length * Vector<T>.Count;
        }
        for (; i < left.Length; i++)
            result[i] = left[i] - right[i];
    }

    public static void Multiply<T>(ReadOnlySpan<T> left, ReadOnlySpan<T> right, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && left.Length >= Vector<T>.Count)
        {
            var vl = MemoryMarshal.Cast<T, Vector<T>>(left);
            var vr = MemoryMarshal.Cast<T, Vector<T>>(right);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vl.Length; v++)
                vo[v] = vl[v] * vr[v];
            i = vl.Length * Vector<T>.Count;
        }
        for (; i < left.Length; i++)
            result[i] = left[i] * right[i];
    }

    public static void MultiplyScalar<T>(ReadOnlySpan<T> src, T scalar, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && src.Length >= Vector<T>.Count)
        {
            var scalarVec = new Vector<T>(scalar);
            var vs = MemoryMarshal.Cast<T, Vector<T>>(src);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vs.Length; v++)
                vo[v] = vs[v] * scalarVec;
            i = vs.Length * Vector<T>.Count;
        }
        for (; i < src.Length; i++)
            result[i] = src[i] * scalar;
    }

    public static void AddScalar<T>(ReadOnlySpan<T> src, T scalar, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && src.Length >= Vector<T>.Count)
        {
            var scalarVec = new Vector<T>(scalar);
            var vs = MemoryMarshal.Cast<T, Vector<T>>(src);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vs.Length; v++)
                vo[v] = vs[v] + scalarVec;
            i = vs.Length * Vector<T>.Count;
        }
        for (; i < src.Length; i++)
            result[i] = src[i] + scalar;
    }

    public static void SubtractScalar<T>(ReadOnlySpan<T> src, T scalar, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && src.Length >= Vector<T>.Count)
        {
            var scalarVec = new Vector<T>(scalar);
            var vs = MemoryMarshal.Cast<T, Vector<T>>(src);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vs.Length; v++)
                vo[v] = vs[v] - scalarVec;
            i = vs.Length * Vector<T>.Count;
        }
        for (; i < src.Length; i++)
            result[i] = src[i] - scalar;
    }

    public static void DivideScalar<T>(ReadOnlySpan<T> src, T scalar, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && src.Length >= Vector<T>.Count)
        {
            var scalarVec = new Vector<T>(scalar);
            var vs = MemoryMarshal.Cast<T, Vector<T>>(src);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vs.Length; v++)
                vo[v] = vs[v] / scalarVec;
            i = vs.Length * Vector<T>.Count;
        }
        for (; i < src.Length; i++)
            result[i] = src[i] / scalar;
    }

    public static void Negate<T>(ReadOnlySpan<T> src, Span<T> result)
        where T : struct, INumber<T>
    {
        int i = 0;
        if (Vector.IsHardwareAccelerated && src.Length >= Vector<T>.Count)
        {
            var vs = MemoryMarshal.Cast<T, Vector<T>>(src);
            var vo = MemoryMarshal.Cast<T, Vector<T>>(result);
            for (int v = 0; v < vs.Length; v++)
                vo[v] = -vs[v];
            i = vs.Length * Vector<T>.Count;
        }
        for (; i < src.Length; i++)
            result[i] = -src[i];
    }

    // -- Zero-copy Column builders: SIMD compute directly into Arrow byte buffer --

    private static Span<T> AsTypedSpan<T>(byte[] bytes, int length) where T : struct
        => MemoryMarshal.Cast<byte, T>(bytes.AsSpan()).Slice(0, length);

    /// <summary>Compute left + right directly into a new Column, zero intermediate copy.</summary>
    public static Column.Column<T> AddToColumn<T>(string name, ReadOnlySpan<T> left, ReadOnlySpan<T> right)
        where T : struct, INumber<T>
    {
        int len = left.Length;
        var bytes = new byte[len * Unsafe.SizeOf<T>()];
        Add(left, right, AsTypedSpan<T>(bytes, len));
        return Column.Column<T>.WrapResult(name, bytes, len);
    }

    public static Column.Column<T> SubtractToColumn<T>(string name, ReadOnlySpan<T> left, ReadOnlySpan<T> right)
        where T : struct, INumber<T>
    {
        int len = left.Length;
        var bytes = new byte[len * Unsafe.SizeOf<T>()];
        Subtract(left, right, AsTypedSpan<T>(bytes, len));
        return Column.Column<T>.WrapResult(name, bytes, len);
    }

    public static Column.Column<T> MultiplyToColumn<T>(string name, ReadOnlySpan<T> left, ReadOnlySpan<T> right)
        where T : struct, INumber<T>
    {
        int len = left.Length;
        var bytes = new byte[len * Unsafe.SizeOf<T>()];
        Multiply(left, right, AsTypedSpan<T>(bytes, len));
        return Column.Column<T>.WrapResult(name, bytes, len);
    }

    public static Column.Column<T> MultiplyScalarToColumn<T>(string name, ReadOnlySpan<T> src, T scalar)
        where T : struct, INumber<T>
    {
        int len = src.Length;
        var bytes = new byte[len * Unsafe.SizeOf<T>()];
        MultiplyScalar(src, scalar, AsTypedSpan<T>(bytes, len));
        return Column.Column<T>.WrapResult(name, bytes, len);
    }

    public static Column.Column<T> AddScalarToColumn<T>(string name, ReadOnlySpan<T> src, T scalar)
        where T : struct, INumber<T>
    {
        int len = src.Length;
        var bytes = new byte[len * Unsafe.SizeOf<T>()];
        AddScalar(src, scalar, AsTypedSpan<T>(bytes, len));
        return Column.Column<T>.WrapResult(name, bytes, len);
    }
}
