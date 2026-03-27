using System.Numerics;
using System.Runtime.InteropServices;
using Cortex.Column;
using Cortex.Storage;

namespace Cortex.ML.Tensors;

/// <summary>
/// Multi-dimensional view over contiguous typed memory.
/// Zero-copy interop with Cortex's Arrow-backed columns.
/// </summary>
public class Tensor<T> where T : struct, INumber<T>
{
    private readonly T[] _data;

    /// <summary>Dimensions of the tensor (e.g., [100, 3] for 100 rows × 3 features).</summary>
    public int[] Shape { get; }
    /// <summary>Number of dimensions (1D, 2D, etc.).</summary>
    public int Rank => Shape.Length;
    /// <summary>Total number of elements.</summary>
    public int Length => _data.Length;
    /// <summary>Flat read-only view of underlying data.</summary>
    public ReadOnlySpan<T> Span => _data;

    /// <summary>Create a tensor from data with specified shape.</summary>
    public Tensor(T[] data, params int[] shape)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        if (total != data.Length)
            throw new ArgumentException($"Shape {string.Join("×", shape)} = {total} elements, but data has {data.Length}.");
        _data = data;
        Shape = shape;
    }

    /// <summary>Create a tensor from a Cortex Column (copies data to owned array).</summary>
    public Tensor(Column<T> column)
    {
        _data = column.Values.ToArray();
        Shape = [column.Length];
    }

    // -- Factories --

    public static Tensor<T> Zeros(params int[] shape)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        return new Tensor<T>(new T[total], shape);
    }

    public static Tensor<T> Ones(params int[] shape)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new T[total];
        Array.Fill(data, T.One);
        return new Tensor<T>(data, shape);
    }

    public static Tensor<T> Random(int seed, params int[] shape)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new T[total];
        var rng = new Random(seed);
        for (int i = 0; i < total; i++)
            data[i] = T.CreateChecked(rng.NextDouble());
        return new Tensor<T>(data, shape);
    }

    /// <summary>Create a tensor with standard-normal random values (Box-Muller transform).</summary>
    public static Tensor<T> RandomNormal(int seed, params int[] shape)
    {
        int total = 1;
        foreach (var s in shape) total *= s;
        var data = new T[total];
        var rng = new Random(seed);
        for (int i = 0; i < total - 1; i += 2)
        {
            // Box-Muller transform: generates pairs of standard normal values
            double u1 = 1.0 - rng.NextDouble(); // avoid log(0)
            double u2 = rng.NextDouble();
            double mag = Math.Sqrt(-2.0 * Math.Log(u1));
            data[i] = T.CreateChecked(mag * Math.Cos(2.0 * Math.PI * u2));
            data[i + 1] = T.CreateChecked(mag * Math.Sin(2.0 * Math.PI * u2));
        }
        if (total % 2 == 1)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = rng.NextDouble();
            data[total - 1] = T.CreateChecked(Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2));
        }
        return new Tensor<T>(data, shape);
    }

    // -- Indexing --

    public T this[params int[] indices]
    {
        get => _data[FlatIndex(indices)];
        set => _data[FlatIndex(indices)] = value;
    }

    /// <summary>Get a row from a 2D tensor.</summary>
    public Tensor<T> Row(int row)
    {
        if (Rank != 2) throw new InvalidOperationException("Row() requires a 2D tensor.");
        int cols = Shape[1];
        var data = new T[cols];
        Array.Copy(_data, row * cols, data, 0, cols);
        return new Tensor<T>(data, cols);
    }

    /// <summary>Slice along an axis. Works for any rank.</summary>
    public Tensor<T> Slice(int axis, int start, int length)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));
        if (start < 0 || length < 0 || start + length > Shape[axis])
            throw new ArgumentOutOfRangeException($"Slice [{start}..{start + length}) out of range for axis {axis} with size {Shape[axis]}.");

        var newShape = (int[])Shape.Clone();
        newShape[axis] = length;

        int total = 1;
        foreach (var s in newShape) total *= s;
        var result = new T[total];

        // Compute strides for original and result
        var srcStrides = ComputeStrides(Shape);
        var dstStrides = ComputeStrides(newShape);

        // Iterate over all elements in the result, map back to source
        var dstIndices = new int[Rank];
        for (int flat = 0; flat < total; flat++)
        {
            UnflattenIndex(flat, dstStrides, dstIndices);
            // Offset the sliced axis
            dstIndices[axis] += start;
            int srcFlat = FlatIndex(dstIndices, srcStrides);
            result[flat] = _data[srcFlat];
            dstIndices[axis] -= start; // restore
        }

        return new Tensor<T>(result, newShape);
    }

    // -- Arithmetic (SIMD-accelerated) --

    public static Tensor<T> operator +(Tensor<T> a, Tensor<T> b)
    {
        CheckSameShape(a, b);
        var result = new T[a.Length];
        SimdArithmetic.Add<T>(a.Span, b.Span, result);
        return new Tensor<T>(result, a.Shape);
    }

    public static Tensor<T> operator -(Tensor<T> a, Tensor<T> b)
    {
        CheckSameShape(a, b);
        var result = new T[a.Length];
        SimdArithmetic.Subtract<T>(a.Span, b.Span, result);
        return new Tensor<T>(result, a.Shape);
    }

    public static Tensor<T> operator *(Tensor<T> a, Tensor<T> b)
    {
        CheckSameShape(a, b);
        var result = new T[a.Length];
        SimdArithmetic.Multiply<T>(a.Span, b.Span, result);
        return new Tensor<T>(result, a.Shape);
    }

    public static Tensor<T> operator *(Tensor<T> a, T scalar)
    {
        var result = new T[a.Length];
        SimdArithmetic.MultiplyScalar<T>(a.Span, scalar, result);
        return new Tensor<T>(result, a.Shape);
    }

    // -- Reductions --

    public T Sum() { T s = T.Zero; foreach (var v in _data) s += v; return s; }
    public double Mean() => double.CreateChecked(Sum()) / Length;

    /// <summary>Sum along a given axis, reducing that dimension. Works for any rank.</summary>
    public Tensor<T> SumAxis(int axis)
    {
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        // Result shape: remove the summed axis
        var newShape = Shape.Where((_, i) => i != axis).ToArray();
        if (newShape.Length == 0) newShape = [1]; // scalar → 1D of length 1

        int resultTotal = 1;
        foreach (var s in newShape) resultTotal *= s;
        var result = new T[resultTotal];

        var srcStrides = ComputeStrides(Shape);
        var dstStrides = ComputeStrides(newShape);
        var srcIndices = new int[Rank];

        // Iterate over all source elements, accumulate into result
        for (int flat = 0; flat < _data.Length; flat++)
        {
            UnflattenIndex(flat, srcStrides, srcIndices);

            // Map source indices to result indices (skip the summed axis)
            int dstFlat = 0;
            int d = 0;
            for (int i = 0; i < Rank; i++)
            {
                if (i == axis) continue;
                dstFlat += srcIndices[i] * dstStrides[d];
                d++;
            }

            result[dstFlat] += _data[flat];
        }

        return new Tensor<T>(result, newShape);
    }

    /// <summary>
    /// ArgMax along a given axis. Returns indices of maximum values.
    /// axis=-1: global argmax (single element). Otherwise, reduces along that axis.
    /// Works for any rank.
    /// </summary>
    public int[] ArgMax(int axis = -1)
    {
        if (axis == -1) return [Array.IndexOf(_data, _data.Max()!)];
        if (axis < 0 || axis >= Rank)
            throw new ArgumentOutOfRangeException(nameof(axis));

        // Result shape: remove the axis dimension
        var resultShape = Shape.Where((_, i) => i != axis).ToArray();
        if (resultShape.Length == 0) resultShape = [1];

        int resultTotal = 1;
        foreach (var s in resultShape) resultTotal *= s;

        var maxValues = new T[resultTotal];
        var maxIndices = new int[resultTotal];
        var initialized = new bool[resultTotal];

        var srcStrides = ComputeStrides(Shape);
        var dstStrides = ComputeStrides(resultShape);
        var srcIndices = new int[Rank];

        for (int flat = 0; flat < _data.Length; flat++)
        {
            UnflattenIndex(flat, srcStrides, srcIndices);

            // Map to result index (skip the axis dimension)
            int dstFlat = 0;
            int d = 0;
            for (int i = 0; i < Rank; i++)
            {
                if (i == axis) continue;
                dstFlat += srcIndices[i] * dstStrides[d];
                d++;
            }

            T val = _data[flat];
            if (!initialized[dstFlat] || val > maxValues[dstFlat])
            {
                maxValues[dstFlat] = val;
                maxIndices[dstFlat] = srcIndices[axis];
                initialized[dstFlat] = true;
            }
        }

        return maxIndices;
    }

    /// <summary>Transpose a 2D tensor.</summary>
    public Tensor<T> Transpose()
    {
        if (Rank != 2) throw new InvalidOperationException("Transpose requires 2D.");
        int rows = Shape[0], cols = Shape[1];
        var result = new T[Length];
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                result[c * rows + r] = _data[r * cols + c];
        return new Tensor<T>(result, cols, rows);
    }

    /// <summary>Matrix multiplication: (M×K) @ (K×N) → (M×N).</summary>
    public Tensor<T> MatMul(Tensor<T> other)
    {
        if (Rank != 2 || other.Rank != 2)
            throw new InvalidOperationException("MatMul requires 2D tensors.");
        int m = Shape[0], k = Shape[1];
        int k2 = other.Shape[0], n = other.Shape[1];
        if (k != k2)
            throw new ArgumentException($"Shape mismatch for MatMul: ({m}×{k}) @ ({k2}×{n})");

        var result = new T[m * n];
        // Cache-friendly ikj loop order
        for (int i = 0; i < m; i++)
        {
            for (int p = 0; p < k; p++)
            {
                T aip = _data[i * k + p];
                for (int j = 0; j < n; j++)
                    result[i * n + j] += aip * other._data[p * n + j];
            }
        }
        return new Tensor<T>(result, m, n);
    }

    /// <summary>Dot product of two 1D tensors.</summary>
    public T Dot(Tensor<T> other)
    {
        if (Rank != 1 || other.Rank != 1 || Length != other.Length)
            throw new ArgumentException("Dot requires same-length 1D tensors.");
        T sum = T.Zero;
        for (int i = 0; i < Length; i++)
            sum += _data[i] * other._data[i];
        return sum;
    }

    /// <summary>Raw data for framework interop. Returns the internal array directly — do not mutate.</summary>
    public T[] ToArray() => _data;
    internal T[] Data => _data;

    // -- Helpers --

    private int FlatIndex(int[] indices)
    {
        int flat = 0, stride = 1;
        for (int i = Shape.Length - 1; i >= 0; i--)
        {
            flat += indices[i] * stride;
            stride *= Shape[i];
        }
        return flat;
    }

    private static int FlatIndex(int[] indices, int[] strides)
    {
        int flat = 0;
        for (int i = 0; i < indices.Length; i++)
            flat += indices[i] * strides[i];
        return flat;
    }

    private static int[] ComputeStrides(int[] shape)
    {
        var strides = new int[shape.Length];
        int stride = 1;
        for (int i = shape.Length - 1; i >= 0; i--)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    }

    private static void UnflattenIndex(int flat, int[] strides, int[] indices)
    {
        for (int i = 0; i < strides.Length; i++)
        {
            indices[i] = flat / strides[i];
            flat %= strides[i];
        }
    }

    private static void CheckSameShape(Tensor<T> a, Tensor<T> b)
    {
        if (!a.Shape.SequenceEqual(b.Shape))
            throw new ArgumentException($"Shape mismatch: {string.Join("×", a.Shape)} vs {string.Join("×", b.Shape)}");
    }
}
