using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.GPU;
using Cortex.ML.Tensors;

namespace Cortex.GPU.Tests;

public class GpuOpsTests : IDisposable
{
    private readonly GpuOps _ops;

    public GpuOpsTests()
    {
        _ops = new GpuOps();
    }

    public void Dispose() { }

    // ═══ Element-wise operations ═══

    [Fact]
    public void Add_TwoArrays_ReturnsElementWiseSum()
    {
        var a = new double[] { 1, 2, 3, 4, 5 };
        var b = new double[] { 10, 20, 30, 40, 50 };
        var result = _ops.Add(a, b);
        result.Should().BeEquivalentTo(new double[] { 11, 22, 33, 44, 55 });
    }

    [Fact]
    public void Subtract_TwoArrays_ReturnsElementWiseDifference()
    {
        var a = new double[] { 10, 20, 30 };
        var b = new double[] { 1, 2, 3 };
        var result = _ops.Subtract(a, b);
        result.Should().BeEquivalentTo(new double[] { 9, 18, 27 });
    }

    [Fact]
    public void Multiply_TwoArrays_ReturnsElementWiseProduct()
    {
        var a = new double[] { 2, 3, 4 };
        var b = new double[] { 5, 6, 7 };
        var result = _ops.Multiply(a, b);
        result.Should().BeEquivalentTo(new double[] { 10, 18, 28 });
    }

    [Fact]
    public void Divide_TwoArrays_ReturnsElementWiseQuotient()
    {
        var a = new double[] { 10, 20, 30 };
        var b = new double[] { 2, 5, 6 };
        var result = _ops.Divide(a, b);
        result.Should().BeEquivalentTo(new double[] { 5, 4, 5 });
    }

    [Fact]
    public void MultiplyScalar_ArrayAndScalar_ScalesAll()
    {
        var a = new double[] { 1, 2, 3, 4 };
        var result = _ops.MultiplyScalar(a, 3.0);
        result.Should().BeEquivalentTo(new double[] { 3, 6, 9, 12 });
    }

    [Fact]
    public void Negate_Array_NegatesAll()
    {
        var a = new double[] { 1, -2, 3 };
        var result = _ops.Negate(a);
        result.Should().BeEquivalentTo(new double[] { -1, 2, -3 });
    }

    [Fact]
    public void Abs_Array_AbsoluteValues()
    {
        var a = new double[] { -5, 3, -1, 0 };
        var result = _ops.Abs(a);
        result.Should().BeEquivalentTo(new double[] { 5, 3, 1, 0 });
    }

    [Fact]
    public void Sqrt_Array_SquareRoots()
    {
        var a = new double[] { 4, 9, 16, 25 };
        var result = _ops.Sqrt(a);
        for (int i = 0; i < result.Length; i++)
            result[i].Should().BeApproximately(Math.Sqrt(a[i]), 1e-10);
    }

    // ═══ Reductions ═══

    [Fact]
    public void SumMinMax_Array_ReturnsCorrectValues()
    {
        var data = new double[] { 3, 1, 4, 1, 5, 9, 2, 6 };
        var (sum, min, max) = _ops.SumMinMax(data);
        sum.Should().BeApproximately(31.0, 1e-10);
        min.Should().Be(1.0);
        max.Should().Be(9.0);
    }

    [Fact]
    public void Variance_Array_ReturnsCorrectVariance()
    {
        var data = new double[] { 2, 4, 4, 4, 5, 5, 7, 9 };
        double mean = data.Average();
        var variance = _ops.Variance(data, mean);
        double expected = data.Select(x => (x - mean) * (x - mean)).Average();
        variance.Should().BeApproximately(expected, 1e-10);
    }

    // ═══ Matrix operations ═══

    [Fact]
    public void MatMul_2x3_Times_3x2_Returns_2x2()
    {
        // A = [[1,2,3],[4,5,6]] (2×3)
        // B = [[7,8],[9,10],[11,12]] (3×2)
        var A = new double[] { 1, 2, 3, 4, 5, 6 };
        var B = new double[] { 7, 8, 9, 10, 11, 12 };
        var C = _ops.MatMul(A, B, 2, 3, 2);

        // C[0,0] = 1*7+2*9+3*11 = 58, C[0,1] = 1*8+2*10+3*12 = 64
        // C[1,0] = 4*7+5*9+6*11 = 139, C[1,1] = 4*8+5*10+6*12 = 154
        C[0].Should().BeApproximately(58, 1e-10);
        C[1].Should().BeApproximately(64, 1e-10);
        C[2].Should().BeApproximately(139, 1e-10);
        C[3].Should().BeApproximately(154, 1e-10);
    }

    [Fact]
    public void GramMatrix_ReturnsSymmetric()
    {
        // A is 3×2
        var A = new double[] { 1, 2, 3, 4, 5, 6 };
        var G = _ops.GramMatrix(A, 3, 2);

        // G = A^T @ A (2×2)
        // G[0,0] = 1*1+3*3+5*5 = 35
        // G[0,1] = 1*2+3*4+5*6 = 44
        // G[1,1] = 2*2+4*4+6*6 = 56
        G[0].Should().BeApproximately(35, 1e-10);
        G[1].Should().BeApproximately(44, 1e-10);
        G[2].Should().BeApproximately(44, 1e-10); // symmetric
        G[3].Should().BeApproximately(56, 1e-10);
    }

    [Fact]
    public void PairwiseDistances_KnownPoints()
    {
        // X = [[0,0], [1,1]], Y = [[0,0], [3,4]]
        var X = new double[] { 0, 0, 1, 1 };
        var Y = new double[] { 0, 0, 3, 4 };
        var dist = _ops.PairwiseDistances(X, Y, 2, 2, 2);

        // dist[0,0] = 0, dist[0,1] = 9+16=25
        // dist[1,0] = 1+1=2, dist[1,1] = 4+9=13
        dist[0].Should().BeApproximately(0, 1e-10);
        dist[1].Should().BeApproximately(25, 1e-10);
        dist[2].Should().BeApproximately(2, 1e-10);
        dist[3].Should().BeApproximately(13, 1e-10);
    }

    // ═══ Extension methods ═══

    [Fact]
    public void GpuAdd_Columns_ReturnsSum()
    {
        var a = new Column<double>("a", [1, 2, 3, 4, 5]);
        var b = new Column<double>("b", [10, 20, 30, 40, 50]);
        var result = a.GpuAdd(b);
        result.Values.ToArray().Should().BeEquivalentTo(new double[] { 11, 22, 33, 44, 55 });
    }

    [Fact]
    public void GpuMatMul_Tensors_ReturnsProduct()
    {
        var A = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);
        var B = new Tensor<double>(new double[] { 5, 6, 7, 8 }, 2, 2);
        var C = A.GpuMatMul(B);
        // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
        C.Span[0].Should().BeApproximately(19, 1e-10);
        C.Span[1].Should().BeApproximately(22, 1e-10);
        C.Span[2].Should().BeApproximately(43, 1e-10);
        C.Span[3].Should().BeApproximately(50, 1e-10);
    }

    [Fact]
    public void GpuInfo_ReturnsDeviceString()
    {
        var info = GpuExtensions.GpuInfo();
        info.Should().NotBeNullOrEmpty();
    }

    // ═══ Large data test ═══

    [Fact]
    public void LargeArray_Add_Correct()
    {
        int n = 100_000;
        var a = new double[n];
        var b = new double[n];
        for (int i = 0; i < n; i++)
        {
            a[i] = i;
            b[i] = i * 2;
        }

        var result = _ops.Add(a, b);
        result.Length.Should().Be(n);
        result[0].Should().Be(0);
        result[n - 1].Should().BeApproximately(n - 1 + (n - 1) * 2, 1e-6);
    }
}
