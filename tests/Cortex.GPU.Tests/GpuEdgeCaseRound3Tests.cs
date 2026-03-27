using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.GPU;
using Cortex.ML.Tensors;

namespace Cortex.GPU.Tests;

public class GpuEdgeCaseRound3Tests : IDisposable
{
    private readonly GpuOps _ops;

    public GpuEdgeCaseRound3Tests()
    {
        _ops = new GpuOps();
    }

    public void Dispose() { }

    // ═══ Bug 1: SumMinMax on empty array should throw or return sensible defaults ═══

    [Fact]
    public void SumMinMax_EmptyArray_ThrowsArgumentException()
    {
        // SumMinMax with empty array has no guard: it tries to allocate 0-length GPU buffer
        // and returns (0, double.MaxValue, double.MinValue) as sentinel values leak through.
        // It should throw or return (0, NaN, NaN) or similar.
        var act = () => _ops.SumMinMax(Array.Empty<double>());
        act.Should().Throw<ArgumentException>();
    }

    // ═══ Bug 2: Variance on empty array — division by zero ═══

    [Fact]
    public void Variance_EmptyArray_ThrowsArgumentException()
    {
        // Variance does result[0] / n where n=0 → returns NaN (or may crash on GPU allocation).
        // Should throw instead.
        var act = () => _ops.Variance(Array.Empty<double>(), 0.0);
        act.Should().Throw<ArgumentException>();
    }

    // ═══ Bug 3: MatMul with zero dimensions ═══

    [Fact]
    public void MatMul_ZeroDimension_M0_ReturnsEmpty()
    {
        // m=0: C should be 0×n = empty. Currently tries to create 0-size GPU buffer.
        var result = _ops.MatMul(Array.Empty<double>(), new double[] { 1, 2 }, m: 0, k: 1, n: 2);
        result.Should().BeEmpty();
    }

    [Fact]
    public void MatMul_ZeroDimension_N0_ReturnsEmpty()
    {
        var result = _ops.MatMul(new double[] { 1, 2 }, Array.Empty<double>(), m: 2, k: 1, n: 0);
        result.Should().BeEmpty();
    }

    [Fact]
    public void MatMul_ZeroDimension_K0_ReturnsZeros()
    {
        // m=2, n=2, k=0: C = 2×2 all zeros (no terms to sum)
        var result = _ops.MatMul(Array.Empty<double>(), Array.Empty<double>(), m: 2, k: 0, n: 2);
        result.Should().HaveCount(4);
        result.Should().AllSatisfy(x => x.Should().Be(0));
    }

    // ═══ Bug 4: GramMatrix with zero dimensions ═══

    [Fact]
    public void GramMatrix_D0_ReturnsEmpty()
    {
        // d=0: result should be 0×0 = empty
        var result = _ops.GramMatrix(Array.Empty<double>(), n: 5, d: 0);
        result.Should().BeEmpty();
    }

    [Fact]
    public void GramMatrix_N0_ReturnsZeros()
    {
        // n=0: A has 0 rows. A^T * A = d×d all zeros.
        var result = _ops.GramMatrix(Array.Empty<double>(), n: 0, d: 3);
        result.Should().HaveCount(9);
        result.Should().AllSatisfy(x => x.Should().Be(0));
    }

    // ═══ Bug 5: PairwiseDistances with zero points ═══

    [Fact]
    public void PairwiseDistances_NX0_ReturnsEmpty()
    {
        var result = _ops.PairwiseDistances(Array.Empty<double>(), new double[] { 1, 2 }, nX: 0, nY: 1, d: 2);
        result.Should().BeEmpty();
    }

    [Fact]
    public void PairwiseDistances_NY0_ReturnsEmpty()
    {
        var result = _ops.PairwiseDistances(new double[] { 1, 2 }, Array.Empty<double>(), nX: 1, nY: 0, d: 2);
        result.Should().BeEmpty();
    }

    // ═══ Bug 6: GpuVariance extension with empty column ═══

    [Fact]
    public void GpuVariance_EmptyColumn_ThrowsOrReturnsNaN()
    {
        // GpuVariance does mean /= data.Length (Length=0) → NaN
        // Then calls Ops.Variance which has the division-by-zero bug.
        var col = new Column<double>("empty", Array.Empty<double>());
        var act = () => col.GpuVariance();
        act.Should().Throw<Exception>();
    }

    // ═══ Bug 7: GpuCorr with 1 row — division by (n-1) = 0 ═══

    [Fact]
    public void GpuCorr_SingleRow_DiagonalShouldBeNaN()
    {
        // With 1 row, stds = sqrt(ss / (n-1)) = sqrt(0/0) = NaN
        // The correlation matrix diagonal should be NaN (undefined for single sample).
        // Currently produces NaN from (Inf * Inf * 0) which propagates unpredictably.
        // At minimum, this should not crash and should handle the edge case.
        var df = new DataFrame(
            new Column<double>("a", [5.0]),
            new Column<double>("b", [10.0])
        );

        var act = () => df.GpuCorr();
        // This should not crash — it should return a reasonable result
        var result = act();

        // The correlation diagonal for a single observation should be NaN
        // (sample correlation is undefined with n=1), not some random value.
        // With the current bug, stds[c] = sqrt(0/0) = NaN, and
        // denom = NaN * NaN * 0 = NaN, so NaN > 0 is false and result is 0.
        // The diagonal should be NaN (undefined) not 0.
        var aCol = result.GetColumn<double>("a");
        aCol[0].Should().Be(double.NaN, "correlation of single observation is undefined");
    }

    // ═══ Variance single element should be 0 ═══

    [Fact]
    public void Variance_SingleElement_ReturnsZero()
    {
        var data = new double[] { 42.0 };
        double mean = 42.0;
        var variance = _ops.Variance(data, mean);
        variance.Should().Be(0.0);
    }

    // ═══ SumMinMax single element ═══

    [Fact]
    public void SumMinMax_SingleElement_ReturnsThatElement()
    {
        var data = new double[] { 7.5 };
        var (sum, min, max) = _ops.SumMinMax(data);
        sum.Should().BeApproximately(7.5, 1e-10);
        min.Should().Be(7.5);
        max.Should().Be(7.5);
    }
}
