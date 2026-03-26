using FluentAssertions;
using PandaSharp.ML.Native;
using PandaSharp.ML.Spatial;
using Xunit;

namespace PandaSharp.ML.Tests.EdgeCases;

public class BlasEdgeCaseRound3Tests
{
    // ═══ BLAS Dgemm with zero-size matrices ═══

    [Fact]
    public void Dgemm_ZeroM_NoOp()
    {
        // m=0: C is 0×n, nothing to compute
        var A = Array.Empty<double>();
        var B = new double[] { 1, 2, 3, 4 };
        var C = Array.Empty<double>();
        // Should not crash
        BlasOps.Dgemm(A, B, C, m: 0, n: 2, k: 2);
    }

    [Fact]
    public void Dgemm_ZeroN_NoOp()
    {
        var A = new double[] { 1, 2, 3, 4 };
        var B = Array.Empty<double>();
        var C = Array.Empty<double>();
        BlasOps.Dgemm(A, B, C, m: 2, n: 0, k: 2);
    }

    [Fact]
    public void Dgemm_ZeroK_ResultIsZero()
    {
        // k=0: no terms in sum, so C = beta * C
        var A = Array.Empty<double>();
        var B = Array.Empty<double>();
        var C = new double[] { 99, 99, 99, 99 };
        BlasOps.Dgemm(A, B, C, m: 2, n: 2, k: 0, alpha: 1.0, beta: 0.0);
        C.Should().AllSatisfy(x => x.Should().Be(0));
    }

    // ═══ BLAS Dgemv with zero-length vector ═══

    [Fact]
    public void Dgemv_ZeroM_NoOp()
    {
        var A = Array.Empty<double>();
        var x = new double[] { 1, 2 };
        var y = Array.Empty<double>();
        BlasOps.Dgemv(A, x, y, m: 0, n: 2);
    }

    [Fact]
    public void Dgemv_ZeroN_ResultIsScaledY()
    {
        // n=0: y = alpha * 0 + beta * y = beta * y
        var A = Array.Empty<double>();
        var x = Array.Empty<double>();
        var y = new double[] { 5, 10 };
        BlasOps.Dgemv(A, x, y, m: 2, n: 0, alpha: 1.0, beta: 2.0);
        y[0].Should().Be(10);
        y[1].Should().Be(20);
    }

    // ═══ BLAS Managed fallback matches native results ═══

    [Fact]
    public void Dgemm_ManagedFallback_MatchesExpected()
    {
        // A = [[1,2],[3,4]] (2×2), B = [[5,6],[7,8]] (2×2)
        // C = A*B = [[19,22],[43,50]]
        var A = new double[] { 1, 2, 3, 4 };
        var B = new double[] { 5, 6, 7, 8 };
        var C = new double[4];
        BlasOps.Dgemm(A, B, C, m: 2, n: 2, k: 2);
        C[0].Should().BeApproximately(19, 1e-10);
        C[1].Should().BeApproximately(22, 1e-10);
        C[2].Should().BeApproximately(43, 1e-10);
        C[3].Should().BeApproximately(50, 1e-10);
    }

    [Fact]
    public void Dgemm_AlphaBeta_Correct()
    {
        // C = 2 * A * B + 3 * C
        var A = new double[] { 1, 0, 0, 1 }; // identity 2×2
        var B = new double[] { 5, 6, 7, 8 };
        var C = new double[] { 1, 1, 1, 1 };
        BlasOps.Dgemm(A, B, C, m: 2, n: 2, k: 2, alpha: 2.0, beta: 3.0);
        // C = 2 * B + 3 * [1,1,1,1] = [10,12,14,16] + [3,3,3,3] = [13,15,17,19]
        C[0].Should().BeApproximately(13, 1e-10);
        C[1].Should().BeApproximately(15, 1e-10);
        C[2].Should().BeApproximately(17, 1e-10);
        C[3].Should().BeApproximately(19, 1e-10);
    }

    [Fact]
    public void Dgemv_ManagedFallback_MatchesExpected()
    {
        // A = [[1,2],[3,4]], x = [1,1], y should be [3, 7]
        var A = new double[] { 1, 2, 3, 4 };
        var x = new double[] { 1, 1 };
        var y = new double[2];
        BlasOps.Dgemv(A, x, y, m: 2, n: 2);
        y[0].Should().BeApproximately(3, 1e-10);
        y[1].Should().BeApproximately(7, 1e-10);
    }

    // ═══ BLAS PairwiseDistances with 0 points ═══

    [Fact]
    public void PairwiseDistances_NX0_NoOp()
    {
        var X = Array.Empty<double>();
        var Y = new double[] { 1, 2, 3, 4 };
        var dist = Array.Empty<double>();
        // Should not crash
        BlasOps.PairwiseDistances(X, Y, dist, nX: 0, nY: 2, d: 2);
    }

    [Fact]
    public void PairwiseDistances_NY0_NoOp()
    {
        var X = new double[] { 1, 2, 3, 4 };
        var Y = Array.Empty<double>();
        var dist = Array.Empty<double>();
        BlasOps.PairwiseDistances(X, Y, dist, nX: 2, nY: 0, d: 2);
    }

    [Fact]
    public void PairwiseDistances_KnownValues()
    {
        // X = [[0,0]], Y = [[3,4]]
        // dist = (3-0)^2 + (4-0)^2 = 25
        var X = new double[] { 0, 0 };
        var Y = new double[] { 3, 4 };
        var dist = new double[1];
        BlasOps.PairwiseDistances(X, Y, dist, nX: 1, nY: 1, d: 2);
        dist[0].Should().BeApproximately(25, 1e-10);
    }

    // ═══ Dgesv with singular matrix ═══

    [Fact]
    public void Dgesv_SingularMatrix_ReturnsFalse()
    {
        // Singular matrix: [[1,2],[2,4]] — row 2 is 2*row 1
        var A = new double[] { 1, 2, 2, 4 };
        var B = new double[] { 1, 2 };
        var result = BlasOps.Dgesv(A, B, n: 2);
        // On macOS with Accelerate, should return false (info != 0 from LAPACK)
        // On non-BLAS platforms, returns false because managed fallback is not implemented
        result.Should().BeFalse("singular matrix should not be solvable");
    }
}

public class KDTreeEdgeCaseRound3Tests
{
    private static (int[] Indices, double[] Distances) BruteForceKnn(
        double[] data, int n, int d, double[] query, int qOff, int k)
    {
        var dists = new (double Dist, int Idx)[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            for (int j = 0; j < d; j++)
            {
                double diff = query[qOff + j] - data[i * d + j];
                sum += diff * diff;
            }
            dists[i] = (Math.Sqrt(sum), i);
        }
        Array.Sort(dists, (a, b) => a.Dist.CompareTo(b.Dist));
        k = Math.Min(k, n);
        var indices = new int[k];
        var distances = new double[k];
        for (int i = 0; i < k; i++)
        {
            indices[i] = dists[i].Idx;
            distances[i] = dists[i].Dist;
        }
        return (indices, distances);
    }

    // ═══ KDTree with 2 points (minimal tree) ═══

    [Fact]
    public void KDTree_TwoPoints_FindsBoth()
    {
        // 2 points in 2D
        var data = new double[] { 0, 0, 10, 10 };
        var tree = new KDTree(data, n: 2, d: 2);

        var query = new double[] { 1, 1 };
        var idx = new int[2];
        var dist = new double[2];
        tree.KnnQuery(query, 0, 2, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, 2, 2, query, 0, 2);
        for (int i = 0; i < 2; i++)
            dist[i].Should().BeApproximately(bfDist[i], 1e-10);
    }

    // ═══ KDTree with all points at same location ═══

    [Fact]
    public void KDTree_AllSameLocation_AllDistancesZero()
    {
        int n = 10, d = 3;
        var data = new double[n * d];
        for (int i = 0; i < n * d; i++) data[i] = 5.0; // all points at (5,5,5)

        var tree = new KDTree(data, n, d);

        // Query at the same location
        var query = new double[] { 5, 5, 5 };
        var idx = new int[5];
        var dist = new double[5];
        tree.KnnQuery(query, 0, 5, idx, dist);

        // All distances should be 0
        for (int i = 0; i < 5; i++)
            dist[i].Should().BeApproximately(0, 1e-10);
    }

    // ═══ KDTree k=n returns ALL points ═══

    [Fact]
    public void KDTree_KEqualsN_SmallSet_ReturnsAllSorted()
    {
        int n = 5, d = 2;
        var data = new double[] { 0, 0, 1, 0, 0, 1, 1, 1, 0.5, 0.5 };

        var tree = new KDTree(data, n, d);

        var query = new double[] { 0.5, 0.5 };
        var idx = new int[n];
        var dist = new double[n];
        tree.KnnQuery(query, 0, n, idx, dist);

        var (bfIdx, bfDist) = BruteForceKnn(data, n, d, query, 0, n);
        for (int i = 0; i < n; i++)
            dist[i].Should().BeApproximately(bfDist[i], 1e-10);

        // All n indices should be present
        idx.Order().Should().Equal(Enumerable.Range(0, n));
    }

    // ═══ KDTree with NaN — should not crash ═══

    [Fact]
    public void KDTree_NaNInData_DoesNotCrash()
    {
        // NaN in coordinates is a degenerate case. The tree should not crash,
        // though results may be unpredictable.
        var data = new double[] { 0, 0, double.NaN, 1, 2, 3 };

        var act = () =>
        {
            var tree = new KDTree(data, n: 3, d: 2);
            var idx = new int[2];
            var dist = new double[2];
            tree.KnnQuery(new double[] { 1, 1 }, 0, 2, idx, dist);
        };

        // Should not throw — it may return garbage distances but should not crash
        act.Should().NotThrow();
    }

    // ═══ KDTree with 1 point ═══

    [Fact]
    public void KDTree_SinglePoint_ReturnsIt()
    {
        var data = new double[] { 3, 4 };
        var tree = new KDTree(data, n: 1, d: 2);

        var query = new double[] { 0, 0 };
        var idx = new int[1];
        var dist = new double[1];
        tree.KnnQuery(query, 0, 1, idx, dist);

        idx[0].Should().Be(0);
        dist[0].Should().BeApproximately(5.0, 1e-10); // sqrt(9+16) = 5
    }
}
