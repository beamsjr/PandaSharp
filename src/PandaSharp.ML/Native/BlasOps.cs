using System.Runtime.InteropServices;

namespace PandaSharp.ML.Native;

/// <summary>
/// P/Invoke bindings to BLAS/LAPACK via Apple Accelerate framework.
/// All methods have managed fallbacks for non-macOS platforms.
/// </summary>
internal static class BlasOps
{
    private const string Accelerate = "/System/Library/Frameworks/Accelerate.framework/Accelerate";

    private static volatile bool _checked;
    private static volatile bool _available;

    internal static bool IsAvailable
    {
        get
        {
            if (!_checked)
            {
                _checked = true;
                try
                {
                    // Probe by calling a cheap function
                    cblas_ddot(0, IntPtr.Zero, 1, IntPtr.Zero, 1);
                    _available = true;
                }
                catch { _available = false; }
            }
            return _available;
        }
    }

    // CBLAS
    [DllImport(Accelerate)] private static extern double cblas_ddot(int n, IntPtr x, int incx, IntPtr y, int incy);

    [DllImport(Accelerate)] private static extern void cblas_dgemm(
        int order, int transA, int transB,
        int m, int n, int k,
        double alpha, IntPtr A, int lda,
        IntPtr B, int ldb,
        double beta, IntPtr C, int ldc);

    [DllImport(Accelerate)] private static extern void cblas_dgemv(
        int order, int trans,
        int m, int n,
        double alpha, IntPtr A, int lda,
        IntPtr x, int incx,
        double beta, IntPtr y, int incy);

    [DllImport(Accelerate)] private static extern void cblas_dsyrk(
        int order, int uplo, int trans,
        int n, int k,
        double alpha, IntPtr A, int lda,
        double beta, IntPtr C, int ldc);

    // LAPACK - linear solve (LU factorization + solve)
    [DllImport(Accelerate, EntryPoint = "dgesv_")]
    private static extern void dgesv_(ref int n, ref int nrhs, IntPtr A, ref int lda, IntPtr ipiv, IntPtr B, ref int ldb, ref int info);

    // LAPACK - SVD
    [DllImport(Accelerate, EntryPoint = "dgesvd_")]
    private static extern void dgesvd_(ref byte jobu, ref byte jobvt, ref int m, ref int n,
        IntPtr A, ref int lda, IntPtr S, IntPtr U, ref int ldu, IntPtr VT, ref int ldvt,
        IntPtr work, ref int lwork, ref int info);

    // ═══ Public API with managed fallbacks ═══

    /// <summary>
    /// Matrix multiply: C = alpha * A * B + beta * C
    /// A is m×k, B is k×n, C is m×n (all row-major)
    /// </summary>
    internal static unsafe void Dgemm(double[] A, double[] B, double[] C, int m, int n, int k,
        double alpha = 1.0, double beta = 0.0)
    {
        if (IsAvailable)
        {
            fixed (double* pA = A, pB = B, pC = C)
            {
                // Row major = 101, NoTrans = 111
                cblas_dgemm(101, 111, 111, m, n, k, alpha, (IntPtr)pA, k, (IntPtr)pB, n, beta, (IntPtr)pC, n);
            }
            return;
        }
        // Managed fallback
        ManagedDgemm(A, B, C, m, n, k, alpha, beta);
    }

    /// <summary>
    /// Matrix-vector multiply: y = alpha * A * x + beta * y
    /// A is m×n (row-major), x is length n, y is length m
    /// </summary>
    internal static unsafe void Dgemv(double[] A, double[] x, double[] y, int m, int n,
        double alpha = 1.0, double beta = 0.0)
    {
        if (IsAvailable)
        {
            fixed (double* pA = A, pX = x, pY = y)
            {
                cblas_dgemv(101, 111, m, n, alpha, (IntPtr)pA, n, (IntPtr)pX, 1, beta, (IntPtr)pY, 1);
            }
            return;
        }
        // Managed fallback
        for (int i = 0; i < m; i++)
        {
            double sum = 0;
            int off = i * n;
            for (int j = 0; j < n; j++)
                sum += A[off + j] * x[j];
            y[i] = alpha * sum + beta * y[i];
        }
    }

    /// <summary>
    /// Matrix^T-vector multiply: y = alpha * A^T * x + beta * y
    /// A is m×n (row-major), x is length m, y is length n
    /// Computes y = alpha * A^T * x + beta * y
    /// </summary>
    internal static unsafe void DgemvT(double[] A, double[] x, double[] y, int m, int n,
        double alpha = 1.0, double beta = 0.0)
    {
        if (IsAvailable)
        {
            fixed (double* pA = A, pX = x, pY = y)
            {
                // Trans = 112
                cblas_dgemv(101, 112, m, n, alpha, (IntPtr)pA, n, (IntPtr)pX, 1, beta, (IntPtr)pY, 1);
            }
            return;
        }
        // Managed fallback
        for (int j = 0; j < n; j++)
            y[j] *= beta;
        for (int i = 0; i < m; i++)
        {
            int off = i * n;
            double xi = alpha * x[i];
            for (int j = 0; j < n; j++)
                y[j] += xi * A[off + j];
        }
    }

    /// <summary>
    /// Solve A*X = B for X using LU factorization.
    /// A is n×n, B is n×nrhs. Solution overwrites B.
    /// A is destroyed. Returns true on success.
    /// </summary>
    internal static unsafe bool Dgesv(double[] A, double[] B, int n, int nrhs = 1)
    {
        if (IsAvailable)
        {
            // LAPACK expects column-major. Transpose A and B in-place.
            TransposeInPlace(A, n, n);
            if (nrhs > 1) TransposeInPlace(B, n, nrhs);

            var ipiv = new int[n];
            int info = 0;
            fixed (double* pA = A, pB = B)
            fixed (int* pIpiv = ipiv)
            {
                dgesv_(ref n, ref nrhs, (IntPtr)pA, ref n, (IntPtr)pIpiv, (IntPtr)pB, ref n, ref info);
            }

            // Transpose B back to row-major
            if (nrhs > 1) TransposeInPlace(B, nrhs, n);
            return info == 0;
        }
        return false; // caller should use Gauss-Jordan fallback
    }

    /// <summary>
    /// Compute thin SVD: A = U * diag(S) * VT
    /// A is m×n. S has min(m,n) singular values.
    /// U is m×min(m,n), VT is min(m,n)×n.
    /// A is destroyed.
    /// </summary>
    internal static unsafe bool Dgesvd(double[] A, double[] S, double[] U, double[] VT, int m, int n)
    {
        if (!IsAvailable) return false;

        // LAPACK expects column-major
        TransposeInPlace(A, m, n);

        int mn = Math.Min(m, n);
        byte jobu = (byte)'S';  // thin U
        byte jobvt = (byte)'S'; // thin VT
        int lda = m;
        int ldu = m;
        int ldvt = mn;
        int info = 0;

        // Query optimal work size
        int lwork = -1;
        double workQuery = 0;
        fixed (double* pA = A, pS = S, pU = U, pVT = VT)
        {
            dgesvd_(ref jobu, ref jobvt, ref m, ref n,
                (IntPtr)pA, ref lda, (IntPtr)pS, (IntPtr)pU, ref ldu, (IntPtr)pVT, ref ldvt,
                (IntPtr)(&workQuery), ref lwork, ref info);
        }

        lwork = (int)workQuery;
        var work = new double[lwork];

        fixed (double* pA = A, pS = S, pU = U, pVT = VT, pWork = work)
        {
            dgesvd_(ref jobu, ref jobvt, ref m, ref n,
                (IntPtr)pA, ref lda, (IntPtr)pS, (IntPtr)pU, ref ldu, (IntPtr)pVT, ref ldvt,
                (IntPtr)pWork, ref lwork, ref info);
        }

        if (info != 0) return false;

        // Transpose U and VT back to row-major
        // U is m×mn in column-major → transpose to row-major
        TransposeInPlace(U, mn, m); // col-major m×mn stored as mn×m after transpose
        // Actually U col-major is m rows × mn cols → need m×mn row-major
        // VT is mn×n in column-major → transpose to row-major
        TransposeInPlace(VT, n, mn);

        return true;
    }

    /// <summary>
    /// Compute all pairwise squared Euclidean distances using BLAS.
    /// dist[i,j] = ||X[i] - Y[j]||^2 = ||X[i]||^2 + ||Y[j]||^2 - 2*X[i]·Y[j]
    /// X is nX×d, Y is nY×d. Output dist is nX×nY.
    /// </summary>
    internal static void PairwiseDistances(double[] X, double[] Y, double[] dist, int nX, int nY, int d)
    {
        // ||x-y||^2 = ||x||^2 + ||y||^2 - 2*x·y
        // Compute -2 * X @ Y^T using BLAS with transB
        if (IsAvailable)
        {
            unsafe
            {
                fixed (double* pX = X, pY = Y, pD = dist)
                {
                    // C = alpha * A * B^T + beta * C
                    // A=X (nX×d), B=Y (nY×d), transB means B^T is (d×nY)
                    // Result C is nX×nY
                    cblas_dgemm(101, 111, 112 /*Trans*/, nX, nY, d, -2.0, (IntPtr)pX, d, (IntPtr)pY, d, 0.0, (IntPtr)pD, nY);
                }
            }
        }
        else
        {
            // Managed: dist = -2 * X @ Y^T
            for (int i = 0; i < nX; i++)
            {
                int offI = i * d;
                int dstOff = i * nY;
                for (int j = 0; j < nY; j++)
                {
                    double dot = 0;
                    int offJ = j * d;
                    for (int k = 0; k < d; k++)
                        dot += X[offI + k] * Y[offJ + k];
                    dist[dstOff + j] = -2.0 * dot;
                }
            }
        }

        // Add ||X[i]||^2 and ||Y[j]||^2
        var normX = new double[nX];
        var normY = new double[nY];
        for (int i = 0; i < nX; i++)
        {
            double s = 0;
            int off = i * d;
            for (int k = 0; k < d; k++) s += X[off + k] * X[off + k];
            normX[i] = s;
        }
        for (int j = 0; j < nY; j++)
        {
            double s = 0;
            int off = j * d;
            for (int k = 0; k < d; k++) s += Y[off + k] * Y[off + k];
            normY[j] = s;
        }

        for (int i = 0; i < nX; i++)
        {
            int off = i * nY;
            for (int j = 0; j < nY; j++)
                dist[off + j] += normX[i] + normY[j];
        }
    }

    // ═══ Helpers ═══

    private static void ManagedDgemm(double[] A, double[] B, double[] C, int m, int n, int k,
        double alpha, double beta)
    {
        for (int i = 0; i < m; i++)
        {
            int cOff = i * n;
            int aOff = i * k;
            for (int j = 0; j < n; j++)
            {
                double sum = 0;
                for (int l = 0; l < k; l++)
                    sum += A[aOff + l] * B[l * n + j];
                C[cOff + j] = alpha * sum + beta * C[cOff + j];
            }
        }
    }

    private static void TransposeInPlace(double[] data, int rows, int cols)
    {
        if (rows == cols)
        {
            // Square: swap in-place
            for (int i = 0; i < rows; i++)
                for (int j = i + 1; j < cols; j++)
                    (data[i * cols + j], data[j * rows + i]) = (data[j * rows + i], data[i * cols + j]);
        }
        else
        {
            // Non-square: need temp buffer
            var temp = new double[rows * cols];
            Array.Copy(data, temp, rows * cols);
            for (int i = 0; i < rows; i++)
                for (int j = 0; j < cols; j++)
                    data[j * rows + i] = temp[i * cols + j];
        }
    }
}
