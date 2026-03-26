using System.Runtime.CompilerServices;

namespace PandaSharp.ML.Spatial;

/// <summary>
/// KD-tree for efficient nearest neighbor search in multidimensional space.
/// Balanced construction via median partitioning. Stored as implicit binary tree
/// in flat arrays for cache-friendly traversal.
/// </summary>
internal sealed class KDTree
{
    private readonly double[] _data;       // flattened N*D training data (row-major)
    private readonly int _n;               // number of points
    private readonly int _d;               // dimensionality
    private readonly int[] _indices;        // permuted point indices (tree structure)
    private int[] _splitDims;      // split dimension per internal node (-1 for leaves)
    private double[] _splitVals;   // split value per internal node
    private readonly int _leafSize;

    // Node layout: implicit binary tree stored as ranges over _indices.
    // We store the tree via recursive build and record split info in parallel arrays
    // indexed by node id. Max nodes = 4*N (upper bound for implicit tree).

    private int _nodeCount;

    // Per-node info
    private int[] _nodeStart;    // start index in _indices
    private int[] _nodeEnd;      // end index (exclusive) in _indices
    private int[] _nodeLeft;     // left child node id (-1 if leaf)
    private int[] _nodeRight;    // right child node id (-1 if leaf)

    /// <summary>
    /// Build a KD-tree from flattened row-major data.
    /// </summary>
    /// <param name="data">Flattened N*D array of training points.</param>
    /// <param name="n">Number of points.</param>
    /// <param name="d">Dimensionality.</param>
    /// <param name="leafSize">Maximum points in a leaf node before brute-force (default 32).</param>
    public KDTree(double[] data, int n, int d, int leafSize = 32)
    {
        _data = data;
        _n = n;
        _d = d;
        _leafSize = Math.Max(1, leafSize);

        _indices = new int[n];
        for (int i = 0; i < n; i++) _indices[i] = i;

        // Preallocate node arrays. A balanced binary tree on N leaves needs ~2N nodes.
        int maxNodes = Math.Max(4, 4 * n / _leafSize + 4);
        _splitDims = new int[maxNodes];
        _splitVals = new double[maxNodes];
        _nodeStart = new int[maxNodes];
        _nodeEnd = new int[maxNodes];
        _nodeLeft = new int[maxNodes];
        _nodeRight = new int[maxNodes];
        _nodeCount = 0;

        BuildRecursive(0, n, 0);

        // Trim oversized arrays to actual node count
        Array.Resize(ref _splitDims, _nodeCount);
        Array.Resize(ref _splitVals, _nodeCount);
        Array.Resize(ref _nodeStart, _nodeCount);
        Array.Resize(ref _nodeEnd, _nodeCount);
        Array.Resize(ref _nodeLeft, _nodeCount);
        Array.Resize(ref _nodeRight, _nodeCount);
    }

    private int BuildRecursive(int start, int end, int depth)
    {
        int nodeId = _nodeCount++;

        // Grow arrays if needed
        if (nodeId >= _splitDims.Length)
        {
            int newSize = _splitDims.Length * 2;
            Array.Resize(ref _splitDims, newSize);
            Array.Resize(ref _splitVals, newSize);
            Array.Resize(ref _nodeStart, newSize);
            Array.Resize(ref _nodeEnd, newSize);
            Array.Resize(ref _nodeLeft, newSize);
            Array.Resize(ref _nodeRight, newSize);
        }

        _nodeStart[nodeId] = start;
        _nodeEnd[nodeId] = end;
        int count = end - start;

        if (count <= _leafSize)
        {
            // Leaf node
            _splitDims[nodeId] = -1;
            _splitVals[nodeId] = 0;
            _nodeLeft[nodeId] = -1;
            _nodeRight[nodeId] = -1;
            return nodeId;
        }

        // Choose split dimension: use dimension with max spread (max - min) for better balance
        int bestDim = 0;
        double bestSpread = double.NegativeInfinity;
        for (int dim = 0; dim < _d; dim++)
        {
            double minVal = double.MaxValue;
            double maxVal = double.MinValue;
            for (int i = start; i < end; i++)
            {
                double v = _data[_indices[i] * _d + dim];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
            }
            double spread = maxVal - minVal;
            if (spread > bestSpread)
            {
                bestSpread = spread;
                bestDim = dim;
            }
        }

        // Partition around median using nth_element-style partitioning
        int mid = (start + end) / 2;
        NthElement(start, end, mid, bestDim);

        double splitVal = _data[_indices[mid] * _d + bestDim];

        _splitDims[nodeId] = bestDim;
        _splitVals[nodeId] = splitVal;

        // Recursively build children
        _nodeLeft[nodeId] = BuildRecursive(start, mid, depth + 1);
        _nodeRight[nodeId] = BuildRecursive(mid, end, depth + 1);

        return nodeId;
    }

    /// <summary>
    /// Partial sort _indices[start..end) so that the element at position nth
    /// is in its correct sorted position (by the given dimension).
    /// Uses introselect (quickselect with fallback).
    /// </summary>
    private void NthElement(int start, int end, int nth, int dim)
    {
        while (end - start > 1)
        {
            // Choose pivot as median of three
            int mid = (start + end) / 2;
            double vStart = _data[_indices[start] * _d + dim];
            double vMid = _data[_indices[mid] * _d + dim];
            double vEnd = _data[_indices[end - 1] * _d + dim];

            // Sort the three to find median, put it at start
            int pivotIdx;
            if ((vStart <= vMid && vMid <= vEnd) || (vEnd <= vMid && vMid <= vStart))
                pivotIdx = mid;
            else if ((vMid <= vStart && vStart <= vEnd) || (vEnd <= vStart && vStart <= vMid))
                pivotIdx = start;
            else
                pivotIdx = end - 1;

            // Move pivot to start
            (_indices[start], _indices[pivotIdx]) = (_indices[pivotIdx], _indices[start]);
            double pivotVal = _data[_indices[start] * _d + dim];

            // Partition
            int lo = start + 1;
            int hi = end - 1;
            while (lo <= hi)
            {
                while (lo <= hi && _data[_indices[lo] * _d + dim] <= pivotVal) lo++;
                while (lo <= hi && _data[_indices[hi] * _d + dim] > pivotVal) hi--;
                if (lo < hi)
                {
                    (_indices[lo], _indices[hi]) = (_indices[hi], _indices[lo]);
                    lo++;
                    hi--;
                }
            }

            // Place pivot in its final position
            (_indices[start], _indices[hi]) = (_indices[hi], _indices[start]);

            if (hi == nth) return;
            if (nth < hi) end = hi;
            else start = hi + 1;
        }
    }

    /// <summary>
    /// Find the K nearest neighbors of a query point.
    /// </summary>
    /// <param name="query">Flattened query array (may contain multiple queries).</param>
    /// <param name="queryOffset">Offset into query array for this query point.</param>
    /// <param name="k">Number of neighbors to find.</param>
    /// <param name="outIndices">Output array for neighbor indices (length >= k).</param>
    /// <param name="outDistances">Output array for neighbor distances (length >= k).</param>
    public void KnnQuery(double[] query, int queryOffset, int k, int[] outIndices, double[] outDistances)
    {
        k = Math.Min(k, _n);

        // Max-heap of size k: we maintain the K closest points seen so far.
        // The root of the max-heap is the farthest of the K closest.
        // We use squared Euclidean distances throughout to avoid sqrt.
        var heapDist = new double[k];
        var heapIdx = new int[k];
        int heapSize = 0;

        SearchRecursive(0, query, queryOffset, k, heapDist, heapIdx, ref heapSize);

        // Extract results sorted by distance (ascending)
        // First, sort the heap by distance
        var pairs = new (double Dist, int Idx)[heapSize];
        for (int i = 0; i < heapSize; i++)
            pairs[i] = (heapDist[i], heapIdx[i]);
        Array.Sort(pairs, (a, b) => a.Dist.CompareTo(b.Dist));

        for (int i = 0; i < heapSize; i++)
        {
            outDistances[i] = Math.Sqrt(Math.Max(0, pairs[i].Dist)); // convert squared -> actual distance
            outIndices[i] = pairs[i].Idx;
        }
    }

    private void SearchRecursive(int nodeId, double[] query, int qOff, int k,
        double[] heapDist, int[] heapIdx, ref int heapSize)
    {
        int start = _nodeStart[nodeId];
        int end = _nodeEnd[nodeId];

        if (_splitDims[nodeId] == -1)
        {
            // Leaf node: brute-force check all points
            for (int i = start; i < end; i++)
            {
                int ptIdx = _indices[i];
                double sqDist = SquaredDistance(query, qOff, ptIdx);
                HeapInsert(heapDist, heapIdx, ref heapSize, k, sqDist, ptIdx);
            }
            return;
        }

        int splitDim = _splitDims[nodeId];
        double splitVal = _splitVals[nodeId];
        double queryVal = query[qOff + splitDim];
        double diff = queryVal - splitVal;

        // Visit the closer child first
        int nearChild, farChild;
        if (diff <= 0)
        {
            nearChild = _nodeLeft[nodeId];
            farChild = _nodeRight[nodeId];
        }
        else
        {
            nearChild = _nodeRight[nodeId];
            farChild = _nodeLeft[nodeId];
        }

        SearchRecursive(nearChild, query, qOff, k, heapDist, heapIdx, ref heapSize);

        // Prune: only visit far child if the splitting plane is closer than our current K-th nearest
        double sqDiff = diff * diff;
        if (heapSize < k || sqDiff < heapDist[0])
        {
            SearchRecursive(farChild, query, qOff, k, heapDist, heapIdx, ref heapSize);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private double SquaredDistance(double[] query, int qOff, int pointIndex)
    {
        double sum = 0;
        int pOff = pointIndex * _d;
        for (int i = 0; i < _d; i++)
        {
            double d = query[qOff + i] - _data[pOff + i];
            sum += d * d;
        }
        return sum;
    }

    /// <summary>
    /// Insert into a max-heap of fixed capacity k.
    /// The heap stores the K smallest squared distances seen so far.
    /// heapDist[0] is always the largest (max-heap root).
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void HeapInsert(double[] heapDist, int[] heapIdx, ref int size, int k,
        double sqDist, int pointIdx)
    {
        if (size < k)
        {
            // Heap not full yet: insert and sift up
            heapDist[size] = sqDist;
            heapIdx[size] = pointIdx;
            size++;
            SiftUp(heapDist, heapIdx, size - 1);
        }
        else if (sqDist < heapDist[0])
        {
            // Replace the root (largest) with the new smaller value
            heapDist[0] = sqDist;
            heapIdx[0] = pointIdx;
            SiftDown(heapDist, heapIdx, 0, size);
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiftUp(double[] heapDist, int[] heapIdx, int i)
    {
        while (i > 0)
        {
            int parent = (i - 1) / 2;
            if (heapDist[i] > heapDist[parent])
            {
                (heapDist[i], heapDist[parent]) = (heapDist[parent], heapDist[i]);
                (heapIdx[i], heapIdx[parent]) = (heapIdx[parent], heapIdx[i]);
                i = parent;
            }
            else break;
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void SiftDown(double[] heapDist, int[] heapIdx, int i, int size)
    {
        while (true)
        {
            int left = 2 * i + 1;
            int right = 2 * i + 2;
            int largest = i;

            if (left < size && heapDist[left] > heapDist[largest])
                largest = left;
            if (right < size && heapDist[right] > heapDist[largest])
                largest = right;

            if (largest == i) break;

            (heapDist[i], heapDist[largest]) = (heapDist[largest], heapDist[i]);
            (heapIdx[i], heapIdx[largest]) = (heapIdx[largest], heapIdx[i]);
            i = largest;
        }
    }
}
