/*
 * PandaSharp Native Quantile Selection
 * =====================================
 * Wraps C++ std::nth_element for O(n) quantile computation.
 * Cascaded selection: median first, then Q25 on left, Q75 on right (~2n work).
 *
 * Build (macOS):  clang++ -O3 -std=c++17 -shared -o libpandasharp_quantile.dylib pandasharp_quantile.cpp
 * Build (Linux):  g++ -O3 -std=c++17 -shared -fPIC -o libpandasharp_quantile.so pandasharp_quantile.cpp
 */

#include <algorithm>
#include <cmath>

extern "C" {

/* Select the k-th smallest element, partitioning data[0..n-1] so that:
 *   data[0..k-1] <= data[k] <= data[k+1..n-1]
 * Uses std::nth_element (IntroSelect, O(n) guaranteed).
 */
void nth_element_double(double* data, int n, int k)
{
    if (n <= 0 || k < 0 || k >= n) return;
    std::nth_element(data, data + k, data + n);
}

/* Find the minimum value in data[lo..hi] (inclusive). */
static double find_min_range(const double* data, int lo, int hi)
{
    double mn = data[lo];
    for (int i = lo + 1; i <= hi; i++)
        if (data[i] < mn) mn = data[i];
    return mn;
}

/* Compute interpolated quantile at position p in data[lo..lo+count-1].
 * Uses nth_element for the lower index, then scans for the next value
 * to interpolate between.
 */
static double interpolated_quantile(double* data, int lo, int count, double p)
{
    if (count <= 0) return NAN;
    if (count == 1) return data[lo];

    double idx = p * (count - 1);
    int lower = (int)idx;
    double frac = idx - lower;

    int loIdx = lo + lower;
    std::nth_element(data + lo, data + loIdx, data + lo + count);
    double loVal = data[loIdx];

    if (frac == 0.0 || lower + 1 >= count)
        return loVal;

    /* After nth_element for loIdx, data[loIdx+1..end] are all >= data[loIdx].
     * Find the minimum of that range for interpolation. */
    double hiVal = find_min_range(data, loIdx + 1, lo + count - 1);
    return loVal + frac * (hiVal - loVal);
}

/* Compute Q25, Q50, Q75 with linear interpolation using cascaded selection.
 * Total work: ~2n comparisons (median partitions, then Q25/Q75 on halves).
 * Data is modified in-place. NaN must be pre-filtered by caller.
 * Results written to out[0]=q25, out[1]=q50, out[2]=q75.
 */
void describe_quantiles(double* data, int n, double* out)
{
    if (n <= 0) { out[0] = out[1] = out[2] = NAN; return; }
    if (n == 1) { out[0] = out[1] = out[2] = data[0]; return; }
    if (n == 2) {
        if (data[0] > data[1]) std::swap(data[0], data[1]);
        out[0] = data[0] + 0.25 * (data[1] - data[0]);
        out[1] = data[0] + 0.50 * (data[1] - data[0]);
        out[2] = data[0] + 0.75 * (data[1] - data[0]);
        return;
    }

    /* Step 1: Find median position — partitions entire array */
    double idx50 = 0.50 * (n - 1);
    int lo50 = (int)idx50;
    double frac50 = idx50 - lo50;

    std::nth_element(data, data + lo50, data + n);
    double val50 = data[lo50];

    double next50 = val50;
    if (frac50 > 0.0 && lo50 + 1 < n)
        next50 = find_min_range(data, lo50 + 1, n - 1);
    out[1] = val50 + frac50 * (next50 - val50);

    /* Step 2: Q25 on left partition [0..lo50] — these are all <= median */
    {
        int leftCount = lo50 + 1;
        double idx25 = 0.25 * (n - 1);
        int lo25 = (int)idx25;
        double frac25 = idx25 - lo25;

        if (lo25 < leftCount)
        {
            std::nth_element(data, data + lo25, data + leftCount);
            double val25 = data[lo25];
            double next25 = val25;
            if (frac25 > 0.0 && lo25 + 1 < n)
            {
                /* Next value could be in left or right partition */
                int searchEnd = (lo25 + 1 < leftCount) ? leftCount - 1 : lo50;
                if (lo25 + 1 <= searchEnd)
                    next25 = find_min_range(data, lo25 + 1, searchEnd);
                /* Also check the median itself if it's the next position */
                if (lo25 + 1 == lo50 && val50 < next25)
                    next25 = val50;
            }
            out[0] = val25 + frac25 * (next25 - val25);
        }
        else
        {
            out[0] = val50;
        }
    }

    /* Step 3: Q75 on right partition [lo50..n-1] — these are all >= median */
    {
        double idx75 = 0.75 * (n - 1);
        int lo75 = (int)idx75;
        double frac75 = idx75 - lo75;

        if (lo75 >= lo50 && lo75 < n)
        {
            std::nth_element(data + lo50, data + lo75, data + n);
            double val75 = data[lo75];
            double next75 = val75;
            if (frac75 > 0.0 && lo75 + 1 < n)
                next75 = find_min_range(data, lo75 + 1, n - 1);
            out[2] = val75 + frac75 * (next75 - val75);
        }
        else
        {
            out[2] = data[n - 1];
        }
    }
}

} /* extern "C" */
