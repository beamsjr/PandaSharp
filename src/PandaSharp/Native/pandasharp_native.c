/*
 * PandaSharp Native Accelerators
 * ================================
 * High-performance C implementations for hot paths.
 * Compiled as a shared library, called via P/Invoke from C#.
 *
 * Build (macOS):  clang -O3 -shared -o libpandasharp.dylib pandasharp_native.c
 * Build (Linux):  gcc -O3 -shared -fPIC -o libpandasharp.so pandasharp_native.c
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

/* ── Rolling Mean ──────────────────────────────────────────
 * O(n) sliding window mean using running sum.
 * No branches in the hot loop. Handles NaN propagation.
 */
void rolling_mean(const double* input, double* output, int length, int window)
{
    if (length <= 0 || window <= 0) return;

    double sum = 0.0;
    int count = 0;

    for (int i = 0; i < length; i++)
    {
        /* Add new element */
        if (!isnan(input[i]))
        {
            sum += input[i];
            count++;
        }

        /* Remove element leaving the window */
        if (i >= window)
        {
            if (!isnan(input[i - window]))
            {
                sum -= input[i - window];
                count--;
            }
        }

        /* Output */
        if (i < window - 1 || count == 0)
            output[i] = NAN;
        else
            output[i] = sum / count;
    }
}

/* ── Expanding Mean ────────────────────────────────────────
 * Cumulative mean in a single pass.
 */
void expanding_mean(const double* input, double* output, int length)
{
    double sum = 0.0;
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        if (!isnan(input[i]))
        {
            sum += input[i];
            count++;
        }
        output[i] = count > 0 ? sum / count : NAN;
    }
}

/* ── Sum (double) ──────────────────────────────────────────
 * Kahan compensated summation for better precision on large arrays.
 */
double sum_double(const double* data, int length)
{
    double sum = 0.0;
    double c = 0.0; /* compensation */
    for (int i = 0; i < length; i++)
    {
        double y = data[i] - c;
        double t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    return sum;
}

/* ── Filter: generate boolean mask from double comparison ──
 * Returns count of true values.
 */
int filter_gt_double(const double* data, int length, double threshold, uint8_t* mask)
{
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        uint8_t v = data[i] > threshold ? 1 : 0;
        mask[i] = v;
        count += v;
    }
    return count;
}

int filter_lt_double(const double* data, int length, double threshold, uint8_t* mask)
{
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        uint8_t v = data[i] < threshold ? 1 : 0;
        mask[i] = v;
        count += v;
    }
    return count;
}

int filter_abs_gt_double(const double* data, int length, double threshold, uint8_t* mask)
{
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        double abs_val = data[i] < 0 ? -data[i] : data[i];
        uint8_t v = abs_val > threshold ? 1 : 0;
        mask[i] = v;
        count += v;
    }
    return count;
}

/* ── Arithmetic: element-wise on double arrays ─────────────
 * Compiler will auto-vectorize with -O3.
 */
void add_arrays(const double* a, const double* b, double* out, int length)
{
    for (int i = 0; i < length; i++) out[i] = a[i] + b[i];
}

void sub_arrays(const double* a, const double* b, double* out, int length)
{
    for (int i = 0; i < length; i++) out[i] = a[i] - b[i];
}

void mul_arrays(const double* a, const double* b, double* out, int length)
{
    for (int i = 0; i < length; i++) out[i] = a[i] * b[i];
}

void div_arrays(const double* a, const double* b, double* out, int length)
{
    for (int i = 0; i < length; i++)
        out[i] = b[i] != 0.0 ? a[i] / b[i] : NAN;
}

void mul_scalar(const double* a, double scalar, double* out, int length)
{
    for (int i = 0; i < length; i++) out[i] = a[i] * scalar;
}

/* ── Eval: (close - open) / open * 100 in one pass ────────
 * Fused operation avoids 3 separate array passes.
 */
void eval_daily_return(const double* close, const double* open, double* out, int length)
{
    for (int i = 0; i < length; i++)
        out[i] = open[i] != 0.0 ? (close[i] - open[i]) / open[i] * 100.0 : NAN;
}

/* ── Fused: (high - low) / close * 100 ─────────────────────*/
void eval_spread(const double* high, const double* low, const double* close, double* out, int length)
{
    for (int i = 0; i < length; i++)
        out[i] = close[i] != 0.0 ? (high[i] - low[i]) / close[i] * 100.0 : NAN;
}

/* ── Complex filter: ret > a AND vol > b AND close > c ─────
 * Fused 3-predicate filter in a single pass.
 */
int filter_complex(const double* ret, const double* vol, const double* close,
                    int length, double minRet, double minVol, double minClose, uint8_t* mask)
{
    int count = 0;
    for (int i = 0; i < length; i++)
    {
        uint8_t v = (ret[i] > minRet && vol[i] > minVol && close[i] > minClose) ? 1 : 0;
        mask[i] = v;
        count += v;
    }
    return count;
}

/* ── Multi-aggregation: compute sum+count+min+max+mean+std in ONE pass ─
 * For each group, compute all aggregates simultaneously.
 * Avoids iterating group indices 7 separate times.
 */
void multi_agg_double(const double* data, const int* group_ids, int n_rows, int n_groups,
                       double* sums, int* counts, double* mins, double* maxs,
                       double* means, double* stds)
{
    /* Initialize */
    for (int g = 0; g < n_groups; g++)
    {
        sums[g] = 0; counts[g] = 0;
        mins[g] = 1e308; maxs[g] = -1e308;
    }

    /* Single pass: accumulate sum, count, min, max */
    for (int i = 0; i < n_rows; i++)
    {
        int g = group_ids[i];
        double v = data[i];
        sums[g] += v;
        counts[g]++;
        if (v < mins[g]) mins[g] = v;
        if (v > maxs[g]) maxs[g] = v;
    }

    /* Compute means */
    for (int g = 0; g < n_groups; g++)
        means[g] = counts[g] > 0 ? sums[g] / counts[g] : 0;

    /* Second pass: compute variance for std */
    double* sum_sq = stds; /* reuse stds buffer temporarily */
    for (int g = 0; g < n_groups; g++) sum_sq[g] = 0;

    for (int i = 0; i < n_rows; i++)
    {
        int g = group_ids[i];
        double d = data[i] - means[g];
        sum_sq[g] += d * d;
    }

    for (int g = 0; g < n_groups; g++)
        stds[g] = counts[g] > 1 ? sqrt(sum_sq[g] / (counts[g] - 1)) : 0;
}

/* ── Dedup hash for 2 string columns ───────────────────────
 * Computes a combined 64-bit hash for each (str1, str2) pair.
 * Uses the pre-computed hash codes passed from managed code.
 */
void dedup_hash_2str(const int* hash1, const int* hash2, int n_rows, int64_t* combined)
{
    for (int i = 0; i < n_rows; i++)
        combined[i] = ((int64_t)hash1[i] << 32) | (uint32_t)hash2[i];
}

/* ── Map column via index array (for JoinMany) ─────────────
 * output[i] = source[rowMap[i]] — auto-vectorized with -O3.
 */
void map_column_double(const double* source, const int* rowMap, double* output, int n_rows)
{
    for (int i = 0; i < n_rows; i++)
        output[i] = rowMap[i] >= 0 ? source[rowMap[i]] : 0;
}

void map_column_int(const int* source, const int* rowMap, int* output, int n_rows)
{
    for (int i = 0; i < n_rows; i++)
        output[i] = rowMap[i] >= 0 ? source[rowMap[i]] : 0;
}

/* ── GroupBy: count per string hash bucket ─────────────────
 * Hash strings and return bucket assignments.
 * Uses FNV-1a for fast string hashing.
 */
static uint32_t fnv1a(const char* str, int len)
{
    uint32_t hash = 2166136261u;
    for (int i = 0; i < len; i++)
    {
        hash ^= (uint8_t)str[i];
        hash *= 16777619u;
    }
    return hash;
}

void hash_strings(const char** strings, const int* lengths, int count,
                   int num_buckets, int* bucket_assignments)
{
    for (int i = 0; i < count; i++)
    {
        uint32_t h = fnv1a(strings[i], lengths[i]);
        bucket_assignments[i] = (int)(h % (uint32_t)num_buckets);
    }
}

/* ── String operations on UTF-8 byte arrays ────────────────
 * Process strings in bulk without per-string managed overhead.
 */

/* toupper: ASCII fast path (covers 99% of stock tickers) */
void str_upper(const char* input, char* output, int total_bytes)
{
    for (int i = 0; i < total_bytes; i++)
        output[i] = (input[i] >= 'a' && input[i] <= 'z') ? input[i] - 32 : input[i];
}

void str_lower(const char* input, char* output, int total_bytes)
{
    for (int i = 0; i < total_bytes; i++)
        output[i] = (input[i] >= 'A' && input[i] <= 'Z') ? input[i] + 32 : input[i];
}

/* str_contains: check if each string contains a substring.
 * strings: packed null-terminated strings
 * offsets: start offset of each string in the packed buffer
 * needle: substring to search for
 */
int str_contains_packed(const char* packed, const int* offsets, const int* lengths,
                         int count, const char* needle, int needle_len, uint8_t* mask)
{
    int matches = 0;
    for (int i = 0; i < count; i++)
    {
        const char* s = packed + offsets[i];
        int slen = lengths[i];
        int found = 0;
        for (int j = 0; j <= slen - needle_len; j++)
        {
            int match = 1;
            for (int k = 0; k < needle_len; k++)
            {
                if (s[j + k] != needle[k]) { match = 0; break; }
            }
            if (match) { found = 1; break; }
        }
        mask[i] = found;
        matches += found;
    }
    return matches;
}

/* ── Typed aggregation: sum/mean/min/max for group indices ──
 * Avoids per-element boxing in managed GroupBy aggregation.
 */
void agg_sum_by_group(const double* data, const int* group_ids, int n_rows,
                       double* group_sums, int n_groups)
{
    for (int g = 0; g < n_groups; g++) group_sums[g] = 0.0;
    for (int i = 0; i < n_rows; i++)
        group_sums[group_ids[i]] += data[i];
}

void agg_count_by_group(const int* group_ids, int n_rows,
                          int* group_counts, int n_groups)
{
    for (int g = 0; g < n_groups; g++) group_counts[g] = 0;
    for (int i = 0; i < n_rows; i++)
        group_counts[group_ids[i]]++;
}

void agg_min_by_group(const double* data, const int* group_ids, int n_rows,
                       double* group_mins, int n_groups)
{
    for (int g = 0; g < n_groups; g++) group_mins[g] = 1e308;
    for (int i = 0; i < n_rows; i++)
    {
        int g = group_ids[i];
        if (data[i] < group_mins[g]) group_mins[g] = data[i];
    }
}

void agg_max_by_group(const double* data, const int* group_ids, int n_rows,
                       double* group_maxs, int n_groups)
{
    for (int g = 0; g < n_groups; g++) group_maxs[g] = -1e308;
    for (int i = 0; i < n_rows; i++)
    {
        int g = group_ids[i];
        if (data[i] > group_maxs[g]) group_maxs[g] = data[i];
    }
}

/* ── Cast int array to double ──────────────────────────────*/
void cast_int_to_double(const int* input, double* output, int length)
{
    for (int i = 0; i < length; i++) output[i] = (double)input[i];
}

/* ── Dedup hash: compute 64-bit hash per row for N columns ─
 * Each column is a double array. Returns hash per row.
 */
void dedup_hash_doubles(const double** columns, int n_cols, int n_rows, int64_t* hashes)
{
    for (int r = 0; r < n_rows; r++)
    {
        int64_t h = 17;
        for (int c = 0; c < n_cols; c++)
        {
            /* Mix bits of double into hash */
            union { double d; int64_t i; } u;
            u.d = columns[c][r];
            h = h * 31 + u.i;
        }
        hashes[r] = h;
    }
}

/* ── Matrix multiply: C = X^T * X (upper triangle only) ─────────
 * X is n×k row-major. C is k×k row-major (only upper triangle filled).
 * Used for wide correlation matrices.
 * Tiled for cache efficiency, auto-vectorized with -O3.
 */
void gram_matrix_upper(const double* X, int n, int k, double* C)
{
    int tile = 64;
    /* Zero output */
    for (int i = 0; i < k * k; i++) C[i] = 0.0;

    /* Tiled X^T * X */
    for (int ci0 = 0; ci0 < k; ci0 += tile)
    {
        int ci1 = ci0 + tile < k ? ci0 + tile : k;
        for (int cj0 = ci0; cj0 < k; cj0 += tile)
        {
            int cj1 = cj0 + tile < k ? cj0 + tile : k;
            for (int r = 0; r < n; r++)
            {
                const double* xr = X + r * k;
                for (int ci = ci0; ci < ci1; ci++)
                {
                    double xi = xr[ci];
                    int cjStart = cj0 > ci ? cj0 : ci;
                    for (int cj = cjStart; cj < cj1; cj++)
                        C[ci * k + cj] += xi * xr[cj];
                }
            }
        }
    }
}
