/*
 * PandaSharp Native Accelerators v2
 * ===================================
 * Dictionary-encoded string operations.
 * Instead of operating on N strings, operate on K unique strings + N int codes.
 */

#include <stdint.h>
#include <string.h>

/* ── Dictionary-coded upper/lower ──────────────────────────
 * Only transform K unique strings instead of N total strings.
 * For 14.7M rows with 6K unique tickers, this is 2000x less work.
 */
void dict_upper(const char* packed_uniques, const int* offsets, const int* lengths,
                int n_uniques, char* output)
{
    int total = 0;
    for (int u = 0; u < n_uniques; u++) total += lengths[u];
    for (int i = 0; i < total; i++)
        output[i] = (packed_uniques[i] >= 'a' && packed_uniques[i] <= 'z')
            ? packed_uniques[i] - 32 : packed_uniques[i];
}

void dict_lower(const char* packed_uniques, const int* offsets, const int* lengths,
                int n_uniques, char* output)
{
    int total = 0;
    for (int u = 0; u < n_uniques; u++) total += lengths[u];
    for (int i = 0; i < total; i++)
        output[i] = (packed_uniques[i] >= 'A' && packed_uniques[i] <= 'Z')
            ? packed_uniques[i] + 32 : packed_uniques[i];
}

/* ── Dictionary-coded contains ─────────────────────────────
 * Check which of K unique strings contain needle,
 * then map back to N rows via codes array.
 */
int dict_contains(const char* packed_uniques, const int* offsets, const int* lengths,
                   int n_uniques, const char* needle, int needle_len,
                   const int* codes, int n_rows, uint8_t* mask)
{
    /* First: check each unique string */
    uint8_t* unique_match = (uint8_t*)__builtin_alloca(n_uniques);
    for (int u = 0; u < n_uniques; u++)
    {
        unique_match[u] = 0;
        const char* s = packed_uniques + offsets[u];
        int slen = lengths[u];
        for (int j = 0; j <= slen - needle_len; j++)
        {
            int match = 1;
            for (int k = 0; k < needle_len; k++)
                if (s[j + k] != needle[k]) { match = 0; break; }
            if (match) { unique_match[u] = 1; break; }
        }
    }

    /* Then: map to N rows via codes (just array lookups) */
    int count = 0;
    for (int i = 0; i < n_rows; i++)
    {
        mask[i] = unique_match[codes[i]];
        count += mask[i];
    }
    return count;
}
