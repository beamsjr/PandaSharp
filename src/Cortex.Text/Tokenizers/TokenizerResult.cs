namespace Cortex.Text.Tokenizers;

/// <summary>
/// Result of tokenizing a text string.
/// Contains token IDs, attention mask, and optional token-to-word alignment.
/// </summary>
/// <param name="TokenIds">Integer IDs corresponding to each token.</param>
/// <param name="AttentionMask">Binary mask indicating real tokens (1) vs padding (0).</param>
/// <param name="TokenToWord">Optional mapping from token index to original word index.</param>
public record TokenizerResult(
    int[] TokenIds,
    int[] AttentionMask,
    (int TokenIndex, int WordIndex)[]? TokenToWord = null);
