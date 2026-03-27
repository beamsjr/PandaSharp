namespace Cortex.Text.Tokenizers;

/// <summary>
/// Common interface for all tokenizers.
/// </summary>
public interface ITokenizer
{
    /// <summary>Encode text into token IDs with attention mask.</summary>
    TokenizerResult Encode(string text);

    /// <summary>Decode token IDs back into text.</summary>
    string Decode(int[] ids);
}
