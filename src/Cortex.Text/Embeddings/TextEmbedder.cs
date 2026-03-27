using Cortex.ML.Tensors;
using Cortex.Text.Tokenizers;

namespace Cortex.Text.Embeddings;

/// <summary>
/// Stub for ONNX-based sentence embedding models.
/// Provides a unified interface for generating text embeddings from transformer models.
/// Actual inference requires ONNX Runtime, which will be integrated via Cortex.ML.Onnx.
/// </summary>
public sealed class TextEmbedder
{
    private readonly string _modelPath;
    private readonly ITokenizer _tokenizer;

    /// <summary>
    /// Creates a text embedder with a specified ONNX model and tokenizer.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX model file.</param>
    /// <param name="tokenizer">Tokenizer to use for text preprocessing.</param>
    public TextEmbedder(string modelPath, ITokenizer tokenizer)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
    }

    /// <summary>
    /// Generate embeddings for an array of texts.
    /// </summary>
    /// <param name="texts">Input texts to embed.</param>
    /// <returns>2D tensor of shape [texts.Length, embeddingDim].</returns>
    /// <exception cref="NotSupportedException">
    /// Always thrown — ONNX Runtime integration is required for inference.
    /// Install Cortex.ML.Onnx and provide a model file.
    /// </exception>
    public Tensor<float> Embed(string[] texts)
    {
        throw new NotSupportedException(
            "TextEmbedder.Embed() requires ONNX Runtime for model inference. " +
            "Install Cortex.ML.Onnx and use OnnxModelRunner to load the model at: " + _modelPath);
    }

    /// <summary>
    /// Create a preset embedder configured for the MiniLM-L6 sentence transformer model.
    /// </summary>
    /// <returns>A configured TextEmbedder.</returns>
    /// <exception cref="NotSupportedException">
    /// Always thrown with download instructions.
    /// </exception>
    public static TextEmbedder MiniLM()
    {
        throw new NotSupportedException(
            "MiniLM preset requires downloading the ONNX model. " +
            "Download 'all-MiniLM-L6-v2' from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 " +
            "and use: new TextEmbedder(\"path/to/model.onnx\", tokenizer)");
    }

    /// <summary>
    /// Create a preset embedder configured for the E5 embedding model.
    /// </summary>
    /// <returns>A configured TextEmbedder.</returns>
    /// <exception cref="NotSupportedException">
    /// Always thrown with download instructions.
    /// </exception>
    public static TextEmbedder E5()
    {
        throw new NotSupportedException(
            "E5 preset requires downloading the ONNX model. " +
            "Download 'e5-small-v2' from https://huggingface.co/intfloat/e5-small-v2 " +
            "and use: new TextEmbedder(\"path/to/model.onnx\", tokenizer)");
    }
}
