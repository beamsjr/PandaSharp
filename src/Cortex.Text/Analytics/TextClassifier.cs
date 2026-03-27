using Cortex;

namespace Cortex.Text.Analytics;

/// <summary>
/// Text classification via ONNX model (sentiment, topic, intent, etc.).
/// This is a stub that requires an ONNX classification model and compatible tokenizer at runtime.
/// </summary>
public class TextClassifier
{
    private readonly string _modelPath;
    private readonly string _tokenizerPath;

    /// <summary>
    /// Create a text classifier backed by an ONNX model.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX classification model file (e.g., sentiment-model.onnx).</param>
    /// <param name="tokenizerPath">Path to the tokenizer vocabulary/config (e.g., vocab.txt or tokenizer.json).</param>
    public TextClassifier(string modelPath, string tokenizerPath)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _tokenizerPath = tokenizerPath ?? throw new ArgumentNullException(nameof(tokenizerPath));
    }

    /// <summary>
    /// Classify text in the specified column and return predictions.
    /// </summary>
    /// <param name="df">Input DataFrame containing a text column.</param>
    /// <param name="textColumn">Name of the column containing text to classify.</param>
    /// <returns>DataFrame with columns: row_index, label, confidence.</returns>
    /// <exception cref="NotImplementedException">
    /// Always thrown. ONNX text classification requires Cortex.ML.Onnx and a compatible
    /// sequence-classification model. Install the model and use OnnxScorer with a
    /// sequence-classification head, or implement this method with your preferred
    /// ONNX Runtime integration.
    /// </exception>
    public DataFrame Predict(DataFrame df, string textColumn)
    {
        throw new NotImplementedException(
            "Text classification requires an ONNX classification model and runtime integration. " +
            $"Model path: '{_modelPath}', Tokenizer: '{_tokenizerPath}'. " +
            "To implement: (1) load the ONNX model via Cortex.ML.Onnx.OnnxScorer, " +
            "(2) tokenize the text column using the specified tokenizer, " +
            "(3) run sequence-classification inference, " +
            "(4) map logits to class labels via softmax. " +
            "See HuggingFace 'distilbert-base-uncased-finetuned-sst-2-english' or similar models.");
    }
}
