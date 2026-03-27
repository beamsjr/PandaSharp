using Cortex;

namespace Cortex.Text.Analytics;

/// <summary>
/// Named Entity Recognition via ONNX model.
/// Identifies entity spans (person, organization, location, etc.) in text.
/// This is a stub that requires an ONNX NER model and compatible tokenizer at runtime.
/// </summary>
public class NamedEntityRecognition
{
    private readonly string _modelPath;
    private readonly string _tokenizerPath;

    /// <summary>
    /// Create a NER predictor backed by an ONNX model.
    /// </summary>
    /// <param name="modelPath">Path to the ONNX NER model file (e.g., bert-ner.onnx).</param>
    /// <param name="tokenizerPath">Path to the tokenizer vocabulary/config (e.g., vocab.txt or tokenizer.json).</param>
    public NamedEntityRecognition(string modelPath, string tokenizerPath)
    {
        _modelPath = modelPath ?? throw new ArgumentNullException(nameof(modelPath));
        _tokenizerPath = tokenizerPath ?? throw new ArgumentNullException(nameof(tokenizerPath));
    }

    /// <summary>
    /// Run NER on a text column and return a DataFrame with entity spans and labels.
    /// </summary>
    /// <param name="df">Input DataFrame containing a text column.</param>
    /// <param name="textColumn">Name of the column containing text to analyze.</param>
    /// <returns>DataFrame with columns: row_index, entity, label, start, end, confidence.</returns>
    /// <exception cref="NotImplementedException">
    /// Always thrown. ONNX NER inference requires Cortex.ML.Onnx and a compatible
    /// NER model (e.g., dslim/bert-base-NER). Install the model and use OnnxScorer
    /// with a token-classification head, or implement this method with your preferred
    /// ONNX Runtime integration.
    /// </exception>
    public DataFrame Predict(DataFrame df, string textColumn)
    {
        throw new NotImplementedException(
            "Named Entity Recognition requires an ONNX NER model and runtime integration. " +
            $"Model path: '{_modelPath}', Tokenizer: '{_tokenizerPath}'. " +
            "To implement: (1) load the ONNX model via Cortex.ML.Onnx.OnnxScorer, " +
            "(2) tokenize the text column using the specified tokenizer, " +
            "(3) run token-classification inference, " +
            "(4) decode BIO/IOB2 tags to entity spans. " +
            "See HuggingFace 'dslim/bert-base-NER' or similar token-classification models.");
    }
}
