using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.ML.Onnx;

/// <summary>
/// Run ONNX model inference on PandaSharp DataFrames.
/// Load a model once, score DataFrames repeatedly.
/// </summary>
public class OnnxScorer : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int _inputWidth;

    /// <summary>Load an ONNX model from a file path.</summary>
    public OnnxScorer(string modelPath)
    {
        _session = new InferenceSession(modelPath);
        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;
        _inputWidth = inputMeta.Value.Dimensions.Length > 1 ? inputMeta.Value.Dimensions[1] : 1;
    }

    /// <summary>Load an ONNX model from bytes.</summary>
    public OnnxScorer(byte[] modelBytes)
    {
        _session = new InferenceSession(modelBytes);
        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;
        _inputWidth = inputMeta.Value.Dimensions.Length > 1 ? inputMeta.Value.Dimensions[1] : 1;
    }

    /// <summary>
    /// Run inference on a DataFrame. Returns a new DataFrame with prediction columns added.
    /// </summary>
    public DataFrame Predict(DataFrame df, string[] inputColumns, string outputName = "prediction")
    {
        int rows = df.RowCount;
        int cols = inputColumns.Length;

        // Build input tensor
        var inputData = new float[rows * cols];
        for (int c = 0; c < cols; c++)
        {
            var col = df[inputColumns[c]];
            for (int r = 0; r < rows; r++)
                inputData[r * cols + c] = col.IsNull(r) ? 0f : Convert.ToSingle(col.GetObject(r));
        }

        var inputTensor = new DenseTensor<float>(inputData, [rows, cols]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First();

        // Extract predictions
        var predictions = ExtractPredictions(output, rows);

        return df.AddColumn(new Column<double>(outputName,
            predictions.Select(p => (double)p).ToArray()));
    }

    /// <summary>
    /// Run batched inference for large DataFrames.
    /// </summary>
    public DataFrame PredictBatched(DataFrame df, string[] inputColumns,
        string outputName = "prediction", int batchSize = 1024)
    {
        var allPredictions = new List<double>();

        for (int start = 0; start < df.RowCount; start += batchSize)
        {
            int end = Math.Min(start + batchSize, df.RowCount);
            int batchLen = end - start;
            int cols = inputColumns.Length;

            var inputData = new float[batchLen * cols];
            for (int c = 0; c < cols; c++)
            {
                var col = df[inputColumns[c]];
                for (int r = 0; r < batchLen; r++)
                {
                    int srcRow = start + r;
                    inputData[r * cols + c] = col.IsNull(srcRow) ? 0f : Convert.ToSingle(col.GetObject(srcRow));
                }
            }

            var inputTensor = new DenseTensor<float>(inputData, [batchLen, cols]);
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
            };

            using var results = _session.Run(inputs);
            var output = results.First();
            var preds = ExtractPredictions(output, batchLen);
            allPredictions.AddRange(preds.Select(p => (double)p));
        }

        return df.AddColumn(new Column<double>(outputName, allPredictions.ToArray()));
    }

    private static float[] ExtractPredictions(DisposableNamedOnnxValue output, int expectedRows)
    {
        if (output.Value is DenseTensor<float> floatTensor)
        {
            var result = new float[expectedRows];
            bool is2D = floatTensor.Dimensions.Length > 1 && floatTensor.Dimensions[1] >= 1;
            for (int i = 0; i < expectedRows; i++)
                result[i] = is2D ? floatTensor[i, 0] : floatTensor[i];
            return result;
        }

        if (output.Value is DenseTensor<long> longTensor)
        {
            var result = new float[expectedRows];
            for (int i = 0; i < expectedRows; i++)
                result[i] = longTensor[i];
            return result;
        }

        throw new InvalidOperationException($"Unsupported ONNX output type: {output.Value?.GetType().Name}");
    }

    /// <summary>Model input/output metadata.</summary>
    public IReadOnlyDictionary<string, NodeMetadata> InputMetadata => _session.InputMetadata;
    public IReadOnlyDictionary<string, NodeMetadata> OutputMetadata => _session.OutputMetadata;

    public void Dispose() => _session.Dispose();
}
