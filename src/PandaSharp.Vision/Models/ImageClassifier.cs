using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PandaSharp.Column;
using PandaSharp.Vision.Transforms;

namespace PandaSharp.Vision.Models;

/// <summary>
/// End-to-end image classification using ONNX models.
/// Handles preprocessing, inference, and top-K prediction extraction.
/// </summary>
public class ImageClassifier : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int[] _inputShape;
    private readonly string[] _classNames;
    private readonly ImagePipeline _preprocessing;
    private readonly bool _inputIsNchw;

    public ImageClassifier(string modelPath, string[] classNames, ImagePipeline? preprocessing = null)
    {
        _session = new InferenceSession(modelPath);
        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;
        _inputShape = inputMeta.Value.Dimensions;
        _inputIsNchw = _inputShape.Length == 4 && (_inputShape[1] == 3 || _inputShape[1] == 1);
        _classNames = classNames;
        _preprocessing = preprocessing ?? ImageEmbedder.DefaultPreprocessing();
    }

    /// <summary>Number of classes.</summary>
    public int NumClasses => _classNames.Length;

    /// <summary>Class name labels.</summary>
    public string[] ClassNames => _classNames;

    /// <summary>
    /// Classify images in a DataFrame. Adds prediction and confidence columns.
    /// </summary>
    public DataFrame Predict(DataFrame df, string imagePathColumn,
        string outputColumn = "prediction", int batchSize = 32)
    {
        var pathCol = df.GetStringColumn(imagePathColumn);
        var predictions = new string?[df.RowCount];
        var confidences = new double[df.RowCount];

        for (int start = 0; start < df.RowCount; start += batchSize)
        {
            int end = Math.Min(start + batchSize, df.RowCount);
            ClassifyBatchInto(pathCol, start, end, predictions, confidences);
        }

        return df.AddColumn(new StringColumn(outputColumn, predictions))
                 .AddColumn(new Column<double>($"{outputColumn}_confidence", confidences));
    }

    /// <summary>
    /// Classify with top-K predictions. Returns DataFrame with columns for each rank.
    /// </summary>
    public DataFrame PredictTopK(DataFrame df, string imagePathColumn, int k = 5, int batchSize = 32)
    {
        var pathCol = df.GetStringColumn(imagePathColumn);
        var topPreds = new string?[k][];
        var topConfs = new double[k][];
        for (int rank = 0; rank < k; rank++)
        {
            topPreds[rank] = new string?[df.RowCount];
            topConfs[rank] = new double[df.RowCount];
        }

        for (int start = 0; start < df.RowCount; start += batchSize)
        {
            int end = Math.Min(start + batchSize, df.RowCount);
            var logits = RunBatch(pathCol, start, end);

            int numClasses = logits.GetLength(1);
            for (int i = 0; i < logits.GetLength(0); i++)
            {
                // Softmax
                float maxVal = float.MinValue;
                for (int c = 0; c < numClasses; c++)
                    if (logits[i, c] > maxVal) maxVal = logits[i, c];
                float sumExp = 0;
                var probs = new float[numClasses];
                for (int c = 0; c < numClasses; c++)
                {
                    probs[c] = MathF.Exp(logits[i, c] - maxVal);
                    sumExp += probs[c];
                }
                for (int c = 0; c < numClasses; c++) probs[c] /= sumExp;

                // Top-K
                var indices = Enumerable.Range(0, numClasses).OrderByDescending(c => probs[c]).Take(k).ToArray();
                for (int rank = 0; rank < Math.Min(k, numClasses); rank++)
                {
                    int idx = indices[rank];
                    topPreds[rank][start + i] = idx < _classNames.Length ? _classNames[idx] : $"class_{idx}";
                    topConfs[rank][start + i] = probs[idx];
                }
            }
        }

        var result = df;
        for (int rank = 0; rank < k; rank++)
        {
            string suffix = rank == 0 ? "" : $"_{rank + 1}";
            result = result.AddColumn(new StringColumn($"prediction{suffix}", topPreds[rank]))
                           .AddColumn(new Column<double>($"confidence{suffix}", topConfs[rank]));
        }
        return result;
    }

    /// <summary>
    /// Classify a batch and write results directly into pre-allocated output arrays,
    /// avoiding intermediate string[] and double[] allocations per batch.
    /// </summary>
    private void ClassifyBatchInto(StringColumn pathCol, int start, int end,
        string?[] predictions, double[] confidences)
    {
        var logits = RunBatch(pathCol, start, end);
        int count = end - start;
        int numClasses = logits.GetLength(1);

        for (int i = 0; i < count; i++)
        {
            int bestIdx = 0;
            float bestVal = logits[i, 0];
            for (int c = 1; c < numClasses; c++)
            {
                if (logits[i, c] > bestVal) { bestVal = logits[i, c]; bestIdx = c; }
            }

            // Softmax for confidence
            float maxVal = bestVal;
            float sumExp = 0;
            for (int c = 0; c < numClasses; c++)
                sumExp += MathF.Exp(logits[i, c] - maxVal);
            confidences[start + i] = 1.0 / sumExp;
            predictions[start + i] = bestIdx < _classNames.Length ? _classNames[bestIdx] : $"class_{bestIdx}";
        }
    }

    private float[,] RunBatch(StringColumn pathCol, int start, int end)
    {
        int count = end - start;
        var batchImages = new ImageTensor[count];

        // Parallel image load + preprocess
        Parallel.For(0, count, i =>
        {
            try
            {
                var img = ImageIO.Load(pathCol[start + i]!);
                batchImages[i] = _preprocessing.Transform(img);
            }
            catch (Exception)
            {
                batchImages[i] = new ImageTensor(new float[224 * 224 * 3], 224, 224, 3);
            }
        });

        int h = batchImages[0].Height, w = batchImages[0].Width, c = batchImages[0].Channels;
        int singleLen = h * w * c;

        // Build NCHW tensor
        var nchwData = new float[count * c * h * w];
        for (int i = 0; i < count; i++)
        {
            var src = batchImages[i].Span;
            if (_inputIsNchw)
            {
                for (int ch = 0; ch < c; ch++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            nchwData[i * c * h * w + ch * h * w + y * w + x] = src[(y * w + x) * c + ch];
            }
            else
            {
                src.CopyTo(nchwData.AsSpan(i * singleLen, singleLen));
            }
        }

        var inputDims = _inputIsNchw ? new[] { count, c, h, w } : new[] { count, h, w, c };
        var inputTensor = new DenseTensor<float>(nchwData, inputDims);
        var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor(_inputName, inputTensor) };

        using var results = _session.Run(inputs);
        var output = results.First();

        if (output.Value is DenseTensor<float> floatOutput)
        {
            int outClasses = floatOutput.Dimensions.Length > 1 ? floatOutput.Dimensions[1] : 1;
            var logits = new float[count, outClasses];
            for (int i = 0; i < count; i++)
                for (int j = 0; j < outClasses; j++)
                    logits[i, j] = floatOutput[i, j];
            return logits;
        }

        throw new InvalidOperationException($"Unsupported output type: {output.Value?.GetType().Name}");
    }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}
