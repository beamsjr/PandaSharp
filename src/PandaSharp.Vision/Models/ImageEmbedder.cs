using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using PandaSharp.Column;
using PandaSharp.Vision.Transforms;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PandaSharp.Vision.Models;

/// <summary>
/// High-level image feature extraction via ONNX models.
/// Handles preprocessing, batching, and embedding extraction.
/// </summary>
public class ImageEmbedder : IDisposable
{
    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly int[] _inputShape; // e.g., [-1, 3, 224, 224] for NCHW
    private readonly ImagePipeline _preprocessing;
    private readonly bool _inputIsNchw;

    /// <summary>Load an ONNX model for feature extraction.</summary>
    /// <param name="modelPath">Path to the .onnx model file.</param>
    /// <param name="preprocessing">Image preprocessing pipeline. If null, uses default (Resize 224 + ImageNet normalize).</param>
    public ImageEmbedder(string modelPath, ImagePipeline? preprocessing = null)
    {
        _session = new InferenceSession(modelPath);
        var inputMeta = _session.InputMetadata.First();
        _inputName = inputMeta.Key;
        _inputShape = inputMeta.Value.Dimensions;
        // Most vision models use NCHW format
        _inputIsNchw = _inputShape.Length == 4 && (_inputShape[1] == 3 || _inputShape[1] == 1);
        _preprocessing = preprocessing ?? DefaultPreprocessing();
    }

    /// <summary>Default preprocessing: Resize(224,224) + Normalize(ImageNet).</summary>
    public static ImagePipeline DefaultPreprocessing() =>
        ImagePipeline.Create()
            .Resize(224, 224)
            .Normalize(Normalize.ImageNet())
            .Build();

    /// <summary>ResNet50 preprocessing: Resize(256) -> CenterCrop(224) -> Normalize(ImageNet).</summary>
    public static ImagePipeline ResNet50Preprocessing() =>
        ImagePipeline.Create()
            .Resize(256, 256)
            .CenterCrop(224, 224)
            .Normalize(Normalize.ImageNet())
            .Build();

    /// <summary>MobileNet preprocessing: Resize(224) -> Normalize(ImageNet).</summary>
    public static ImagePipeline MobileNetPreprocessing() =>
        ImagePipeline.Create()
            .Resize(224, 224)
            .Normalize(Normalize.ImageNet())
            .Build();

    /// <summary>Extract embeddings from preprocessed images.</summary>
    public ML.Tensors.Tensor<float> Embed(ImageTensor images)
    {
        var batch = images.IsBatch ? images : images.ToBatch();
        int n = batch.BatchSize;
        int h = batch.Height, w = batch.Width, c = batch.Channels;

        // Convert from NHWC [N,H,W,C] to NCHW [N,C,H,W] for ONNX
        var nchwData = new float[n * c * h * w];
        var src = batch.Span;

        if (_inputIsNchw)
        {
            for (int i = 0; i < n; i++)
            {
                int srcOff = i * h * w * c;
                for (int ch = 0; ch < c; ch++)
                    for (int y = 0; y < h; y++)
                        for (int x = 0; x < w; x++)
                            nchwData[i * c * h * w + ch * h * w + y * w + x] = src[srcOff + (y * w + x) * c + ch];
            }
        }
        else
        {
            // NHWC input - just copy
            src.CopyTo(nchwData);
        }

        // Create ONNX tensor
        var inputDims = _inputIsNchw ? new[] { n, c, h, w } : new[] { n, h, w, c };
        var inputTensor = new DenseTensor<float>(nchwData, inputDims);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(_inputName, inputTensor)
        };

        // Run inference
        using var results = _session.Run(inputs);
        var output = results.First();

        // Extract embedding tensor
        if (output.Value is DenseTensor<float> floatOutput)
        {
            var dims = floatOutput.Dimensions.ToArray();
            var data = new float[floatOutput.Length];
            for (int i = 0; i < data.Length; i++)
                data[i] = floatOutput.GetValue(i);
            return new ML.Tensors.Tensor<float>(data, dims);
        }

        throw new InvalidOperationException($"Unsupported ONNX output type: {output.Value?.GetType().Name}");
    }

    /// <summary>
    /// Extract embeddings for images referenced in a DataFrame column.
    /// Adds an embedding column (comma-separated float values) to the DataFrame.
    /// </summary>
    public DataFrame EmbedColumn(DataFrame df, string imagePathColumn,
        string outputColumn = "embedding", int batchSize = 32)
    {
        var pathCol = df.GetStringColumn(imagePathColumn);
        var embeddings = new string?[df.RowCount];

        for (int start = 0; start < df.RowCount; start += batchSize)
        {
            int end = Math.Min(start + batchSize, df.RowCount);
            int count = end - start;

            // Parallel load and preprocess batch
            var batchImages = new ImageTensor[count];
            Parallel.For(0, count, idx =>
            {
                try
                {
                    var img = ImageIO.Load(pathCol[start + idx]!);
                    batchImages[idx] = _preprocessing.Transform(img);
                }
                catch (Exception)
                {
                    int ph = 224, pw = 224;
                    batchImages[idx] = new ImageTensor(new float[ph * pw * 3], ph, pw, 3);
                }
            });

            // Stack into batch tensor
            int bh = batchImages[0].Height, bw = batchImages[0].Width, bc = batchImages[0].Channels;
            int singleLen = bh * bw * bc;
            var batchData = new float[count * singleLen];
            for (int i = 0; i < count; i++)
                batchImages[i].Span.CopyTo(batchData.AsSpan(i * singleLen, singleLen));
            var batchTensor = new ImageTensor(new ML.Tensors.Tensor<float>(batchData, count, bh, bw, bc));

            // Get embeddings
            var emb = Embed(batchTensor);

            // Store as comma-separated strings
            int embDim = emb.Length / batchImages.Length;
            for (int i = 0; i < batchImages.Length; i++)
            {
                var vals = new float[embDim];
                for (int j = 0; j < embDim; j++)
                    vals[j] = emb.Span[i * embDim + j];
                embeddings[start + i] = string.Join(",", vals.Select(v => v.ToString("G6")));
            }
        }

        return df.AddColumn(new StringColumn(outputColumn, embeddings));
    }

    private bool _disposed;

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}
