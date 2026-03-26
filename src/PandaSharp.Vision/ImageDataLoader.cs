using PandaSharp.ML.Tensors;
using PandaSharp.Vision.Transforms;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System.Collections;

namespace PandaSharp.Vision;

/// <summary>
/// DataFrame-driven image batch loader. Yields (ImageTensor images, Tensor&lt;int&gt; labels) batches.
/// Loads and decodes images on-the-fly, applies augmentation pipeline per image.
/// </summary>
public class ImageDataLoader : IEnumerable<(ImageTensor Images, Tensor<int> Labels)>
{
    private readonly string[] _paths;
    private readonly int[] _labelIndices;
    private readonly int _batchSize;
    private readonly bool _shuffle;
    private readonly int? _seed;
    private readonly ImagePipeline? _augmentation;
    private readonly int _resizeW;
    private readonly int _resizeH;
    private int _epochCount;

    /// <summary>Number of batches per epoch.</summary>
    public int BatchCount => (_paths.Length + _batchSize - 1) / _batchSize;

    /// <summary>Total number of images in the dataset.</summary>
    public int TotalImages => _paths.Length;

    /// <summary>Sorted unique class names discovered from the label column.</summary>
    public string[] ClassNames { get; }

    /// <summary>
    /// Create a data loader from a DataFrame with image paths and labels.
    /// </summary>
    /// <param name="df">DataFrame containing the dataset.</param>
    /// <param name="pathColumn">Name of the column containing image file paths.</param>
    /// <param name="labelColumn">Name of the column containing class labels.</param>
    /// <param name="batchSize">Number of images per batch.</param>
    /// <param name="shuffle">Whether to shuffle indices each epoch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="augmentation">Optional augmentation pipeline applied per image.</param>
    /// <param name="resizeWidth">Target width for loaded images.</param>
    /// <param name="resizeHeight">Target height for loaded images.</param>
    public ImageDataLoader(DataFrame df, string pathColumn, string labelColumn,
        int batchSize = 32, bool shuffle = true, int? seed = null,
        ImagePipeline? augmentation = null, int resizeWidth = 224, int resizeHeight = 224)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(pathColumn);
        ArgumentNullException.ThrowIfNull(labelColumn);
        var pathCol = df.GetStringColumn(pathColumn);
        _paths = new string[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            _paths[i] = pathCol[i] ?? "";

        // Encode labels as integer indices
        var labelCol = df[labelColumn];
        var uniqueLabels = new SortedSet<string>();
        for (int i = 0; i < df.RowCount; i++)
            uniqueLabels.Add(labelCol.GetObject(i)?.ToString() ?? "");
        ClassNames = uniqueLabels.ToArray();
        var labelMap = new Dictionary<string, int>();
        for (int i = 0; i < ClassNames.Length; i++)
            labelMap[ClassNames[i]] = i;
        _labelIndices = new int[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            _labelIndices[i] = labelMap.GetValueOrDefault(labelCol.GetObject(i)?.ToString() ?? "", 0);

        _batchSize = batchSize;
        _shuffle = shuffle;
        _seed = seed;
        _augmentation = augmentation;
        _resizeW = resizeWidth;
        _resizeH = resizeHeight;
    }

    /// <summary>
    /// Enumerate batches for one epoch. Each batch is a tuple of (images, labels).
    /// Shuffling uses a different permutation each epoch when enabled.
    /// </summary>
    public IEnumerator<(ImageTensor Images, Tensor<int> Labels)> GetEnumerator()
    {
        var indices = Enumerable.Range(0, _paths.Length).ToArray();
        if (_shuffle)
        {
            var rng = _seed.HasValue ? new Random(_seed.Value + _epochCount) : new Random();
            for (int i = indices.Length - 1; i > 0; i--)
            {
                int j = rng.Next(i + 1);
                (indices[i], indices[j]) = (indices[j], indices[i]);
            }
        }
        Interlocked.Increment(ref _epochCount);

        for (int batch = 0; batch < BatchCount; batch++)
        {
            int start = batch * _batchSize;
            int end = Math.Min(start + _batchSize, _paths.Length);
            int count = end - start;

            var batchData = new float[count * _resizeH * _resizeW * 3];
            var batchLabels = new int[count];
            int singleLen = _resizeH * _resizeW * 3;

            Parallel.For(0, count, i =>
            {
                int idx = indices[start + i];
                batchLabels[i] = _labelIndices[idx];
                try
                {
                    using var img = Image.Load<Rgb24>(_paths[idx]);
                    if (img.Width != _resizeW || img.Height != _resizeH)
                        img.Mutate(x => x.Resize(_resizeW, _resizeH));
                    var tensor = ImageTensor.FromImage(img);
                    if (_augmentation is not null)
                        tensor = _augmentation.Transform(tensor);
                    var data = tensor.Span;
                    int copyLen = Math.Min(singleLen, data.Length);
                    data.Slice(0, copyLen).CopyTo(batchData.AsSpan(i * singleLen, copyLen));
                }
                catch (Exception)
                {
                    // Corrupt image — leave as zeros
                }
            });

            var images = new ImageTensor(new Tensor<float>(batchData, count, _resizeH, _resizeW, 3));
            var labels = new Tensor<int>(batchLabels, count);
            yield return (images, labels);
        }
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
