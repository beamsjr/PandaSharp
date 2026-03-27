using Cortex.ML.Tensors;
using Cortex.Vision.Transforms;
using Cortex.Column;
using System.Collections;

namespace Cortex.Vision.Video;

/// <summary>
/// Batched video frame data loader. Given a DataFrame with video paths and labels,
/// extracts frames from each video and yields batches of (ImageTensor, Tensor&lt;int&gt;).
/// </summary>
public class VideoFrameDataLoader : IEnumerable<(ImageTensor Frames, Tensor<int> Labels)>
{
    private readonly string[] _videoPaths;
    private readonly int[] _labels;
    private readonly int _framesPerClip;
    private readonly int _batchSize;
    private readonly ImagePipeline? _augmentation;
    private readonly int _resizeW, _resizeH;

    /// <summary>Unique class names in sorted order, indexed by label value.</summary>
    public string[] ClassNames { get; }

    /// <summary>
    /// Create a video frame data loader from a DataFrame with video paths and labels.
    /// </summary>
    /// <param name="df">DataFrame containing video paths and labels.</param>
    /// <param name="videoPathColumn">Name of the column containing video file paths.</param>
    /// <param name="labelColumn">Name of the column containing class labels.</param>
    /// <param name="framesPerClip">Number of frames to extract per video clip.</param>
    /// <param name="batchSize">Number of clips per batch.</param>
    /// <param name="augmentation">Optional augmentation pipeline applied per frame.</param>
    /// <param name="resizeWidth">Target width for frames.</param>
    /// <param name="resizeHeight">Target height for frames.</param>
    public VideoFrameDataLoader(DataFrame df, string videoPathColumn, string labelColumn,
        int framesPerClip = 16, int batchSize = 8,
        ImagePipeline? augmentation = null,
        int resizeWidth = 224, int resizeHeight = 224)
    {
        var pathCol = df.GetStringColumn(videoPathColumn);
        _videoPaths = new string[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            _videoPaths[i] = pathCol[i] ?? "";

        // Encode labels
        var labelCol = df[labelColumn];
        var unique = new SortedSet<string>();
        for (int i = 0; i < df.RowCount; i++)
            unique.Add(labelCol.GetObject(i)?.ToString() ?? "");
        ClassNames = unique.ToArray();
        var labelMap = new Dictionary<string, int>();
        for (int i = 0; i < ClassNames.Length; i++) labelMap[ClassNames[i]] = i;
        _labels = new int[df.RowCount];
        for (int i = 0; i < df.RowCount; i++)
            _labels[i] = labelMap.GetValueOrDefault(labelCol.GetObject(i)?.ToString() ?? "", 0);

        _framesPerClip = framesPerClip;
        _batchSize = batchSize;
        _augmentation = augmentation;
        _resizeW = resizeWidth;
        _resizeH = resizeHeight;
    }

    /// <summary>Enumerate batches of (frames, labels).</summary>
    public IEnumerator<(ImageTensor Frames, Tensor<int> Labels)> GetEnumerator()
    {
        // Each video produces one clip of _framesPerClip frames
        // Batch _batchSize clips together
        var clipBuffer = new List<(float[] Data, int Label)>();

        for (int v = 0; v < _videoPaths.Length; v++)
        {
            var clip = TryExtractClip(v);
            if (clip is null)
                continue;

            clipBuffer.Add(clip.Value);

            if (clipBuffer.Count >= _batchSize)
            {
                yield return BuildBatch(clipBuffer);
                clipBuffer.Clear();
            }
        }

        // Yield remaining
        if (clipBuffer.Count > 0)
            yield return BuildBatch(clipBuffer);
    }

    private (float[] Data, int Label)? TryExtractClip(int videoIndex)
    {
        try
        {
            using var reader = new VideoReader(_videoPaths[videoIndex]);
            var tensor = reader.ExtractTensors(
                maxFrames: _framesPerClip,
                resizeWidth: _resizeW,
                resizeHeight: _resizeH);

            // Apply augmentation per frame if provided
            if (_augmentation is not null && tensor.IsBatch)
            {
                tensor = _augmentation.TransformBatch(tensor);
            }

            return (tensor.Span.ToArray(), _labels[videoIndex]);
        }
        catch
        {
            // Skip corrupt or unreadable videos
            return null;
        }
    }

    private (ImageTensor, Tensor<int>) BuildBatch(List<(float[] Data, int Label)> clips)
    {
        int n = clips.Count;
        int frameSize = _resizeH * _resizeW * 3;
        int clipSize = _framesPerClip * frameSize;

        var batchData = new float[n * clipSize];
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            var data = clips[i].Data;
            int copyLen = Math.Min(data.Length, clipSize);
            Array.Copy(data, 0, batchData, i * clipSize, copyLen);
            labels[i] = clips[i].Label;
        }

        var tensor = new ImageTensor(
            new Tensor<float>(batchData, n, _framesPerClip, _resizeH, _resizeW, 3));
        return (tensor, new Tensor<int>(labels, n));
    }

    /// <inheritdoc />
    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}
