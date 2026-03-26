using PandaSharp.ML.Tensors;

namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Randomly crops a region from the image with optional zero-padding.
/// Uses a <see cref="Random"/> instance for reproducible augmentation.
/// </summary>
public class RandomCrop : IImageTransformer
{
    private readonly int _width;
    private readonly int _height;
    private readonly int _padding;
    private readonly Random _random;
    private readonly object _lock = new();

    /// <inheritdoc />
    public string Name => "RandomCrop";

    /// <summary>
    /// Creates a new RandomCrop transform.
    /// </summary>
    /// <param name="width">Crop width in pixels.</param>
    /// <param name="height">Crop height in pixels.</param>
    /// <param name="padding">Pixels of zero-padding added to each side before cropping.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomCrop(int width, int height, int padding = 0, int? seed = null)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        if (padding < 0) throw new ArgumentOutOfRangeException(nameof(padding));
        _width = width;
        _height = height;
        _padding = padding;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.IsBatch)
            return TransformBatch(input);

        return CropSingle(input);
    }

    private ImageTensor CropSingle(ImageTensor input)
    {
        int channels = input.Channels;
        int paddedH = input.Height + 2 * _padding;
        int paddedW = input.Width + 2 * _padding;

        if (_width > paddedW || _height > paddedH)
            throw new ArgumentException(
                $"Crop size ({_width}x{_height}) exceeds padded image size ({paddedW}x{paddedH}).");

        // Build padded image if needed
        ReadOnlySpan<float> src;
        float[]? paddedData = null;
        int srcW, srcH;

        if (_padding > 0)
        {
            paddedData = new float[paddedH * paddedW * channels];
            var origSpan = input.Span;
            int origRowStride = input.Width * channels;
            int paddedRowStride = paddedW * channels;

            for (int y = 0; y < input.Height; y++)
            {
                int srcOffset = y * origRowStride;
                int dstOffset = (y + _padding) * paddedRowStride + _padding * channels;
                origSpan.Slice(srcOffset, origRowStride).CopyTo(paddedData.AsSpan(dstOffset, origRowStride));
            }

            src = paddedData;
            srcW = paddedW;
            srcH = paddedH;
        }
        else
        {
            src = input.Span;
            srcW = input.Width;
            srcH = input.Height;
        }

        int maxX = srcW - _width;
        int maxY = srcH - _height;
        int offsetX, offsetY;
        lock (_lock)
        {
            offsetX = maxX > 0 ? _random.Next(maxX + 1) : 0;
            offsetY = maxY > 0 ? _random.Next(maxY + 1) : 0;
        }

        var result = new float[_height * _width * channels];
        int srcRowStride2 = srcW * channels;
        int dstRowStride = _width * channels;

        for (int y = 0; y < _height; y++)
        {
            int srcOff = (offsetY + y) * srcRowStride2 + offsetX * channels;
            int dstOff = y * dstRowStride;
            src.Slice(srcOff, dstRowStride).CopyTo(result.AsSpan(dstOff, dstRowStride));
        }

        return new ImageTensor(result, _height, _width, channels, input.ChannelOrder);
    }

    private ImageTensor TransformBatch(ImageTensor batch)
    {
        int channels = batch.Channels;
        int singleLen = _height * _width * channels;
        var results = new float[batch.BatchSize * singleLen];

        for (int i = 0; i < batch.BatchSize; i++)
        {
            var single = batch.GetImage(i);
            var transformed = CropSingle(single);
            transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
        }

        return new ImageTensor(
            new Tensor<float>(results, batch.BatchSize, _height, _width, channels),
            batch.ChannelOrder);
    }
}
