using PandaSharp.ML.Tensors;

namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Crops the center region of an image to the specified dimensions.
/// Operates directly on the float array for performance.
/// </summary>
public class CenterCrop : IImageTransformer
{
    private readonly int _width;
    private readonly int _height;

    /// <inheritdoc />
    public string Name => "CenterCrop";

    /// <summary>
    /// Creates a new CenterCrop transform.
    /// </summary>
    /// <param name="width">Target crop width in pixels.</param>
    /// <param name="height">Target crop height in pixels.</param>
    public CenterCrop(int width, int height)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        _width = width;
        _height = height;
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
        if (_width > input.Width || _height > input.Height)
            throw new ArgumentException(
                $"Crop size ({_width}x{_height}) exceeds image size ({input.Width}x{input.Height}).");

        int offsetX = (input.Width - _width) / 2;
        int offsetY = (input.Height - _height) / 2;
        int channels = input.Channels;

        var src = input.Span;
        var result = new float[_height * _width * channels];
        int srcRowStride = input.Width * channels;
        int dstRowStride = _width * channels;

        for (int y = 0; y < _height; y++)
        {
            int srcOffset = (offsetY + y) * srcRowStride + offsetX * channels;
            int dstOffset = y * dstRowStride;
            src.Slice(srcOffset, dstRowStride).CopyTo(result.AsSpan(dstOffset, dstRowStride));
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
