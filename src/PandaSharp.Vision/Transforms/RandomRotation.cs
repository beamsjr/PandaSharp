using PandaSharp.ML.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Randomly rotates images by an angle sampled uniformly from [-degrees, +degrees].
/// Uses ImageSharp's affine rotation via convert-process-reconvert.
/// </summary>
public class RandomRotation : IImageTransformer
{
    private readonly float _degrees;
    private readonly Random _random;
    private readonly object _lock = new();

    /// <inheritdoc />
    public string Name => "RandomRotation";

    /// <summary>
    /// Creates a new RandomRotation transform.
    /// </summary>
    /// <param name="degrees">Maximum rotation angle in degrees. Angle is sampled from [-degrees, +degrees].</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomRotation(float degrees = 10f, int? seed = null)
    {
        if (degrees < 0) throw new ArgumentOutOfRangeException(nameof(degrees));
        _degrees = degrees;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.IsBatch)
            return TransformBatch(input);

        return RotateSingle(input);
    }

    private ImageTensor RotateSingle(ImageTensor input)
    {
        float angle;
        lock (_lock) { angle = ((float)_random.NextDouble() * 2f - 1f) * _degrees; }

        using var image = input.ToImage();
        image.Mutate(x => x.Rotate(angle));

        // ImageSharp may change image size after rotation; resize back to original dimensions
        if (image.Width != input.Width || image.Height != input.Height)
        {
            image.Mutate(x => x.Resize(new SixLabors.ImageSharp.Processing.ResizeOptions
            {
                Size = new Size(input.Width, input.Height),
                Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch
            }));
        }

        return ImageTensor.FromImage(image);
    }

    private ImageTensor TransformBatch(ImageTensor batch)
    {
        int singleLen = batch.Height * batch.Width * batch.Channels;
        var results = new float[batch.BatchSize * singleLen];

        for (int i = 0; i < batch.BatchSize; i++)
        {
            var single = batch.GetImage(i);
            var transformed = RotateSingle(single);
            transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
        }

        return new ImageTensor(
            new Tensor<float>(results, batch.BatchSize, batch.Height, batch.Width, batch.Channels),
            batch.ChannelOrder);
    }
}
