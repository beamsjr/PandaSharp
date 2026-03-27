namespace Cortex.Vision.Transforms;

/// <summary>
/// Randomly flips an image horizontally with a given probability.
/// Operates directly on the float array by reversing pixels in each row.
/// </summary>
public class RandomHorizontalFlip : IImageTransformer
{
    private readonly double _probability;
    private readonly Random _random;
    private readonly object _lock = new();

    /// <inheritdoc />
    public string Name => "RandomHorizontalFlip";

    /// <summary>
    /// Creates a new RandomHorizontalFlip transform.
    /// </summary>
    /// <param name="probability">Probability of flipping (0.0 to 1.0).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomHorizontalFlip(double probability = 0.5, int? seed = null)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability), "Probability must be in [0, 1].");
        _probability = probability;
        _random = seed.HasValue ? new Random(seed.Value) : new Random();
    }

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.IsBatch)
        {
            int singleLen = input.Height * input.Width * input.Channels;
            var results = new float[input.BatchSize * singleLen];

            for (int i = 0; i < input.BatchSize; i++)
            {
                var single = input.GetImage(i);
                var transformed = FlipSingle(single);
                transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
            }

            return new ImageTensor(
                new Cortex.ML.Tensors.Tensor<float>(results,
                    input.BatchSize, input.Height, input.Width, input.Channels),
                input.ChannelOrder);
        }

        return FlipSingle(input);
    }

    private ImageTensor FlipSingle(ImageTensor input)
    {
        bool shouldFlip;
        lock (_lock)
        {
            shouldFlip = _random.NextDouble() < _probability;
        }
        if (!shouldFlip)
            return input;

        int h = input.Height, w = input.Width, c = input.Channels;
        var src = input.Span;
        var result = new float[h * w * c];
        int rowStride = w * c;

        for (int y = 0; y < h; y++)
        {
            int rowOffset = y * rowStride;
            for (int x = 0; x < w; x++)
            {
                int srcIdx = rowOffset + x * c;
                int dstIdx = rowOffset + (w - 1 - x) * c;
                for (int ch = 0; ch < c; ch++)
                    result[dstIdx + ch] = src[srcIdx + ch];
            }
        }

        return new ImageTensor(result, h, w, c, input.ChannelOrder);
    }
}
