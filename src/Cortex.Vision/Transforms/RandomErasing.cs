namespace Cortex.Vision.Transforms;

/// <summary>
/// Randomly erases a rectangular region of the image, filling it with random values.
/// This is a regularization technique also known as "cutout".
/// Operates directly on the float array.
/// </summary>
public class RandomErasing : IImageTransformer
{
    private readonly double _probability;
    private readonly float _scaleMin;
    private readonly float _scaleMax;
    private readonly float _ratioMin;
    private readonly float _ratioMax;
    private readonly Random _random;
    private readonly object _lock = new();

    /// <inheritdoc />
    public string Name => "RandomErasing";

    /// <summary>
    /// Creates a new RandomErasing transform.
    /// </summary>
    /// <param name="probability">Probability of applying the erasing (0.0 to 1.0).</param>
    /// <param name="scaleMin">Minimum fraction of image area to erase.</param>
    /// <param name="scaleMax">Maximum fraction of image area to erase.</param>
    /// <param name="ratioMin">Minimum aspect ratio of the erased region.</param>
    /// <param name="ratioMax">Maximum aspect ratio of the erased region.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public RandomErasing(double probability = 0.5, float scaleMin = 0.02f, float scaleMax = 0.33f,
        float ratioMin = 0.3f, float ratioMax = 3.3f, int? seed = null)
    {
        if (probability < 0 || probability > 1)
            throw new ArgumentOutOfRangeException(nameof(probability));
        _probability = probability;
        _scaleMin = scaleMin;
        _scaleMax = scaleMax;
        _ratioMin = ratioMin;
        _ratioMax = ratioMax;
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
                var transformed = EraseSingle(single);
                transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
            }

            return new ImageTensor(
                new Cortex.ML.Tensors.Tensor<float>(results,
                    input.BatchSize, input.Height, input.Width, input.Channels),
                input.ChannelOrder);
        }

        return EraseSingle(input);
    }

    private ImageTensor EraseSingle(ImageTensor input)
    {
        bool shouldErase;
        lock (_lock) { shouldErase = _random.NextDouble() < _probability; }
        if (!shouldErase)
            return input;

        int h = input.Height, w = input.Width, c = input.Channels;
        float imageArea = h * w;

        var result = input.GetMutableData();

        // Try up to 10 times to find a valid region
        for (int attempt = 0; attempt < 10; attempt++)
        {
            float scale, ratio;
            int top, left;
            lock (_lock)
            {
                scale = _scaleMin + (float)_random.NextDouble() * (_scaleMax - _scaleMin);
                float logRatioMin = MathF.Log(_ratioMin);
                float logRatioMax = MathF.Log(_ratioMax);
                ratio = MathF.Exp(logRatioMin + (float)_random.NextDouble() * (logRatioMax - logRatioMin));
            }
            float area = imageArea * scale;

            int eraseH = (int)MathF.Round(MathF.Sqrt(area / ratio));
            int eraseW = (int)MathF.Round(MathF.Sqrt(area * ratio));

            if (eraseH >= h || eraseW >= w || eraseH <= 0 || eraseW <= 0)
                continue;

            lock (_lock)
            {
                top = _random.Next(h - eraseH);
                left = _random.Next(w - eraseW);
            }

            // Fill with random values
            for (int y = top; y < top + eraseH; y++)
            {
                for (int x = left; x < left + eraseW; x++)
                {
                    int offset = (y * w + x) * c;
                    for (int ch = 0; ch < c; ch++)
                    {
                        float val;
                        lock (_lock) { val = (float)_random.NextDouble(); }
                        result[offset + ch] = val;
                    }
                }
            }

            break;
        }

        return new ImageTensor(result, h, w, c, input.ChannelOrder);
    }
}
