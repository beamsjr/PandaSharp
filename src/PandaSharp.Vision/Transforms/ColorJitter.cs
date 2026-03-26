namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Applies random brightness and contrast adjustments to images.
/// Each parameter specifies the maximum deviation from the identity transform.
/// Operates directly on the float array for performance.
/// </summary>
public class ColorJitter : IImageTransformer
{
    private readonly float _brightness;
    private readonly float _contrast;
    private readonly float _saturation;
    private readonly float _hue;
    private readonly Random _random;
    private readonly object _lock = new();

    /// <inheritdoc />
    public string Name => "ColorJitter";

    /// <summary>
    /// Creates a new ColorJitter transform.
    /// </summary>
    /// <param name="brightness">Maximum brightness deviation. Factor is sampled uniformly from [1-b, 1+b].</param>
    /// <param name="contrast">Maximum contrast deviation. Pixel values are linearly interpolated toward the mean gray value.</param>
    /// <param name="saturation">Maximum saturation deviation (reserved for future use).</param>
    /// <param name="hue">Maximum hue deviation (reserved for future use).</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    public ColorJitter(float brightness = 0, float contrast = 0, float saturation = 0, float hue = 0, int? seed = null)
    {
        if (brightness < 0) throw new ArgumentOutOfRangeException(nameof(brightness));
        if (contrast < 0) throw new ArgumentOutOfRangeException(nameof(contrast));
        if (saturation < 0) throw new ArgumentOutOfRangeException(nameof(saturation));
        if (hue < 0) throw new ArgumentOutOfRangeException(nameof(hue));
        _brightness = brightness;
        _contrast = contrast;
        _saturation = saturation;
        _hue = hue;
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
                var transformed = JitterSingle(single);
                transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
            }

            return new ImageTensor(
                new PandaSharp.ML.Tensors.Tensor<float>(results,
                    input.BatchSize, input.Height, input.Width, input.Channels),
                input.ChannelOrder);
        }

        return JitterSingle(input);
    }

    private ImageTensor JitterSingle(ImageTensor input)
    {
        var src = input.Span;
        var result = new float[src.Length];
        src.CopyTo(result);

        int channels = input.Channels;
        int totalPixels = src.Length / channels;

        // Brightness: multiply all channels by a random factor in [1-b, 1+b]
        if (_brightness > 0)
        {
            float factor;
            lock (_lock) { factor = 1f + ((float)_random.NextDouble() * 2f - 1f) * _brightness; }
            for (int i = 0; i < result.Length; i++)
                result[i] *= factor;
        }

        // Contrast: lerp toward mean gray
        if (_contrast > 0)
        {
            // Compute mean gray value
            float sum = 0;
            for (int i = 0; i < result.Length; i++)
                sum += result[i];
            float meanGray = sum / result.Length;

            float factor;
            lock (_lock) { factor = 1f + ((float)_random.NextDouble() * 2f - 1f) * _contrast; }
            for (int i = 0; i < result.Length; i++)
                result[i] = meanGray + factor * (result[i] - meanGray);
        }

        // Saturation: lerp toward per-pixel luminance
        if (_saturation > 0 && channels >= 3)
        {
            float factor;
            lock (_lock) { factor = 1f + ((float)_random.NextDouble() * 2f - 1f) * _saturation; }
            for (int p = 0; p < totalPixels; p++)
            {
                int offset = p * channels;
                float luma = 0.2989f * result[offset] + 0.5870f * result[offset + 1] + 0.1140f * result[offset + 2];
                for (int c = 0; c < 3; c++)
                    result[offset + c] = luma + factor * (result[offset + c] - luma);
            }
        }

        return new ImageTensor(result, input.Height, input.Width, channels, input.ChannelOrder);
    }
}
