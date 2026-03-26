namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Per-channel normalization: (pixel - mean) / std.
/// Provides static presets for common datasets such as ImageNet and CIFAR-10.
/// </summary>
public class Normalize : IImageTransformer
{
    private readonly float[] _mean;
    private readonly float[] _std;

    /// <inheritdoc />
    public string Name => "Normalize";

    /// <summary>
    /// Creates a new Normalize transform with per-channel mean and standard deviation.
    /// </summary>
    /// <param name="mean">Per-channel mean values to subtract.</param>
    /// <param name="std">Per-channel standard deviation values to divide by.</param>
    public Normalize(float[] mean, float[] std)
    {
        ArgumentNullException.ThrowIfNull(mean);
        ArgumentNullException.ThrowIfNull(std);
        if (mean.Length != std.Length)
            throw new ArgumentException("Mean and std arrays must have the same length.");
        if (mean.Length == 0)
            throw new ArgumentException("Mean and std arrays must not be empty.");
        _mean = (float[])mean.Clone();
        _std = (float[])std.Clone();
    }

    /// <summary>
    /// Returns a Normalize preset for ImageNet (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
    /// </summary>
    public static Normalize ImageNet() =>
        new([0.485f, 0.456f, 0.406f], [0.229f, 0.224f, 0.225f]);

    /// <summary>
    /// Returns a Normalize preset for CIFAR-10 (mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]).
    /// </summary>
    public static Normalize CIFAR10() =>
        new([0.4914f, 0.4822f, 0.4465f], [0.2470f, 0.2435f, 0.2616f]);

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (_mean.Length != input.Channels)
            throw new ArgumentException(
                $"Normalize has {_mean.Length} channels but input has {input.Channels} channels.");

        var src = input.Span;
        var result = new float[src.Length];
        int channels = input.Channels;

        // Pre-compute inverse std for multiplication instead of division
        Span<float> invStd = stackalloc float[channels];
        for (int c = 0; c < channels; c++)
            invStd[c] = 1f / Math.Max(_std[c], 1e-7f);

        // Iterate over all pixels, applying per-channel normalization
        int totalPixels = src.Length / channels;
        for (int p = 0; p < totalPixels; p++)
        {
            int offset = p * channels;
            for (int c = 0; c < channels; c++)
                result[offset + c] = (src[offset + c] - _mean[c]) * invStd[c];
        }

        if (input.IsBatch)
        {
            return new ImageTensor(
                new PandaSharp.ML.Tensors.Tensor<float>(result,
                    input.BatchSize, input.Height, input.Width, channels),
                input.ChannelOrder);
        }

        return new ImageTensor(result, input.Height, input.Width, channels, input.ChannelOrder);
    }
}
