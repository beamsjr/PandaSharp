using PandaSharp.ML.Tensors;

namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Converts an RGB image to single-channel grayscale using ITU-R BT.601 weights:
/// Gray = 0.2989 * R + 0.5870 * G + 0.1140 * B.
/// Output has Channels=1 and ChannelOrder.Grayscale.
/// </summary>
public class Grayscale : IImageTransformer
{
    /// <inheritdoc />
    public string Name => "Grayscale";

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.Channels == 1)
            return input;

        if (input.Channels < 3)
            throw new ArgumentException("Grayscale transform requires at least 3 channels (RGB).");

        if (input.IsBatch)
            return TransformBatch(input);

        return GrayscaleSingle(input);
    }

    private static ImageTensor GrayscaleSingle(ImageTensor input)
    {
        int h = input.Height, w = input.Width;
        int srcChannels = input.Channels;
        var src = input.Span;
        var result = new float[h * w];

        int totalPixels = h * w;
        for (int p = 0; p < totalPixels; p++)
        {
            int srcIdx = p * srcChannels;
            result[p] = 0.2989f * src[srcIdx] + 0.5870f * src[srcIdx + 1] + 0.1140f * src[srcIdx + 2];
        }

        return new ImageTensor(result, h, w, 1, ChannelOrder.Grayscale);
    }

    private static ImageTensor TransformBatch(ImageTensor batch)
    {
        int h = batch.Height, w = batch.Width;
        int singleLen = h * w;
        var results = new float[batch.BatchSize * singleLen];

        for (int i = 0; i < batch.BatchSize; i++)
        {
            var single = batch.GetImage(i);
            var transformed = GrayscaleSingle(single);
            transformed.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
        }

        return new ImageTensor(
            new Tensor<float>(results, batch.BatchSize, h, w, 1),
            ChannelOrder.Grayscale);
    }
}
