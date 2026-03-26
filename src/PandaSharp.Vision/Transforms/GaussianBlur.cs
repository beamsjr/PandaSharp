using PandaSharp.ML.Tensors;

namespace PandaSharp.Vision.Transforms;

/// <summary>
/// Applies Gaussian blur using a separable kernel directly on the float tensor.
/// No ImageSharp round-trip — operates on raw pixel data for maximum performance.
/// </summary>
public class GaussianBlur : IImageTransformer
{
    private readonly float _sigma;
    private readonly float[] _kernel;
    private readonly int _radius;

    /// <inheritdoc />
    public string Name => "GaussianBlur";

    /// <summary>Creates a new GaussianBlur transform.</summary>
    /// <param name="sigma">Standard deviation of the Gaussian kernel.</param>
    public GaussianBlur(float sigma = 1.0f)
    {
        if (sigma <= 0) throw new ArgumentOutOfRangeException(nameof(sigma));
        _sigma = sigma;
        _radius = Math.Max(1, (int)Math.Ceiling(sigma * 3));
        _kernel = BuildKernel(_radius, sigma);
    }

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.IsBatch)
        {
            int singleLen = input.Height * input.Width * input.Channels;
            var results = new float[input.Length];
            Parallel.For(0, input.BatchSize, i =>
            {
                var single = input.GetImage(i);
                var blurred = BlurSingle(single);
                blurred.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
            });
            return new ImageTensor(new Tensor<float>(results, input.BatchSize, input.Height, input.Width, input.Channels), input.ChannelOrder);
        }
        return BlurSingle(input);
    }

    private ImageTensor BlurSingle(ImageTensor input)
    {
        int h = input.Height, w = input.Width, c = input.Channels;
        var src = input.Span.ToArray();
        var temp = new float[h * w * c];
        var dst = new float[h * w * c];
        int kSize = _kernel.Length;

        // Horizontal pass — process all channels per pixel together for cache locality
        for (int y = 0; y < h; y++)
        {
            int rowOff = y * w * c;
            for (int x = 0; x < w; x++)
            {
                int dstOff = rowOff + x * c;
                // Accumulate across kernel
                if (c == 3) // RGB fast path
                {
                    float s0 = 0, s1 = 0, s2 = 0;
                    for (int k = 0; k < kSize; k++)
                    {
                        int sx = Math.Clamp(x + k - _radius, 0, w - 1);
                        int srcOff = rowOff + sx * 3;
                        float w_ = _kernel[k];
                        s0 += src[srcOff] * w_;
                        s1 += src[srcOff + 1] * w_;
                        s2 += src[srcOff + 2] * w_;
                    }
                    temp[dstOff] = s0; temp[dstOff + 1] = s1; temp[dstOff + 2] = s2;
                }
                else
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        float sum = 0;
                        for (int k = 0; k < kSize; k++)
                        {
                            int sx = Math.Clamp(x + k - _radius, 0, w - 1);
                            sum += src[rowOff + sx * c + ch] * _kernel[k];
                        }
                        temp[dstOff + ch] = sum;
                    }
                }
            }
        }

        // Vertical pass
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                int dstOff = (y * w + x) * c;
                if (c == 3)
                {
                    float s0 = 0, s1 = 0, s2 = 0;
                    for (int k = 0; k < kSize; k++)
                    {
                        int sy = Math.Clamp(y + k - _radius, 0, h - 1);
                        int srcOff = (sy * w + x) * 3;
                        float w_ = _kernel[k];
                        s0 += temp[srcOff] * w_;
                        s1 += temp[srcOff + 1] * w_;
                        s2 += temp[srcOff + 2] * w_;
                    }
                    dst[dstOff] = s0; dst[dstOff + 1] = s1; dst[dstOff + 2] = s2;
                }
                else
                {
                    for (int ch = 0; ch < c; ch++)
                    {
                        float sum = 0;
                        for (int k = 0; k < kSize; k++)
                        {
                            int sy = Math.Clamp(y + k - _radius, 0, h - 1);
                            sum += temp[(sy * w + x) * c + ch] * _kernel[k];
                        }
                        dst[dstOff + ch] = sum;
                    }
                }
            }
        }

        return new ImageTensor(dst, h, w, c, input.ChannelOrder);
    }

    private static float[] BuildKernel(int radius, float sigma)
    {
        int size = 2 * radius + 1;
        var kernel = new float[size];
        float sum = 0;
        float s2 = 2 * sigma * sigma;
        for (int i = 0; i < size; i++)
        {
            float x = i - radius;
            kernel[i] = MathF.Exp(-x * x / s2);
            sum += kernel[i];
        }
        for (int i = 0; i < size; i++) kernel[i] /= sum;
        return kernel;
    }
}
