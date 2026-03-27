using Cortex.ML.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Cortex.Vision.Transforms;

/// <summary>Interpolation mode used when resizing images.</summary>
public enum ResizeMode { NearestNeighbor, Bilinear, Bicubic }

/// <summary>
/// Resizes images to a target width and height.
/// Uses direct tensor bilinear interpolation for performance (no ImageSharp round-trip).
/// </summary>
public class Resize : IImageTransformer
{
    private readonly int _width;
    private readonly int _height;
    private readonly ResizeMode _mode;

    /// <inheritdoc />
    public string Name => "Resize";

    public Resize(int width, int height, ResizeMode mode = ResizeMode.Bilinear)
    {
        if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
        if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
        _width = width;
        _height = height;
        _mode = mode;
    }

    /// <inheritdoc />
    public ImageTensor Transform(ImageTensor input)
    {
        if (input.Height == _height && input.Width == _width) return input;

        if (input.IsBatch)
        {
            int singleLen = _height * _width * input.Channels;
            var results = new float[input.BatchSize * singleLen];
            Parallel.For(0, input.BatchSize, i =>
            {
                var single = input.GetImage(i);
                var resized = ResizeDirect(single);
                resized.Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
            });
            return new ImageTensor(new Tensor<float>(results, input.BatchSize, _height, _width, input.Channels), input.ChannelOrder);
        }

        return ResizeDirect(input);
    }

    private ImageTensor ResizeDirect(ImageTensor input)
    {
        int srcH = input.Height, srcW = input.Width, c = input.Channels;
        var src = input.Span;
        var dst = new float[_height * _width * c];

        float scaleY = (float)srcH / _height;
        float scaleX = (float)srcW / _width;

        if (_mode == ResizeMode.NearestNeighbor)
        {
            for (int y = 0; y < _height; y++)
            {
                int srcY = Math.Min((int)(y * scaleY), srcH - 1);
                for (int x = 0; x < _width; x++)
                {
                    int srcX = Math.Min((int)(x * scaleX), srcW - 1);
                    int srcIdx = (srcY * srcW + srcX) * c;
                    int dstIdx = (y * _width + x) * c;
                    for (int ch = 0; ch < c; ch++)
                        dst[dstIdx + ch] = src[srcIdx + ch];
                }
            }
        }
        else // Bilinear (default) or Bicubic (use bilinear for speed)
        {
            for (int y = 0; y < _height; y++)
            {
                float srcYf = y * scaleY - 0.5f;
                // Compute interpolation weights before clamping source coordinates
                float fy = srcYf - MathF.Floor(srcYf);
                int y0 = Math.Clamp((int)MathF.Floor(srcYf), 0, srcH - 1);
                int y1 = Math.Min(y0 + 1, srcH - 1);

                for (int x = 0; x < _width; x++)
                {
                    float srcXf = x * scaleX - 0.5f;
                    // Compute interpolation weights before clamping source coordinates
                    float fx = srcXf - MathF.Floor(srcXf);
                    int x0 = Math.Clamp((int)MathF.Floor(srcXf), 0, srcW - 1);
                    int x1 = Math.Min(x0 + 1, srcW - 1);

                    int i00 = (y0 * srcW + x0) * c;
                    int i01 = (y0 * srcW + x1) * c;
                    int i10 = (y1 * srcW + x0) * c;
                    int i11 = (y1 * srcW + x1) * c;
                    int dstIdx = (y * _width + x) * c;

                    float w00 = (1 - fx) * (1 - fy);
                    float w01 = fx * (1 - fy);
                    float w10 = (1 - fx) * fy;
                    float w11 = fx * fy;

                    for (int ch = 0; ch < c; ch++)
                        dst[dstIdx + ch] = src[i00 + ch] * w00 + src[i01 + ch] * w01 +
                                           src[i10 + ch] * w10 + src[i11 + ch] * w11;
                }
            }
        }

        return new ImageTensor(dst, _height, _width, c, input.ChannelOrder);
    }
}
