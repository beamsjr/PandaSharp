using System.Numerics;
using PandaSharp.ML.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace PandaSharp.Vision;

/// <summary>Pixel channel ordering.</summary>
public enum ChannelOrder
{
    /// <summary>Red, Green, Blue channel order.</summary>
    RGB,
    /// <summary>Blue, Green, Red channel order.</summary>
    BGR,
    /// <summary>Single-channel grayscale.</summary>
    Grayscale
}

/// <summary>How to handle corrupt or unreadable images during batch loading.</summary>
public enum CorruptImageHandling
{
    /// <summary>Skip the corrupt image (batch will have fewer images).</summary>
    Skip,
    /// <summary>Replace the corrupt image with a zero-filled tensor.</summary>
    ZeroFill,
    /// <summary>Throw an exception on corrupt images.</summary>
    Error
}

/// <summary>
/// Image-aware tensor wrapper. Pixel values normalized to [0,1] float32.
/// Shape: [Height, Width, Channels] for single image, [N, Height, Width, Channels] for batch.
/// </summary>
public class ImageTensor
{
    private readonly Tensor<float> _tensor;

    /// <summary>The underlying tensor storing normalized pixel data.</summary>
    public Tensor<float> Tensor => _tensor;

    /// <summary>Shape of the tensor: [H,W,C] or [N,H,W,C].</summary>
    public int[] Shape => _tensor.Shape;

    /// <summary>True if the tensor has a batch dimension (rank 4).</summary>
    public bool IsBatch => _tensor.Rank == 4;

    /// <summary>Number of images in the batch (1 if not batched).</summary>
    public int BatchSize => IsBatch ? _tensor.Shape[0] : 1;

    /// <summary>Image height in pixels.</summary>
    public int Height => _tensor.Rank switch
    {
        3 => _tensor.Shape[0],
        4 => _tensor.Shape[1],
        _ => throw new InvalidOperationException($"Unsupported tensor rank {_tensor.Rank}. ImageTensor supports rank 3 [H,W,C] or rank 4 [N,H,W,C].")
    };

    /// <summary>Image width in pixels.</summary>
    public int Width => _tensor.Rank switch
    {
        3 => _tensor.Shape[1],
        4 => _tensor.Shape[2],
        _ => throw new InvalidOperationException($"Unsupported tensor rank {_tensor.Rank}. ImageTensor supports rank 3 [H,W,C] or rank 4 [N,H,W,C].")
    };

    /// <summary>Number of channels (e.g. 3 for RGB, 1 for grayscale).</summary>
    public int Channels => _tensor.Rank switch
    {
        3 => _tensor.Shape[2],
        4 => _tensor.Shape[3],
        _ => throw new InvalidOperationException($"Unsupported tensor rank {_tensor.Rank}. ImageTensor supports rank 3 [H,W,C] or rank 4 [N,H,W,C].")
    };

    /// <summary>The channel ordering of the pixel data.</summary>
    public ChannelOrder ChannelOrder { get; }

    /// <summary>Flat read-only view of the normalized pixel data.</summary>
    public ReadOnlySpan<float> Span => _tensor.Span;

    /// <summary>Total number of float elements in the tensor.</summary>
    public int Length => _tensor.Length;

    /// <summary>
    /// Create an ImageTensor from an existing tensor.
    /// </summary>
    /// <param name="tensor">A rank-3 [H,W,C], rank-4 [N,H,W,C], or rank-5 [N,F,H,W,C] tensor.</param>
    /// <param name="order">The channel ordering of the pixel data.</param>
    /// <exception cref="ArgumentException">Thrown when tensor rank is not 3, 4, or 5.</exception>
    public ImageTensor(Tensor<float> tensor, ChannelOrder order = ChannelOrder.RGB)
    {
        if (tensor.Rank < 3 || tensor.Rank > 5)
            throw new ArgumentException("ImageTensor requires rank 3 [H,W,C], rank 4 [N,H,W,C], or rank 5 [N,F,H,W,C].");
        // Validate that all spatial dimensions are positive
        for (int i = 0; i < tensor.Rank; i++)
        {
            if (tensor.Shape[i] <= 0)
                throw new ArgumentException($"All tensor dimensions must be positive, but dimension {i} is {tensor.Shape[i]}.");
        }
        _tensor = tensor;
        ChannelOrder = order;
    }

    /// <summary>
    /// Create an ImageTensor from raw float data with explicit dimensions.
    /// </summary>
    /// <param name="data">Normalized pixel values in [0,1].</param>
    /// <param name="height">Image height.</param>
    /// <param name="width">Image width.</param>
    /// <param name="channels">Number of channels.</param>
    /// <param name="order">The channel ordering of the pixel data.</param>
    public ImageTensor(float[] data, int height, int width, int channels, ChannelOrder order = ChannelOrder.RGB)
        : this(new Tensor<float>(data, height, width, channels), order) { }

    /// <summary>Create from an ImageSharp Image, normalizing pixels to [0,1].</summary>
    /// <param name="image">The source image.</param>
    /// <returns>An ImageTensor with RGB channel order.</returns>
    public static ImageTensor FromImage(Image<Rgb24> image)
    {
        int h = image.Height, w = image.Width;
        var data = new float[h * w * 3];
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < h; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < w; x++)
                {
                    int idx = (y * w + x) * 3;
                    data[idx] = row[x].R / 255f;
                    data[idx + 1] = row[x].G / 255f;
                    data[idx + 2] = row[x].B / 255f;
                }
            }
        });
        return new ImageTensor(data, h, w, 3);
    }

    /// <summary>Add batch dimension N=1 if not already batched.</summary>
    /// <returns>A batched ImageTensor with rank 4.</returns>
    public ImageTensor ToBatch()
    {
        if (IsBatch) return this;
        return new ImageTensor(new Tensor<float>(_tensor.Span.ToArray(), 1, Height, Width, Channels), ChannelOrder);
    }

    /// <summary>Remove batch dimension (takes first image if batched).</summary>
    /// <returns>An unbatched ImageTensor with rank 3.</returns>
    public ImageTensor Unbatch()
    {
        if (!IsBatch) return this;
        int singleLen = Height * Width * Channels;
        var data = _tensor.Span.Slice(0, singleLen).ToArray();
        return new ImageTensor(data, Height, Width, Channels, ChannelOrder);
    }

    /// <summary>Get the underlying Tensor.</summary>
    /// <returns>The raw <see cref="Tensor{T}"/> backing this image.</returns>
    public Tensor<float> ToTensor() => _tensor;

    /// <summary>Get a single image from a batch by index.</summary>
    /// <param name="index">Zero-based index into the batch.</param>
    /// <returns>An unbatched ImageTensor for the requested image.</returns>
    /// <exception cref="IndexOutOfRangeException">Thrown when index is out of range.</exception>
    public ImageTensor GetImage(int index)
    {
        if (!IsBatch) return index == 0 ? this : throw new IndexOutOfRangeException();
        int singleLen = Height * Width * Channels;
        var data = _tensor.Span.Slice(index * singleLen, singleLen).ToArray();
        return new ImageTensor(data, Height, Width, Channels, ChannelOrder);
    }

    /// <summary>Convert back to ImageSharp Image (denormalize from [0,1]).</summary>
    /// <returns>An <see cref="Image{Rgb24}"/> with pixel values scaled to [0,255].</returns>
    public Image<Rgb24> ToImage()
    {
        var img = IsBatch ? Unbatch() : this;
        var image = new Image<Rgb24>(img.Width, img.Height);
        var pixelData = img.Span.ToArray();
        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < img.Height; y++)
            {
                var row = accessor.GetRowSpan(y);
                for (int x = 0; x < img.Width; x++)
                {
                    int idx = (y * img.Width + x) * img.Channels;
                    byte r = (byte)Math.Clamp(pixelData[idx] * 255f, 0, 255);
                    byte g = img.Channels > 1 ? (byte)Math.Clamp(pixelData[idx + 1] * 255f, 0, 255) : r;
                    byte b = img.Channels > 2 ? (byte)Math.Clamp(pixelData[idx + 2] * 255f, 0, 255) : r;
                    row[x] = new Rgb24(r, g, b);
                }
            }
        });
        return image;
    }

    /// <summary>Get the raw float data as a mutable array (for transforms that modify in-place).</summary>
    /// <returns>A copy of the underlying data as a mutable array.</returns>
    internal float[] GetMutableData() => _tensor.Span.ToArray();
}
