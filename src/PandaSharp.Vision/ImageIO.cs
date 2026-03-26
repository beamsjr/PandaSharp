using PandaSharp.ML.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace PandaSharp.Vision;

/// <summary>
/// Static helpers for loading and saving images as ImageTensor.
/// All I/O via SixLabors.ImageSharp — no System.Drawing dependency.
/// </summary>
public static class ImageIO
{
    /// <summary>Load a single image from disk, normalized to [0,1] float.</summary>
    /// <param name="path">Path to the image file.</param>
    /// <returns>An ImageTensor with shape [H,W,3].</returns>
    public static ImageTensor Load(string path)
    {
        ArgumentNullException.ThrowIfNull(path);
        using var image = Image.Load<Rgb24>(path);
        return ImageTensor.FromImage(image);
    }

    /// <summary>Load a single image from a stream.</summary>
    /// <param name="stream">A readable stream containing image data.</param>
    /// <returns>An ImageTensor with shape [H,W,3].</returns>
    public static ImageTensor Load(Stream stream)
    {
        ArgumentNullException.ThrowIfNull(stream);
        using var image = Image.Load<Rgb24>(stream);
        return ImageTensor.FromImage(image);
    }

    /// <summary>Load multiple images as a batch tensor, resized to common dimensions.</summary>
    /// <param name="paths">Array of image file paths.</param>
    /// <param name="resizeWidth">Target width; uses first image width if null.</param>
    /// <param name="resizeHeight">Target height; uses first image height if null.</param>
    /// <returns>A batched ImageTensor with shape [N,H,W,3].</returns>
    /// <exception cref="ArgumentException">Thrown when paths array is empty.</exception>
    public static ImageTensor Load(string[] paths, int? resizeWidth = null, int? resizeHeight = null)
    {
        ArgumentNullException.ThrowIfNull(paths);
        if (paths.Length == 0) throw new ArgumentException("No image paths provided.");

        // Load first image to determine dimensions
        using var first = Image.Load<Rgb24>(paths[0]);
        int w = resizeWidth ?? first.Width;
        int h = resizeHeight ?? first.Height;

        var batchData = new float[paths.Length * h * w * 3];
        int singleLen = h * w * 3;

        Parallel.For(0, paths.Length, i =>
        {
            using var img = Image.Load<Rgb24>(paths[i]);
            if (img.Width != w || img.Height != h)
                img.Mutate(x => x.Resize(w, h));

            img.ProcessPixelRows(accessor =>
            {
                int offset = i * singleLen;
                for (int y = 0; y < h; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < w; x++)
                    {
                        int idx = offset + (y * w + x) * 3;
                        batchData[idx] = row[x].R / 255f;
                        batchData[idx + 1] = row[x].G / 255f;
                        batchData[idx + 2] = row[x].B / 255f;
                    }
                }
            });
        });

        return new ImageTensor(new Tensor<float>(batchData, paths.Length, h, w, 3));
    }

    /// <summary>Save an ImageTensor to disk as PNG or JPEG (based on extension).</summary>
    /// <param name="tensor">The image tensor to save.</param>
    /// <param name="path">Output file path. Format inferred from extension.</param>
    public static void Save(ImageTensor tensor, string path)
    {
        ArgumentNullException.ThrowIfNull(tensor);
        ArgumentNullException.ThrowIfNull(path);
        using var image = tensor.ToImage();
        image.Save(path);
    }

    /// <summary>Load images referenced by a DataFrame column.</summary>
    /// <param name="df">The DataFrame containing image paths.</param>
    /// <param name="pathColumn">Name of the column containing file paths.</param>
    /// <param name="resizeWidth">Target width; uses first image width if null.</param>
    /// <param name="resizeHeight">Target height; uses first image height if null.</param>
    /// <returns>A batched ImageTensor with shape [N,H,W,3].</returns>
    public static ImageTensor LoadFromColumn(DataFrame df, string pathColumn,
        int? resizeWidth = null, int? resizeHeight = null)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(pathColumn);
        var col = df.GetStringColumn(pathColumn);
        var paths = new List<string>();
        for (int i = 0; i < col.Length; i++)
            if (col[i] is { } p) paths.Add(p);
        return Load(paths.ToArray(), resizeWidth, resizeHeight);
    }
}
