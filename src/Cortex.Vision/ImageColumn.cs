using Cortex.Column;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Cortex.Vision;

/// <summary>
/// Lightweight column type that stores image paths with helper methods for batch loading.
/// Wraps a <see cref="StringColumn"/> and provides image-aware operations.
/// </summary>
public class ImageColumn
{
    private readonly StringColumn _pathColumn;

    /// <summary>Name of the underlying path column.</summary>
    public string Name => _pathColumn.Name;

    /// <summary>Number of entries in the column.</summary>
    public int Length => _pathColumn.Length;

    /// <summary>
    /// Create an ImageColumn from an existing StringColumn of file paths.
    /// </summary>
    /// <param name="pathColumn">The string column containing image file paths.</param>
    public ImageColumn(StringColumn pathColumn)
    {
        ArgumentNullException.ThrowIfNull(pathColumn);
        _pathColumn = pathColumn;
    }

    /// <summary>Get the image path at the given index.</summary>
    /// <param name="index">Zero-based row index.</param>
    public string? this[int index] => _pathColumn[index];

    /// <summary>Load all images as a batch tensor, optionally resizing to common dimensions.</summary>
    /// <param name="resizeWidth">Target width, or null to use the first image's width.</param>
    /// <param name="resizeHeight">Target height, or null to use the first image's height.</param>
    /// <returns>A batched ImageTensor with shape [N,H,W,3].</returns>
    public ImageTensor LoadAll(int? resizeWidth = null, int? resizeHeight = null)
    {
        var paths = new List<string>();
        for (int i = 0; i < Length; i++)
            if (_pathColumn[i] is { } p) paths.Add(p);
        return ImageIO.Load(paths.ToArray(), resizeWidth, resizeHeight);
    }

    /// <summary>Load a single image by row index.</summary>
    /// <param name="index">Zero-based row index.</param>
    /// <returns>An ImageTensor with shape [H,W,3].</returns>
    /// <exception cref="ArgumentException">Thrown when the path at the given index is null.</exception>
    public ImageTensor LoadAt(int index)
    {
        var path = _pathColumn[index] ?? throw new ArgumentException($"Null image path at index {index}");
        return ImageIO.Load(path);
    }

    /// <summary>
    /// Add a base64-encoded PNG thumbnail column to a DataFrame for display.
    /// </summary>
    /// <param name="df">The DataFrame to extend.</param>
    /// <param name="width">Thumbnail width in pixels.</param>
    /// <param name="height">Thumbnail height in pixels.</param>
    /// <returns>A new DataFrame with the thumbnail column appended.</returns>
    public DataFrame WithThumbnails(DataFrame df, int width = 64, int height = 64)
    {
        var thumbs = new string?[Length];
        Parallel.For(0, Length, i =>
        {
            try
            {
                using var img = Image.Load<Rgb24>(_pathColumn[i]!);
                img.Mutate(x => x.Resize(width, height));
                using var ms = new MemoryStream();
                img.SaveAsPng(ms);
                thumbs[i] = $"data:image/png;base64,{Convert.ToBase64String(ms.ToArray())}";
            }
            catch (Exception)
            {
                thumbs[i] = null;
            }
        });
        return df.AddColumn(new StringColumn($"{Name}_thumb", thumbs));
    }
}
