using Cortex.Column;
using Cortex.Vision.Transforms;

namespace Cortex.Vision;

/// <summary>
/// Extension methods for DataFrame image operations.
/// </summary>
public static class VisionExtensions
{
    /// <summary>
    /// Wrap a string column as an <see cref="ImageColumn"/> for image loading helpers.
    /// </summary>
    /// <param name="df">The DataFrame containing the path column.</param>
    /// <param name="pathColumn">Name of the string column holding image file paths.</param>
    /// <returns>An <see cref="ImageColumn"/> wrapping the specified column.</returns>
    public static ImageColumn AsImageColumn(this DataFrame df, string pathColumn)
        => new(df.GetStringColumn(pathColumn));

    /// <summary>
    /// Create an <see cref="ImageDataLoader"/> from a DataFrame with image paths and labels.
    /// </summary>
    /// <param name="df">The DataFrame containing the dataset.</param>
    /// <param name="pathColumn">Name of the column containing image file paths.</param>
    /// <param name="labelColumn">Name of the column containing class labels.</param>
    /// <param name="batchSize">Number of images per batch.</param>
    /// <param name="shuffle">Whether to shuffle indices each epoch.</param>
    /// <param name="seed">Optional random seed for reproducibility.</param>
    /// <param name="augmentation">Optional augmentation pipeline applied per image.</param>
    /// <param name="resizeWidth">Target width for loaded images.</param>
    /// <param name="resizeHeight">Target height for loaded images.</param>
    /// <returns>An <see cref="ImageDataLoader"/> ready for iteration.</returns>
    public static ImageDataLoader ToImageDataLoader(this DataFrame df,
        string pathColumn, string labelColumn,
        int batchSize = 32, bool shuffle = true, int? seed = null,
        ImagePipeline? augmentation = null,
        int resizeWidth = 224, int resizeHeight = 224)
        => new(df, pathColumn, labelColumn, batchSize, shuffle, seed,
            augmentation, resizeWidth, resizeHeight);
}
