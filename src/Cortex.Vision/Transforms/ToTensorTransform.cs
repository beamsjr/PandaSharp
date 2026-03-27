using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace Cortex.Vision.Transforms;

/// <summary>
/// Converts an Image&lt;Rgb24&gt; to an ImageTensor by wrapping <see cref="ImageTensor.FromImage"/>.
/// Used as the first step in an augmentation pipeline when starting from raw images.
/// Optionally resizes the image before conversion.
/// </summary>
public class ToTensorTransform : IImageTransformer
{
    private readonly int? _resizeWidth;
    private readonly int? _resizeHeight;

    /// <inheritdoc />
    public string Name => "ToTensor";

    /// <summary>
    /// Creates a new ToTensorTransform.
    /// </summary>
    /// <param name="resizeWidth">Optional width to resize to before converting.</param>
    /// <param name="resizeHeight">Optional height to resize to before converting.</param>
    public ToTensorTransform(int? resizeWidth = null, int? resizeHeight = null)
    {
        _resizeWidth = resizeWidth;
        _resizeHeight = resizeHeight;
    }

    /// <summary>
    /// Passes through an existing ImageTensor, optionally applying resize.
    /// If resize dimensions are specified and differ from the input, a Resize transform is applied.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <returns>The image tensor, optionally resized.</returns>
    public ImageTensor Transform(ImageTensor input)
    {
        if (_resizeWidth.HasValue && _resizeHeight.HasValue &&
            (input.Width != _resizeWidth.Value || input.Height != _resizeHeight.Value))
        {
            var resize = new Resize(_resizeWidth.Value, _resizeHeight.Value);
            return resize.Transform(input);
        }

        return input;
    }

    /// <summary>
    /// Creates an ImageTensor from an ImageSharp image, optionally resizing first.
    /// </summary>
    /// <param name="image">The source image to convert.</param>
    /// <returns>A normalized ImageTensor with pixel values in [0, 1].</returns>
    public ImageTensor TransformImage(Image<Rgb24> image)
    {
        if (_resizeWidth.HasValue && _resizeHeight.HasValue)
        {
            image.Mutate(x => x.Resize(new ResizeOptions
            {
                Size = new Size(_resizeWidth.Value, _resizeHeight.Value),
                Mode = SixLabors.ImageSharp.Processing.ResizeMode.Stretch
            }));
        }

        return ImageTensor.FromImage(image);
    }
}
