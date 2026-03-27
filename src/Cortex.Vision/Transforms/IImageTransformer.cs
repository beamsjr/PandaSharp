namespace Cortex.Vision.Transforms;

/// <summary>
/// Stateless image transformation. Unlike ML transformers, image augmentations
/// don't learn parameters (no Fit step).
/// </summary>
public interface IImageTransformer
{
    /// <summary>Name of the transform for logging/display.</summary>
    string Name { get; }

    /// <summary>Apply the transform to an image tensor.</summary>
    /// <param name="input">The input image tensor to transform.</param>
    /// <returns>A new ImageTensor with the transform applied.</returns>
    ImageTensor Transform(ImageTensor input);
}
