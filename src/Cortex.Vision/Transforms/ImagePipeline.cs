using Cortex.ML.Tensors;

namespace Cortex.Vision.Transforms;

/// <summary>
/// Composable chain of image transforms applied sequentially.
/// Supports both single-image and per-image batch processing with independent randomness.
/// </summary>
public class ImagePipeline
{
    private readonly List<IImageTransformer> _transforms;

    /// <summary>The ordered list of transform steps in this pipeline.</summary>
    public IReadOnlyList<IImageTransformer> Steps => _transforms.AsReadOnly();

    /// <summary>Number of transforms in the pipeline.</summary>
    public int Count => _transforms.Count;

    /// <summary>Create an empty pipeline.</summary>
    public ImagePipeline()
    {
        _transforms = new List<IImageTransformer>();
    }

    /// <summary>Create a pipeline from a list of transforms.</summary>
    /// <param name="transforms">The transforms to apply in order.</param>
    public ImagePipeline(IEnumerable<IImageTransformer> transforms)
    {
        _transforms = new List<IImageTransformer>(transforms);
    }

    /// <summary>Create a pipeline from one or more transforms.</summary>
    /// <param name="transforms">The transforms to apply in order.</param>
    public ImagePipeline(params IImageTransformer[] transforms)
    {
        _transforms = new List<IImageTransformer>(transforms);
    }

    /// <summary>Add a transform to the end of the pipeline.</summary>
    /// <param name="transform">The transform to append.</param>
    /// <returns>This pipeline, for fluent chaining.</returns>
    public ImagePipeline Add(IImageTransformer transform)
    {
        _transforms.Add(transform);
        return this;
    }

    /// <summary>
    /// Applies all transform steps sequentially to the input tensor.
    /// </summary>
    /// <param name="input">The input image tensor.</param>
    /// <returns>The transformed image tensor.</returns>
    public ImageTensor Transform(ImageTensor input)
    {
        var result = input;
        foreach (var step in _transforms)
            result = step.Transform(result);
        return result;
    }

    /// <summary>
    /// Applies the pipeline per-image within a batch, so each image gets independent randomness.
    /// For non-batch tensors, delegates to <see cref="Transform"/>.
    /// </summary>
    /// <param name="batch">The input batch tensor.</param>
    /// <returns>A batch tensor with all images independently transformed.</returns>
    public ImageTensor TransformBatch(ImageTensor batch)
    {
        if (!batch.IsBatch)
            return Transform(batch);

        // Apply the pipeline to each image independently
        var transformed = new ImageTensor[batch.BatchSize];
        for (int i = 0; i < batch.BatchSize; i++)
        {
            var single = batch.GetImage(i);
            transformed[i] = Transform(single);
        }

        if (transformed.Length == 0)
            return batch;

        // Use the first transformed image to determine output dimensions
        var first = transformed[0];
        int outH = first.Height;
        int outW = first.Width;
        int outC = first.Channels;
        int singleLen = outH * outW * outC;

        var results = new float[batch.BatchSize * singleLen];
        for (int i = 0; i < batch.BatchSize; i++)
        {
            transformed[i].Span.CopyTo(results.AsSpan(i * singleLen, singleLen));
        }

        return new ImageTensor(
            new Tensor<float>(results, batch.BatchSize, outH, outW, outC),
            first.ChannelOrder);
    }

    /// <summary>
    /// Creates a new <see cref="ImagePipelineBuilder"/> for fluent pipeline construction.
    /// </summary>
    /// <returns>A new builder instance.</returns>
    public static ImagePipelineBuilder Create() => new();
}

/// <summary>
/// Fluent builder for constructing <see cref="ImagePipeline"/> instances.
/// </summary>
public class ImagePipelineBuilder
{
    private readonly List<IImageTransformer> _steps = new();

    /// <summary>
    /// Adds a custom transform step to the pipeline.
    /// </summary>
    /// <param name="transform">The transform to add.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder Add(IImageTransformer transform)
    {
        _steps.Add(transform);
        return this;
    }

    /// <summary>Adds a <see cref="Transforms.Resize"/> step.</summary>
    /// <param name="w">Target width.</param>
    /// <param name="h">Target height.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder Resize(int w, int h) => Add(new Resize(w, h));

    /// <summary>Adds a <see cref="Transforms.CenterCrop"/> step.</summary>
    /// <param name="w">Crop width.</param>
    /// <param name="h">Crop height.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder CenterCrop(int w, int h) => Add(new CenterCrop(w, h));

    /// <summary>Adds a <see cref="Transforms.RandomHorizontalFlip"/> step.</summary>
    /// <param name="p">Probability of flipping.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder RandomHorizontalFlip(double p = 0.5) => Add(new RandomHorizontalFlip(p));

    /// <summary>Adds a <see cref="Transforms.RandomVerticalFlip"/> step.</summary>
    /// <param name="p">Probability of flipping.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder RandomVerticalFlip(double p = 0.5) => Add(new RandomVerticalFlip(p));

    /// <summary>Adds a <see cref="Transforms.Normalize"/> step with explicit mean and std.</summary>
    /// <param name="mean">Per-channel mean values.</param>
    /// <param name="std">Per-channel standard deviation values.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder Normalize(float[] mean, float[] std) => Add(new Normalize(mean, std));

    /// <summary>Adds a pre-configured <see cref="Transforms.Normalize"/> preset (e.g. Normalize.ImageNet()).</summary>
    /// <param name="preset">The Normalize instance to add.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder Normalize(Normalize preset) => Add(preset);

    /// <summary>Adds a <see cref="Transforms.ColorJitter"/> step.</summary>
    /// <param name="b">Maximum brightness deviation.</param>
    /// <param name="c">Maximum contrast deviation.</param>
    /// <param name="s">Maximum saturation deviation.</param>
    /// <param name="h">Maximum hue deviation.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder ColorJitter(float b = 0, float c = 0, float s = 0, float h = 0) =>
        Add(new ColorJitter(b, c, s, h));

    /// <summary>Adds a <see cref="Transforms.GaussianBlur"/> step.</summary>
    /// <param name="sigma">Standard deviation of the Gaussian kernel.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder GaussianBlur(float sigma = 1f) => Add(new GaussianBlur(sigma));

    /// <summary>Adds a <see cref="Transforms.RandomRotation"/> step.</summary>
    /// <param name="degrees">Maximum rotation angle in degrees.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder RandomRotation(float degrees = 10f) => Add(new RandomRotation(degrees));

    /// <summary>Adds a <see cref="Transforms.RandomErasing"/> step.</summary>
    /// <param name="p">Probability of erasing.</param>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder RandomErasing(double p = 0.5) => Add(new RandomErasing(p));

    /// <summary>Adds a <see cref="Transforms.Grayscale"/> step.</summary>
    /// <returns>This builder for chaining.</returns>
    public ImagePipelineBuilder Grayscale() => Add(new Grayscale());

    /// <summary>
    /// Builds the configured pipeline.
    /// </summary>
    /// <returns>A new <see cref="ImagePipeline"/> with all configured steps.</returns>
    public ImagePipeline Build() => new(_steps.ToArray());
}
