using FluentAssertions;
using Cortex.ML.Tensors;
using Cortex.Vision;
using Cortex.Vision.Transforms;
using Xunit;

namespace Cortex.Vision.Tests.EdgeCases;

public class VisionEdgeCaseTests
{
    private static ImageTensor CreateTestImage(int h = 4, int w = 4, int c = 3)
    {
        var data = new float[h * w * c];
        for (int i = 0; i < data.Length; i++)
            data[i] = (i % 256) / 255f;
        return new ImageTensor(data, h, w, c);
    }

    // === ImageTensor edge cases ===

    [Fact]
    public void ImageTensor_ZeroHeight_ShouldThrow()
    {
        var act = () => new ImageTensor(Array.Empty<float>(), 0, 4, 3);
        act.Should().Throw<Exception>();
    }

    [Fact]
    public void ImageTensor_ZeroWidth_ShouldThrow()
    {
        var act = () => new ImageTensor(Array.Empty<float>(), 4, 0, 3);
        act.Should().Throw<Exception>();
    }

    [Fact]
    public void ImageTensor_ZeroChannels_ShouldThrow()
    {
        var act = () => new ImageTensor(Array.Empty<float>(), 4, 4, 0);
        act.Should().Throw<Exception>();
    }

    [Fact]
    public void ImageTensor_NegativeDimensions_ShouldThrow()
    {
        var act = () => new ImageTensor(new float[12], -1, 4, 3);
        act.Should().Throw<Exception>();
    }

    // === Resize edge cases ===

    [Fact]
    public void Resize_ToZeroDimension_ShouldThrow()
    {
        var act = () => new Resize(0, 10);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Resize_ToSameSize_ShouldReturnSameData()
    {
        var img = CreateTestImage(4, 4, 3);
        var resize = new Resize(4, 4);
        var result = resize.Transform(img);
        // Same dimensions means identity transform
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        result.Span.ToArray().Should().BeEquivalentTo(img.Span.ToArray());
    }

    [Fact]
    public void Resize_1x1Image_ShouldWork()
    {
        var img = new ImageTensor(new float[] { 0.5f, 0.3f, 0.1f }, 1, 1, 3);
        var resize = new Resize(4, 4);
        var result = resize.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        // All pixels should be approximately the original pixel value (bilinear interpolation of single pixel)
        for (int i = 0; i < result.Length; i += 3)
        {
            result.Span[i].Should().BeApproximately(0.5f, 0.01f);
            result.Span[i + 1].Should().BeApproximately(0.3f, 0.01f);
            result.Span[i + 2].Should().BeApproximately(0.1f, 0.01f);
        }
    }

    // === Normalize edge cases ===

    [Fact]
    public void Normalize_StdZero_ShouldClampNotInfinity()
    {
        var img = CreateTestImage(2, 2, 3);
        // std = 0 should be clamped to avoid division by zero
        var norm = new Normalize(new float[] { 0f, 0f, 0f }, new float[] { 0f, 0f, 0f });
        var result = norm.Transform(img);

        // Result should not contain Infinity or NaN
        foreach (var v in result.Span.ToArray())
        {
            float.IsInfinity(v).Should().BeFalse();
            float.IsNaN(v).Should().BeFalse();
        }
    }

    [Fact]
    public void Normalize_EmptyMeanStd_ShouldThrow()
    {
        var act = () => new Normalize(Array.Empty<float>(), Array.Empty<float>());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Normalize_MismatchedChannels_ShouldThrow()
    {
        var img = CreateTestImage(2, 2, 3);
        var norm = new Normalize(new float[] { 0.5f }, new float[] { 0.5f }); // 1 channel vs 3
        var act = () => norm.Transform(img);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Normalize_BatchInput_ShouldWork()
    {
        var data = new float[2 * 2 * 2 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5f;
        var batch = new ImageTensor(new Tensor<float>(data, 2, 2, 2, 3));

        var norm = new Normalize(new float[] { 0.5f, 0.5f, 0.5f }, new float[] { 0.5f, 0.5f, 0.5f });
        var result = norm.Transform(batch);
        result.IsBatch.Should().BeTrue();
        result.BatchSize.Should().Be(2);
        // (0.5 - 0.5) / 0.5 = 0
        foreach (var v in result.Span.ToArray())
            v.Should().BeApproximately(0f, 0.001f);
    }

    // === GaussianBlur edge cases ===

    [Fact]
    public void GaussianBlur_SigmaZero_ShouldThrow()
    {
        var act = () => new GaussianBlur(0f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GaussianBlur_NegativeSigma_ShouldThrow()
    {
        var act = () => new GaussianBlur(-1f);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void GaussianBlur_VerySmallSigma_ShouldApproxIdentity()
    {
        // Very small sigma should produce near-identity blur
        var img = CreateTestImage(4, 4, 3);
        var blur = new GaussianBlur(0.01f);
        var result = blur.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void GaussianBlur_SinglePixelImage_ShouldNotCrash()
    {
        var img = new ImageTensor(new float[] { 0.5f, 0.3f, 0.1f }, 1, 1, 3);
        var blur = new GaussianBlur(1.0f);
        var result = blur.Transform(img);
        result.Height.Should().Be(1);
        result.Width.Should().Be(1);
        // Single pixel blurred should stay approximately the same
        result.Span[0].Should().BeApproximately(0.5f, 0.01f);
    }

    // === RandomCrop edge cases ===

    [Fact]
    public void RandomCrop_LargerThanImage_ShouldThrow()
    {
        var img = CreateTestImage(4, 4, 3);
        var crop = new RandomCrop(8, 8, seed: 42);
        var act = () => crop.Transform(img);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void RandomCrop_SameSizeAsImage_ShouldReturnSameData()
    {
        var img = CreateTestImage(4, 4, 3);
        var crop = new RandomCrop(4, 4, seed: 42);
        var result = crop.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        result.Span.ToArray().Should().BeEquivalentTo(img.Span.ToArray());
    }

    [Fact]
    public void RandomCrop_ZeroSize_ShouldThrow()
    {
        var act = () => new RandomCrop(0, 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void RandomCrop_WithPadding_LargerThanImage_ShouldWork()
    {
        // With enough padding, crop larger than original image is valid
        var img = CreateTestImage(4, 4, 3);
        var crop = new RandomCrop(6, 6, padding: 2, seed: 42);
        var result = crop.Transform(img);
        result.Height.Should().Be(6);
        result.Width.Should().Be(6);
    }

    // === CenterCrop edge cases ===

    [Fact]
    public void CenterCrop_LargerThanImage_ShouldThrow()
    {
        var img = CreateTestImage(4, 4, 3);
        var crop = new CenterCrop(8, 8);
        var act = () => crop.Transform(img);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CenterCrop_SameSizeAsImage_ShouldReturnSameData()
    {
        var img = CreateTestImage(4, 4, 3);
        var crop = new CenterCrop(4, 4);
        var result = crop.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        result.Span.ToArray().Should().BeEquivalentTo(img.Span.ToArray());
    }

    [Fact]
    public void CenterCrop_ZeroSize_ShouldThrow()
    {
        var act = () => new CenterCrop(0, 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // === ImagePipeline edge cases ===

    [Fact]
    public void ImagePipeline_Empty_ShouldReturnInputUnchanged()
    {
        var img = CreateTestImage(4, 4, 3);
        var pipeline = new ImagePipeline();
        var result = pipeline.Transform(img);
        result.Span.ToArray().Should().BeEquivalentTo(img.Span.ToArray());
    }

    [Fact]
    public void ImagePipeline_SingleTransform_ShouldWork()
    {
        var img = CreateTestImage(8, 8, 3);
        var pipeline = new ImagePipeline(new CenterCrop(4, 4));
        var result = pipeline.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void ImagePipeline_TransformBatch_EmptyPipeline()
    {
        var data = new float[2 * 4 * 4 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5f;
        var batch = new ImageTensor(new Tensor<float>(data, 2, 4, 4, 3));

        var pipeline = new ImagePipeline();
        var result = pipeline.TransformBatch(batch);
        result.IsBatch.Should().BeTrue();
        result.BatchSize.Should().Be(2);
    }

    [Fact]
    public void ImagePipeline_Count_ReflectsSteps()
    {
        var pipeline = new ImagePipeline();
        pipeline.Count.Should().Be(0);

        pipeline.Add(new CenterCrop(4, 4));
        pipeline.Count.Should().Be(1);

        pipeline.Add(Normalize.ImageNet());
        pipeline.Count.Should().Be(2);
    }
}
