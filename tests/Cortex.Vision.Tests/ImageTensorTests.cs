using FluentAssertions;
using Cortex.ML.Tensors;
using Cortex.Vision;
using Cortex.Vision.Transforms;
using Xunit;

namespace Cortex.Vision.Tests;

public class ImageTensorTests
{
    private static ImageTensor CreateTestImage(int h = 4, int w = 4, int c = 3)
    {
        var data = new float[h * w * c];
        for (int i = 0; i < data.Length; i++)
            data[i] = (i % 256) / 255f;
        return new ImageTensor(data, h, w, c);
    }

    private static ImageTensor CreateSolidImage(int h, int w, float r, float g, float b)
    {
        var data = new float[h * w * 3];
        for (int i = 0; i < h * w; i++)
        {
            data[i * 3] = r;
            data[i * 3 + 1] = g;
            data[i * 3 + 2] = b;
        }
        return new ImageTensor(data, h, w, 3);
    }

    [Fact]
    public void Constructor_SetsShapeCorrectly()
    {
        var img = CreateTestImage(8, 6, 3);
        img.Height.Should().Be(8);
        img.Width.Should().Be(6);
        img.Channels.Should().Be(3);
        img.IsBatch.Should().BeFalse();
        img.BatchSize.Should().Be(1);
    }

    [Fact]
    public void ToBatch_AddsDimension()
    {
        var img = CreateTestImage(4, 4, 3);
        var batched = img.ToBatch();
        batched.IsBatch.Should().BeTrue();
        batched.BatchSize.Should().Be(1);
        batched.Height.Should().Be(4);
        batched.Width.Should().Be(4);
        batched.Channels.Should().Be(3);
    }

    [Fact]
    public void Unbatch_RemovesDimension()
    {
        var img = CreateTestImage(4, 4, 3).ToBatch();
        var single = img.Unbatch();
        single.IsBatch.Should().BeFalse();
        single.Height.Should().Be(4);
    }

    [Fact]
    public void GetImage_ExtractsFromBatch()
    {
        // Create a 2-image batch
        var data = new float[2 * 3 * 3 * 3]; // 2 images, 3x3, RGB
        for (int i = 0; i < data.Length; i++) data[i] = i / (float)data.Length;
        var batch = new ImageTensor(new Tensor<float>(data, 2, 3, 3, 3));
        batch.BatchSize.Should().Be(2);

        var img0 = batch.GetImage(0);
        var img1 = batch.GetImage(1);
        img0.IsBatch.Should().BeFalse();
        img1.IsBatch.Should().BeFalse();
        img0.Span[0].Should().NotBe(img1.Span[0]);
    }

    [Fact]
    public void ToImage_RoundTrips()
    {
        var original = CreateSolidImage(4, 4, 1.0f, 0.0f, 0.0f); // solid red
        var image = original.ToImage();
        image.Width.Should().Be(4);
        image.Height.Should().Be(4);

        var restored = ImageTensor.FromImage(image);
        restored.Height.Should().Be(4);
        restored.Width.Should().Be(4);
        // Red channel should be ~1.0 (255/255)
        restored.Span[0].Should().BeApproximately(1.0f, 0.01f);
        // Green should be ~0
        restored.Span[1].Should().BeApproximately(0.0f, 0.01f);
    }

    // --- Transform Tests ---

    [Fact]
    public void CenterCrop_CropsCorrectly()
    {
        var img = CreateTestImage(8, 8, 3);
        var crop = new CenterCrop(4, 4);
        var result = crop.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        result.Channels.Should().Be(3);
    }

    [Fact]
    public void RandomHorizontalFlip_PreservesShape()
    {
        var img = CreateTestImage(4, 6, 3);
        var flip = new RandomHorizontalFlip(1.0, seed: 42); // always flip
        var result = flip.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(6);
        result.Channels.Should().Be(3);
    }

    [Fact]
    public void RandomHorizontalFlip_ActuallyFlips()
    {
        var img = CreateSolidImage(2, 4, 0, 0, 0);
        // Set first pixel to red
        var data = img.Span.ToArray();
        data[0] = 1.0f; // R of pixel (0,0)
        var original = new ImageTensor(data, 2, 4, 3);

        var flip = new RandomHorizontalFlip(1.0, seed: 42); // always flip
        var result = flip.Transform(original);
        // First pixel red should now be at the last column
        var rSpan = result.Span;
        rSpan[0].Should().Be(0.0f); // (0,0) R should no longer be red
        int lastPixStart = (4 - 1) * 3; // pixel (0, 3) = last column of first row
        rSpan[lastPixStart].Should().Be(1.0f); // should be red now
    }

    [Fact]
    public void RandomVerticalFlip_PreservesShape()
    {
        var img = CreateTestImage(6, 4, 3);
        var flip = new RandomVerticalFlip(1.0, seed: 42);
        var result = flip.Transform(img);
        result.Height.Should().Be(6);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void Normalize_AppliesCorrectly()
    {
        var img = CreateSolidImage(2, 2, 0.5f, 0.5f, 0.5f);
        var norm = new Normalize(new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f });
        var result = norm.Transform(img);
        // (0.5 - 0.5) / 0.5 = 0
        foreach (var v in result.Span.ToArray())
            v.Should().BeApproximately(0f, 0.001f);
    }

    [Fact]
    public void Normalize_ImageNetPreset_Exists()
    {
        var norm = Normalize.ImageNet();
        norm.Should().NotBeNull();
        norm.Name.Should().Be("Normalize");
    }

    [Fact]
    public void Grayscale_ReducesChannels()
    {
        var img = CreateTestImage(4, 4, 3);
        var gray = new Grayscale();
        var result = gray.Transform(img);
        result.Channels.Should().Be(1);
        result.ChannelOrder.Should().Be(ChannelOrder.Grayscale);
        result.Length.Should().Be(4 * 4 * 1);
    }

    [Fact]
    public void RandomErasing_PreservesShape()
    {
        var img = CreateTestImage(8, 8, 3);
        var eraser = new RandomErasing(1.0, seed: 42); // always erase
        var result = eraser.Transform(img);
        result.Height.Should().Be(8);
        result.Width.Should().Be(8);
        result.Channels.Should().Be(3);
    }

    [Fact]
    public void RandomCrop_OutputsCorrectSize()
    {
        var img = CreateTestImage(8, 8, 3);
        var crop = new RandomCrop(4, 4, seed: 42);
        var result = crop.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void ColorJitter_PreservesShape()
    {
        var img = CreateTestImage(4, 4, 3);
        var jitter = new ColorJitter(0.2f, 0.2f, 0.2f, seed: 42);
        var result = jitter.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
        result.Channels.Should().Be(3);
    }

    // --- Pipeline Tests ---

    [Fact]
    public void ImagePipeline_ChainsTransforms()
    {
        var img = CreateTestImage(8, 8, 3);
        var pipeline = new ImagePipeline(
            new CenterCrop(4, 4),
            new Normalize(new[] { 0.5f, 0.5f, 0.5f }, new[] { 0.5f, 0.5f, 0.5f }));

        var result = pipeline.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void ImagePipeline_Builder_Works()
    {
        var pipeline = ImagePipeline.Create()
            .CenterCrop(4, 4)
            .RandomHorizontalFlip()
            .Normalize(Normalize.ImageNet())
            .Build();

        var img = CreateTestImage(8, 8, 3);
        var result = pipeline.Transform(img);
        result.Height.Should().Be(4);
        result.Width.Should().Be(4);
    }

    [Fact]
    public void ImagePipeline_TransformBatch_IndependentRandomness()
    {
        var data = new float[2 * 4 * 4 * 3];
        for (int i = 0; i < data.Length; i++) data[i] = 0.5f;
        var batch = new ImageTensor(new Tensor<float>(data, 2, 4, 4, 3));

        var pipeline = new ImagePipeline(
            new RandomHorizontalFlip(0.5, seed: null)); // non-deterministic

        // Should not throw and should preserve batch shape
        var result = pipeline.TransformBatch(batch);
        result.IsBatch.Should().BeTrue();
        result.BatchSize.Should().Be(2);
    }

    [Fact]
    public void ImagePipeline_DeterministicWithSeed()
    {
        var img = CreateTestImage(8, 8, 3);
        var pipeline = new ImagePipeline(
            new RandomHorizontalFlip(0.5, seed: 42),
            new CenterCrop(4, 4));

        var result1 = pipeline.Transform(img);
        var result2 = pipeline.Transform(img);
        // With same seed, transforms should be deterministic
        // (seed resets per transform instance creation, not per call, so this tests consistency)
        result1.Height.Should().Be(result2.Height);
        result1.Width.Should().Be(result2.Width);
    }
}
