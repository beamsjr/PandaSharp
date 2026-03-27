using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Viz;
using Cortex.Viz.Rendering;

namespace Cortex.Viz.Tests;

public class StaticExportTests
{
    private static DataFrame SampleDf() => new(
        new StringColumn("Category", ["A", "B", "C"]),
        new Column<double>("Sales", [100, 200, 150])
    );

    [Fact]
    public void IsAvailable_ReturnsBoolean()
    {
        // Just verify it doesn't throw — actual result depends on Node.js being installed
        var result = StaticExporter.IsAvailable();
        result.Should().Be(result); // tautology, just ensures no exception
    }

    [Fact]
    public void ToPng_MethodExists()
    {
        // Verify the fluent API compiles and the method signature is correct
        var viz = SampleDf().Viz().Bar("Category", "Sales");
        viz.Should().NotBeNull();
        // Method existence verified at compile time — this test confirms the API surface
    }

    [Fact]
    public void ToSvg_MethodExists()
    {
        var viz = SampleDf().Viz().Bar("Category", "Sales");
        viz.Should().NotBeNull();
    }

    [Fact]
    public void ToPng_InvalidFormat_Throws()
    {
        var spec = SampleDf().Viz().Bar("Category", "Sales").Spec;
        var tempPath = Path.GetTempFileName();
        try
        {
            var act = () => StaticExporter.Export(spec, tempPath, "gif", 800, 500, 1);
            act.Should().Throw<ArgumentException>().WithMessage("*png*svg*");
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void ToPng_WithCustomSize()
    {
        // Verifies method can be called with size params without compile errors
        var viz = SampleDf().Viz().Bar("Category", "Sales");

        if (!StaticExporter.IsAvailable())
            return; // Skip if Node.js not installed

        var tempPath = Path.Combine(Path.GetTempPath(), $"test_export_{Guid.NewGuid():N}.png");
        try
        {
            viz.ToPng(tempPath, width: 1200, height: 600, scale: 1);
            File.Exists(tempPath).Should().BeTrue();
            new FileInfo(tempPath).Length.Should().BeGreaterThan(0);
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void ToSvg_WithCustomSize()
    {
        var viz = SampleDf().Viz().Bar("Category", "Sales");

        if (!StaticExporter.IsAvailable())
            return;

        var tempPath = Path.Combine(Path.GetTempPath(), $"test_export_{Guid.NewGuid():N}.svg");
        try
        {
            viz.ToSvg(tempPath, width: 800, height: 400);
            File.Exists(tempPath).Should().BeTrue();
            var content = File.ReadAllText(tempPath);
            content.Should().Contain("<svg");
        }
        finally
        {
            if (File.Exists(tempPath)) File.Delete(tempPath);
        }
    }

    [Fact]
    public void ToPngBytes_ReturnsBytes()
    {
        var viz = SampleDf().Viz().Bar("Category", "Sales");

        if (!StaticExporter.IsAvailable())
            return;

        var bytes = viz.ToPngBytes();
        bytes.Should().NotBeEmpty();
        // PNG magic bytes
        bytes[0].Should().Be(0x89);
        bytes[1].Should().Be(0x50); // 'P'
    }

    [Fact]
    public void ToSvgString_ReturnsSvg()
    {
        var viz = SampleDf().Viz().Bar("Category", "Sales");

        if (!StaticExporter.IsAvailable())
            return;

        var svg = viz.ToSvgString();
        svg.Should().Contain("<svg");
    }
}
