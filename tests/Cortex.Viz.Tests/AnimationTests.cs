using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Viz;

namespace Cortex.Viz.Tests;

public class AnimationTests
{
    private static DataFrame TimeDf() => new(
        new StringColumn("Category", ["A", "B", "A", "B", "A", "B"]),
        new Column<double>("Sales", [100, 200, 150, 250, 180, 300]),
        new Column<int>("Year", [2021, 2021, 2022, 2022, 2023, 2023])
    );

    [Fact]
    public void Animate_Bar_ProducesFrames()
    {
        var html = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year")
            .ToHtmlString();

        html.Should().Contain("addFrames");
        html.Should().Contain("2021");
        html.Should().Contain("2022");
        html.Should().Contain("2023");
    }

    [Fact]
    public void Animate_Bar_ContainsSlider()
    {
        var html = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year")
            .ToHtmlString();

        html.Should().Contain("sliders");
        html.Should().Contain("updatemenus");
    }

    [Fact]
    public void Animate_Line_ProducesFrames()
    {
        var df = new DataFrame(
            new Column<double>("Month", [1, 2, 1, 2, 1, 2]),
            new Column<double>("Sales", [100, 200, 150, 250, 180, 300]),
            new Column<int>("Year", [2021, 2021, 2022, 2022, 2023, 2023])
        );

        var html = df.Viz()
            .Line("Month", "Sales")
            .Animate("Year")
            .ToHtmlString();

        html.Should().Contain("addFrames");
    }

    [Fact]
    public void Animate_Scatter_ProducesFrames()
    {
        var df = new DataFrame(
            new Column<double>("X", [1, 2, 3, 4, 5, 6]),
            new Column<double>("Y", [10, 20, 30, 40, 50, 60]),
            new Column<int>("Frame", [1, 1, 2, 2, 3, 3])
        );

        var html = df.Viz()
            .Scatter("X", "Y")
            .Animate("Frame")
            .ToHtmlString();

        html.Should().Contain("addFrames");
    }

    [Fact]
    public void Animate_Spec_HasCorrectFrameCount()
    {
        var viz = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year");

        viz.Spec.Frames.Should().HaveCount(3);
        viz.Spec.Frames[0].Name.Should().Be("2021");
        viz.Spec.Frames[1].Name.Should().Be("2022");
        viz.Spec.Frames[2].Name.Should().Be("2023");
    }

    [Fact]
    public void Animate_InitialTraces_ShowFirstFrame()
    {
        var viz = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year");

        // Initial traces should show first frame (2021) — 2 rows
        viz.Spec.Traces.Should().HaveCount(1);
    }

    [Fact]
    public void Animate_Fragment_ContainsFrames()
    {
        var fragment = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year")
            .ToHtmlFragment("animated_chart");

        fragment.Should().Contain("id=\"animated_chart\"");
        fragment.Should().Contain("addFrames");
    }

    [Fact]
    public void Animate_CustomDuration()
    {
        var html = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year", frameDurationMs: 1000, transitionDurationMs: 500)
            .ToHtmlString();

        html.Should().Contain("1000");
        html.Should().Contain("500");
    }

    [Fact]
    public void Animate_WithTitle()
    {
        var html = TimeDf().Viz()
            .Bar("Category", "Sales")
            .Animate("Year")
            .Title("Sales Animation")
            .ToHtmlString();

        html.Should().Contain("Sales Animation");
        html.Should().Contain("addFrames");
    }

    [Fact]
    public void Animate_SingleFrame_Works()
    {
        var df = new DataFrame(
            new StringColumn("X", ["A", "B"]),
            new Column<double>("Y", [10, 20]),
            new Column<int>("Frame", [1, 1])
        );

        var html = df.Viz()
            .Bar("X", "Y")
            .Animate("Frame")
            .ToHtmlString();

        html.Should().Contain("addFrames");
    }
}
