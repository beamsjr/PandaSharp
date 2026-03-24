using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Transformers;

namespace PandaSharp.ML.Tests;

public class OneHotUnseenTests
{
    [Fact]
    public void Ignore_UnseenCategory_ProducesAllZeros()
    {
        var trainDf = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"]),
            new Column<int>("Value", [1, 2])
        );

        var encoder = new OneHotEncoder("Color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("Color", ["Red", "Green"]),
            new Column<int>("Value", [1, 2])
        );

        var result = encoder.Transform(testDf);
        // "Green" unseen → all zeros (row 1)
        ((int?)((Column<int>)result["Color_Red"])[1]).Should().Be(0);
        ((int?)((Column<int>)result["Color_Blue"])[1]).Should().Be(0);
    }

    [Fact]
    public void Error_UnseenCategory_Throws()
    {
        var trainDf = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"]),
            new Column<int>("Value", [1, 2])
        );

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Error, "Color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("Color", ["Red", "Green"]),
            new Column<int>("Value", [1, 2])
        );

        var act = () => encoder.Transform(testDf);
        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*unseen*Green*");
    }

    [Fact]
    public void Indicator_UnseenCategory_AddsUnknownColumn()
    {
        var trainDf = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"]),
            new Column<int>("Value", [1, 2])
        );

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Indicator, "Color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("Color", ["Red", "Green", "Blue"]),
            new Column<int>("Value", [1, 2, 3])
        );

        var result = encoder.Transform(testDf);

        result.ColumnNames.Should().Contain("Color_unknown");
        var unknownCol = (Column<int>)result["Color_unknown"];
        ((int?)unknownCol[0]).Should().Be(0); // Red is known
        ((int?)unknownCol[1]).Should().Be(1); // Green is unseen
        ((int?)unknownCol[2]).Should().Be(0); // Blue is known
    }

    [Fact]
    public void Indicator_NoUnseen_UnknownColumnAllZeros()
    {
        var trainDf = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"]),
            new Column<int>("Value", [1, 2])
        );

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Indicator, "Color");
        encoder.Fit(trainDf);

        var result = encoder.Transform(trainDf);

        result.ColumnNames.Should().Contain("Color_unknown");
        var unknownCol = (Column<int>)result["Color_unknown"];
        ((int?)unknownCol[0]).Should().Be(0);
        ((int?)unknownCol[1]).Should().Be(0);
    }

    [Fact]
    public void Error_NoUnseen_DoesNotThrow()
    {
        var df = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"]),
            new Column<int>("Value", [1, 2])
        );

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Error, "Color");
        encoder.Fit(df);

        var act = () => encoder.Transform(df);
        act.Should().NotThrow();
    }

    [Fact]
    public void Indicator_MultipleUnseen()
    {
        var trainDf = new DataFrame(
            new StringColumn("Color", ["Red"]),
            new Column<int>("Value", [1])
        );

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Indicator, "Color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("Color", ["Green", "Blue", "Red"]),
            new Column<int>("Value", [1, 2, 3])
        );

        var result = encoder.Transform(testDf);
        var unknownCol = (Column<int>)result["Color_unknown"];
        ((int?)unknownCol[0]).Should().Be(1); // Green unseen
        ((int?)unknownCol[1]).Should().Be(1); // Blue unseen
        ((int?)unknownCol[2]).Should().Be(0); // Red known
    }
}
