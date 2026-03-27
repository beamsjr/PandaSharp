using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Indexing;

public class IndexingTests
{
    private static DataFrame SampleDf() => new(
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"]),
        new Column<int>("Age", [25, 30, 35, 28, 42])
    );

    [Fact]
    public void Loc_SingleRow()
    {
        var row = SampleDf().Loc[0];
        row.GetString("Name").Should().Be("Alice");
    }

    [Fact]
    public void Loc_Range()
    {
        var result = SampleDf().Loc[1..3];
        result.RowCount.Should().Be(2);
        result.GetStringColumn("Name")[0].Should().Be("Bob");
        result.GetStringColumn("Name")[1].Should().Be("Charlie");
    }

    [Fact]
    public void Loc_BoolMask()
    {
        var df = SampleDf();
        var mask = df.GetColumn<int>("Age").Gt(30);
        var result = df.Loc[mask];

        result.RowCount.Should().Be(2);
        result.GetStringColumn("Name")[0].Should().Be("Charlie");
    }

    [Fact]
    public void ILoc_SingleRow()
    {
        var row = SampleDf().ILoc[2];
        row.GetString("Name").Should().Be("Charlie");
    }

    [Fact]
    public void ILoc_Range()
    {
        var result = SampleDf().ILoc[..2];
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void ILoc_ExplicitIndices()
    {
        var result = SampleDf().ILoc[[4, 0, 2]];
        result.RowCount.Should().Be(3);
        result.GetStringColumn("Name")[0].Should().Be("Eve");
        result.GetStringColumn("Name")[1].Should().Be("Alice");
        result.GetStringColumn("Name")[2].Should().Be("Charlie");
    }

    [Fact]
    public void Slice_ReturnsSubset()
    {
        var result = SampleDf().Slice(1, 3);
        result.RowCount.Should().Be(3);
        result.GetStringColumn("Name")[0].Should().Be("Bob");
        result.GetStringColumn("Name")[2].Should().Be("Diana");
    }

    [Fact]
    public void Slice_ThrowsOnOutOfRange()
    {
        var act = () => SampleDf().Slice(3, 5);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Sample_ReturnsNRows()
    {
        var result = SampleDf().Sample(3, seed: 42);
        result.RowCount.Should().Be(3);
    }

    [Fact]
    public void Sample_ClampsToDfSize()
    {
        var result = SampleDf().Sample(100);
        result.RowCount.Should().Be(5);
    }

    [Fact]
    public void Sample_WithSeed_IsDeterministic()
    {
        var df = SampleDf();
        var r1 = df.Sample(3, seed: 42);
        var r2 = df.Sample(3, seed: 42);

        for (int i = 0; i < 3; i++)
            r1.GetStringColumn("Name")[i].Should().Be(r2.GetStringColumn("Name")[i]);
    }

    [Fact]
    public void ToHtml_ProducesHtmlTable()
    {
        var html = SampleDf().ToHtml();
        html.Should().Contain("<table");
        html.Should().Contain("Alice");
        html.Should().Contain("5 rows");
    }
}
