using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameMemoryTests
{
    [Fact]
    public void Memory_ReturnsPositiveValue()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new Column<double>("B", [1.0, 2.0, 3.0]),
            new StringColumn("C", ["hello", "world", "!"])
        );

        df.Memory().Should().BeGreaterThan(0);
    }

    [Fact]
    public void Memory_IntColumn_CorrectEstimate()
    {
        var df = new DataFrame(new Column<int>("A", new int[1000]));
        // 1000 ints * 4 bytes + null bitmap (~125 bytes) = ~4125
        df.Memory().Should().BeInRange(4000, 4200);
    }

    [Fact]
    public void Memory_DoubleColumn_CorrectEstimate()
    {
        var df = new DataFrame(new Column<double>("A", new double[1000]));
        // 1000 doubles * 8 bytes + bitmap = ~8125
        df.Memory().Should().BeInRange(8000, 8200);
    }

    [Fact]
    public void Memory_CategoricalColumn_LessThanString()
    {
        var values = new string?[10_000];
        var cats = new[] { "A", "B", "C", "D", "E" };
        for (int i = 0; i < values.Length; i++) values[i] = cats[i % 5];

        var strDf = new DataFrame(new StringColumn("X", values));
        var catDf = new DataFrame(new CategoricalColumn("X", values));

        catDf.Memory().Should().BeLessThan(strDf.Memory());
    }

    [Fact]
    public void Memory_Empty_IsZero()
    {
        new DataFrame().Memory().Should().Be(0);
    }
}
