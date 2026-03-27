using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit.Column;

public class MaskExtensionsTests
{
    [Fact]
    public void And_CombinesMasks()
    {
        bool[] a = [true, true, false, false];
        bool[] b = [true, false, true, false];
        a.And(b).Should().Equal([true, false, false, false]);
    }

    [Fact]
    public void Or_CombinesMasks()
    {
        bool[] a = [true, true, false, false];
        bool[] b = [true, false, true, false];
        a.Or(b).Should().Equal([true, true, true, false]);
    }

    [Fact]
    public void Not_InvertsMask()
    {
        bool[] a = [true, false, true];
        a.Not().Should().Equal([false, true, false]);
    }

    [Fact]
    public void Xor_ExclusiveOr()
    {
        bool[] a = [true, true, false, false];
        bool[] b = [true, false, true, false];
        a.Xor(b).Should().Equal([false, true, true, false]);
    }

    [Fact]
    public void CountTrue_CountsTrueValues()
    {
        bool[] a = [true, false, true, true];
        a.CountTrue().Should().Be(3);
    }

    [Fact]
    public void Any_ReturnsTrueIfAnyTrue()
    {
        new bool[] { false, false, true }.Any().Should().BeTrue();
        new bool[] { false, false, false }.Any().Should().BeFalse();
    }

    [Fact]
    public void All_ReturnsTrueIfAllTrue()
    {
        new bool[] { true, true, true }.All().Should().BeTrue();
        new bool[] { true, false, true }.All().Should().BeFalse();
    }

    [Fact]
    public void CombinedFilter_WorksWithDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("Age", [25, 30, 35, 40, 45]),
            new Column<double>("Salary", [50_000, 60_000, 70_000, 80_000, 90_000])
        );

        // Age > 30 AND Salary < 85000
        var mask = df.GetColumn<int>("Age").Gt(30)
            .And(df.GetColumn<double>("Salary").Lt(85_000));

        var filtered = df.Filter(mask);
        filtered.RowCount.Should().Be(2); // Age 35 & 40
    }

    [Fact]
    public void And_ThrowsOnLengthMismatch()
    {
        bool[] a = [true, false];
        bool[] b = [true];
        var act = () => a.And(b);
        act.Should().Throw<ArgumentException>();
    }
}
