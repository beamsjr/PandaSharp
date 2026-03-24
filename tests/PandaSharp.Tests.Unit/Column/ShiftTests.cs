using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class ShiftTests
{
    [Fact]
    public void Shift_Positive_LagsValues()
    {
        var col = new Column<int>("X", [10, 20, 30, 40]);
        var result = col.Shift(1);

        result[0].Should().BeNull(); // shifted in
        result[1].Should().Be(10);
        result[2].Should().Be(20);
        result[3].Should().Be(30);
    }

    [Fact]
    public void Shift_Negative_LeadsValues()
    {
        var col = new Column<int>("X", [10, 20, 30, 40]);
        var result = col.Shift(-1);

        result[0].Should().Be(20);
        result[1].Should().Be(30);
        result[2].Should().Be(40);
        result[3].Should().BeNull();
    }

    [Fact]
    public void Shift_Zero_ReturnsSameValues()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var result = col.Shift(0);
        result[0].Should().Be(1);
        result[1].Should().Be(2);
        result[2].Should().Be(3);
    }

    [Fact]
    public void Shift_LargerThanLength_AllNull()
    {
        var col = new Column<int>("X", [1, 2]);
        var result = col.Shift(5);
        result[0].Should().BeNull();
        result[1].Should().BeNull();
    }

    [Fact]
    public void Shift_String()
    {
        var col = new StringColumn("S", ["a", "b", "c"]);
        var result = col.Shift(1);
        result[0].Should().BeNull();
        result[1].Should().Be("a");
        result[2].Should().Be("b");
    }

    [Fact]
    public void Shift_WithNulls_PreservesExistingNulls()
    {
        var col = Column<int>.FromNullable("X", [1, null, 3]);
        var result = col.Shift(1);
        result[0].Should().BeNull(); // shifted in
        result[1].Should().Be(1);
        result[2].Should().BeNull(); // was null in original
    }

    [Fact]
    public void Shift_UsefulForPctChange()
    {
        // Demonstrate: pct_change = (current - shifted) / shifted
        var prices = new Column<double>("Price", [100.0, 110.0, 105.0, 115.0]);
        var lagged = prices.Shift(1);

        // Manual pct change using shift
        var pctChange = new double?[4];
        for (int i = 0; i < 4; i++)
        {
            if (prices[i].HasValue && lagged[i].HasValue && lagged[i]!.Value != 0)
                pctChange[i] = (prices[i]!.Value - lagged[i]!.Value) / lagged[i]!.Value;
        }

        pctChange[0].Should().BeNull();
        pctChange[1].Should().BeApproximately(0.1, 0.001);
    }
}
