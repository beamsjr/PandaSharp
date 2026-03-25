using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Reshape;

namespace PandaSharp.Tests.Unit.Reshape;

public class CrossTabTests
{
    [Fact]
    public void CrossTab_ReturnsFrequencyTable()
    {
        var df = new DataFrame(
            new StringColumn("Gender", ["M", "F", "M", "F", "M"]),
            new StringColumn("Dept", ["Sales", "Eng", "Sales", "Sales", "Eng"])
        );

        var ct = df.CrossTab("Gender", "Dept");

        ct.ColumnNames.Should().Contain("Gender");
        ct.ColumnNames.Should().Contain("Sales");
        ct.ColumnNames.Should().Contain("Eng");

        // M: Sales=2, Eng=1
        // F: Sales=1, Eng=1
        for (int i = 0; i < ct.RowCount; i++)
        {
            var gender = ct.GetStringColumn("Gender")[i];
            if (gender == "M")
            {
                ct.GetColumn<int>("Sales")[i].Should().Be(2);
                ct.GetColumn<int>("Eng")[i].Should().Be(1);
            }
            else
            {
                ct.GetColumn<int>("Sales")[i].Should().Be(1);
                ct.GetColumn<int>("Eng")[i].Should().Be(1);
            }
        }
    }

    [Fact]
    public void CrossTab_HandlesZeroCounts()
    {
        var df = new DataFrame(
            new StringColumn("A", ["x", "x", "y"]),
            new StringColumn("B", ["1", "2", "1"])
        );

        var ct = df.CrossTab("A", "B");

        // y+2 should be 0
        for (int i = 0; i < ct.RowCount; i++)
        {
            if (ct.GetStringColumn("A")[i] == "y")
                ct.GetColumn<int>("2")[i].Should().Be(0);
        }
    }

    // -- CrossTab string fast path --

    [Fact]
    public void CrossTab_TwoStringColumns_ProducesCorrectCounts()
    {
        var df = new DataFrame(
            new StringColumn("Color", ["Red", "Blue", "Red", "Green", "Blue", "Red"]),
            new StringColumn("Size", ["S", "M", "L", "S", "M", "S"])
        );

        var ct = df.CrossTab("Color", "Size");

        // Red: S=2, L=1, M=0
        // Blue: M=2, S=0, L=0
        // Green: S=1, M=0, L=0
        ct.ColumnNames.Should().Contain("Color");
        ct.ColumnNames.Should().Contain("S");
        ct.ColumnNames.Should().Contain("M");
        ct.ColumnNames.Should().Contain("L");

        for (int i = 0; i < ct.RowCount; i++)
        {
            var color = ct.GetStringColumn("Color")[i];
            if (color == "Red")
            {
                ct.GetColumn<int>("S")[i].Should().Be(2);
                ct.GetColumn<int>("M")[i].Should().Be(0);
                ct.GetColumn<int>("L")[i].Should().Be(1);
            }
            else if (color == "Blue")
            {
                ct.GetColumn<int>("S")[i].Should().Be(0);
                ct.GetColumn<int>("M")[i].Should().Be(2);
                ct.GetColumn<int>("L")[i].Should().Be(0);
            }
            else if (color == "Green")
            {
                ct.GetColumn<int>("S")[i].Should().Be(1);
                ct.GetColumn<int>("M")[i].Should().Be(0);
                ct.GetColumn<int>("L")[i].Should().Be(0);
            }
        }
    }

    [Fact]
    public void CrossTab_OneStringOneNonString_FallsBackCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("Category", ["A", "A", "B", "B", "A"]),
            new Column<int>("Value", [1, 2, 1, 1, 2])
        );

        var ct = df.CrossTab("Category", "Value");

        // A: 1=1, 2=2
        // B: 1=2, 2=0
        ct.ColumnNames.Should().Contain("Category");
        ct.ColumnNames.Should().Contain("1");
        ct.ColumnNames.Should().Contain("2");

        for (int i = 0; i < ct.RowCount; i++)
        {
            var cat = ct.GetStringColumn("Category")[i];
            if (cat == "A")
            {
                ct.GetColumn<int>("1")[i].Should().Be(1);
                ct.GetColumn<int>("2")[i].Should().Be(2);
            }
            else if (cat == "B")
            {
                ct.GetColumn<int>("1")[i].Should().Be(2);
                ct.GetColumn<int>("2")[i].Should().Be(0);
            }
        }
    }
}
