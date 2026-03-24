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
}
