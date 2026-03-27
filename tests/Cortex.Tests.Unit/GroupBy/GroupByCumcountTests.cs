using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;

namespace Cortex.Tests.Unit.GroupBy;

public class GroupByCumcountTests
{
    [Fact]
    public void Cumcount_ReturnsPerGroupRunningCount()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["A", "B", "A", "B", "A"]),
            new Column<int>("Val", [1, 2, 3, 4, 5])
        );

        var cc = df.GroupBy("Dept").Cumcount();

        // A rows at indices 0, 2, 4 → cumcount 0, 1, 2
        // B rows at indices 1, 3 → cumcount 0, 1
        cc[0].Should().Be(0); // A first
        cc[1].Should().Be(0); // B first
        cc[2].Should().Be(1); // A second
        cc[3].Should().Be(1); // B second
        cc[4].Should().Be(2); // A third
    }

    [Fact]
    public void Ngroup_ReturnsGroupNumber()
    {
        var df = new DataFrame(
            new StringColumn("Cat", ["X", "Y", "X", "Z", "Y"]),
            new Column<int>("Val", [1, 2, 3, 4, 5])
        );

        var ng = df.GroupBy("Cat").Ngroup();
        // X=group 0, Y=group 1, Z=group 2
        ng[0].Should().Be(ng[2]); // both X
        ng[1].Should().Be(ng[4]); // both Y
        ng[0].Should().NotBe(ng[1]); // X != Y
    }

    [Fact]
    public void Cumcount_SingleGroup()
    {
        var df = new DataFrame(
            new StringColumn("Key", ["A", "A", "A"]),
            new Column<int>("Val", [1, 2, 3])
        );

        var cc = df.GroupBy("Key").Cumcount();
        cc[0].Should().Be(0);
        cc[1].Should().Be(1);
        cc[2].Should().Be(2);
    }

    [Fact]
    public void Cumcount_UsefulForRanking()
    {
        var df = new DataFrame(
            new StringColumn("Team", ["A", "A", "B", "A", "B"]),
            new Column<double>("Score", [90, 85, 95, 80, 88])
        );

        // Sort within groups by adding cumcount
        var cc = df.GroupBy("Team").Cumcount();
        cc.Length.Should().Be(5);
    }
}
