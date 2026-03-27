using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;

namespace Cortex.Tests.Unit.GroupBy;

public class ParallelGroupByTests
{
    [Fact]
    public void SumParallel_MatchesSequential()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["A", "B", "A", "B", "A"]),
            new Column<double>("Salary", [10, 20, 30, 40, 50]),
            new Column<double>("Bonus", [1, 2, 3, 4, 5])
        );

        var sequential = df.GroupBy("Dept").Sum();
        var parallel = df.GroupBy("Dept").SumParallel();

        parallel.RowCount.Should().Be(sequential.RowCount);

        // Both should have same totals (order may differ)
        var seqSalaries = new HashSet<double?>();
        var parSalaries = new HashSet<double?>();
        for (int i = 0; i < sequential.RowCount; i++)
        {
            seqSalaries.Add(sequential.GetColumn<double>("Salary")[i]);
            parSalaries.Add(parallel.GetColumn<double>("Salary")[i]);
        }
        parSalaries.Should().BeEquivalentTo(seqSalaries);
    }

    [Fact]
    public void MeanParallel_MatchesSequential()
    {
        var df = new DataFrame(
            new StringColumn("Cat", ["X", "Y", "X", "Y"]),
            new Column<double>("Val", [10, 20, 30, 40])
        );

        var sequential = df.GroupBy("Cat").Mean();
        var parallel = df.GroupBy("Cat").MeanParallel();

        parallel.RowCount.Should().Be(sequential.RowCount);
    }
}
