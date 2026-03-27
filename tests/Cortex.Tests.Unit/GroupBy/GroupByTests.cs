using FluentAssertions;
using Cortex.Column;
using Cortex.GroupBy;

namespace Cortex.Tests.Unit.GroupBy;

public class GroupByTests
{
    private static DataFrame CreateSampleDf() => new(
        new StringColumn("Dept", ["Sales", "Eng", "Sales", "Eng", "Sales"]),
        new Column<int>("Age", [25, 30, 35, 28, 42]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000])
    );

    [Fact]
    public void GroupBy_SingleColumn_CreatesGroups()
    {
        var grouped = CreateSampleDf().GroupBy("Dept");
        grouped.GroupCount.Should().Be(2);
    }

    [Fact]
    public void GroupBy_GetGroup_ReturnsCorrectRows()
    {
        var grouped = CreateSampleDf().GroupBy("Dept");
        var salesGroup = grouped.GetGroup(new GroupKey(["Sales"]));
        salesGroup.RowCount.Should().Be(3);
    }

    [Fact]
    public void GroupBy_Sum_ReturnsPerGroupSum()
    {
        var result = CreateSampleDf().GroupBy("Dept").Sum();

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Dept");
        result.ColumnNames.Should().Contain("Salary");

        // Find the Eng row
        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Eng")
                result.GetColumn<double>("Salary")[i].Should().Be(120_000);
            else
                result.GetColumn<double>("Salary")[i].Should().Be(216_000);
        }
    }

    [Fact]
    public void GroupBy_Mean_ReturnsPerGroupMean()
    {
        var result = CreateSampleDf().GroupBy("Dept").Mean();

        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Eng")
                result.GetColumn<double>("Salary")[i].Should().Be(60_000);
            else
                result.GetColumn<double>("Salary")[i].Should().Be(72_000);
        }
    }

    [Fact]
    public void GroupBy_Count_ReturnsPerGroupCount()
    {
        var result = CreateSampleDf().GroupBy("Dept").Count();

        result.ColumnNames.Should().Contain("Dept");
        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Eng")
                result.GetColumn<int>("Salary")[i].Should().Be(2);
            else
                result.GetColumn<int>("Salary")[i].Should().Be(3);
        }
    }

    [Fact]
    public void GroupBy_Min_ReturnsPerGroupMin()
    {
        var result = CreateSampleDf().GroupBy("Dept").Min();

        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Eng")
                result.GetColumn<double>("Salary")[i].Should().Be(58_000);
            else
                result.GetColumn<double>("Salary")[i].Should().Be(50_000);
        }
    }

    [Fact]
    public void GroupBy_Max_ReturnsPerGroupMax()
    {
        var result = CreateSampleDf().GroupBy("Dept").Max();

        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Eng")
                result.GetColumn<double>("Salary")[i].Should().Be(62_000);
            else
                result.GetColumn<double>("Salary")[i].Should().Be(91_000);
        }
    }

    [Fact]
    public void GroupBy_First_ReturnsFirstRowPerGroup()
    {
        var result = CreateSampleDf().GroupBy("Dept").First();
        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Age");
    }

    [Fact]
    public void GroupBy_Last_ReturnsLastRowPerGroup()
    {
        var result = CreateSampleDf().GroupBy("Dept").Last();
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void GroupBy_MultipleKeys()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["Sales", "Eng", "Sales", "Eng"]),
            new StringColumn("Level", ["Jr", "Jr", "Sr", "Sr"]),
            new Column<double>("Salary", [50_000, 60_000, 80_000, 90_000])
        );

        var result = df.GroupBy("Dept", "Level").Sum();
        result.RowCount.Should().Be(4); // 4 unique dept+level combos
    }

    [Fact]
    public void GroupBy_NamedAgg_AppliesDifferentFunctions()
    {
        var result = CreateSampleDf().GroupBy("Dept").Agg(b => b
            .Sum("Salary", alias: "TotalSalary")
            .Mean("Age", alias: "AvgAge")
            .Count("Age", alias: "HeadCount")
        );

        result.ColumnNames.Should().Contain("Dept");
        result.ColumnNames.Should().Contain("TotalSalary");
        result.ColumnNames.Should().Contain("AvgAge");
        result.ColumnNames.Should().Contain("HeadCount");
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void GroupBy_Filter_DropsGroups()
    {
        var result = CreateSampleDf().GroupBy("Dept")
            .Filter(group => group.RowCount >= 3);

        // Only Sales has 3 rows
        result.RowCount.Should().Be(3);
        for (int i = 0; i < result.RowCount; i++)
            result.GetStringColumn("Dept")[i].Should().Be("Sales");
    }

    [Fact]
    public void GroupBy_Transform_ReturnsSameShape()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["Sales", "Eng", "Sales"]),
            new Column<double>("Salary", [50_000, 60_000, 80_000])
        );

        var result = df.GroupBy("Dept").Transform(group =>
        {
            // Normalize salary within group by subtracting group mean
            var sal = group.GetColumn<double>("Salary");
            var mean = sal.Mean()!.Value;
            var normalized = new double[group.RowCount];
            for (int i = 0; i < group.RowCount; i++)
                normalized[i] = sal[i]!.Value - mean;
            return new DataFrame(new Column<double>("Salary", normalized));
        });

        result.RowCount.Should().Be(3); // same as original
        result.ColumnNames.Should().Contain("Dept");
        result.ColumnNames.Should().Contain("Salary");

        // Sales mean = 65000, so first Sales row = 50000 - 65000 = -15000
        result.GetColumn<double>("Salary")[0].Should().Be(-15_000);
    }

    [Fact]
    public void GroupBy_Apply_ConcatenatesResults()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["Sales", "Eng", "Sales"]),
            new Column<double>("Salary", [50_000, 60_000, 80_000])
        );

        var result = df.GroupBy("Dept").Apply(group => group.Head(1));

        // One row per group
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void GroupBy_ThrowsOnMissingColumn()
    {
        var act = () => CreateSampleDf().GroupBy("Missing");
        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void GroupBy_ThrowsOnNoColumns()
    {
        var act = () => CreateSampleDf().GroupBy();
        act.Should().Throw<ArgumentException>();
    }
}
