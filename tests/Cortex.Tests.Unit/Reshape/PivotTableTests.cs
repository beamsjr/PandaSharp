using FluentAssertions;
using Cortex.Column;
using Cortex.Reshape;

namespace Cortex.Tests.Unit.Reshape;

public class PivotTableTests
{
    [Fact]
    public void PivotTable_Sum_Default()
    {
        var df = new DataFrame(
            new StringColumn("Region", ["East", "East", "West", "West", "East"]),
            new StringColumn("Product", ["A", "B", "A", "B", "A"]),
            new Column<double>("Sales", [100, 200, 150, 250, 50])
        );

        var result = df.PivotTable(index: "Region", columns: "Product", values: "Sales");

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("A");
        result.ColumnNames.Should().Contain("B");

        // East + A = 100 + 50 = 150, East + B = 200
        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Region")[i] == "East")
            {
                result.GetColumn<double>("A")[i].Should().Be(150);
                result.GetColumn<double>("B")[i].Should().Be(200);
            }
            else // West
            {
                result.GetColumn<double>("A")[i].Should().Be(150);
                result.GetColumn<double>("B")[i].Should().Be(250);
            }
        }
    }

    [Fact]
    public void PivotTable_Mean()
    {
        var df = new DataFrame(
            new StringColumn("Dept", ["Sales", "Sales", "Eng"]),
            new StringColumn("Quarter", ["Q1", "Q1", "Q1"]),
            new Column<double>("Revenue", [100, 200, 300])
        );

        var result = df.PivotTable("Dept", "Quarter", "Revenue",
            aggFunc: vals => vals.Average());

        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Dept")[i] == "Sales")
                result.GetColumn<double>("Q1")[i].Should().Be(150); // avg(100, 200)
            else
                result.GetColumn<double>("Q1")[i].Should().Be(300);
        }
    }

    [Fact]
    public void PivotTable_Count()
    {
        var df = new DataFrame(
            new StringColumn("Color", ["Red", "Red", "Blue", "Red"]),
            new StringColumn("Size", ["S", "M", "S", "S"]),
            new Column<double>("Qty", [1, 1, 1, 1])
        );

        var result = df.PivotTable("Color", "Size", "Qty",
            aggFunc: vals => vals.Count());

        // Red+S = 2, Red+M = 1, Blue+S = 1
        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("Color")[i] == "Red")
                result.GetColumn<double>("S")[i].Should().Be(2);
        }
    }

    [Fact]
    public void PivotTable_MissingCombinations_AreNull()
    {
        var df = new DataFrame(
            new StringColumn("A", ["x", "y"]),
            new StringColumn("B", ["1", "2"]),
            new Column<double>("V", [10, 20])
        );

        var result = df.PivotTable("A", "B", "V");

        // x+2 doesn't exist, should be null
        for (int i = 0; i < result.RowCount; i++)
        {
            if (result.GetStringColumn("A")[i] == "x")
                result["2"].IsNull(i).Should().BeTrue();
        }
    }
}
