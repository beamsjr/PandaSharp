using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Reshape;

namespace PandaSharp.Tests.Unit.Reshape;

public class ReshapeTests
{
    [Fact]
    public void Pivot_ConvertsLongToWide()
    {
        var df = new DataFrame(
            new StringColumn("Date", ["2024-01", "2024-01", "2024-02", "2024-02"]),
            new StringColumn("Product", ["A", "B", "A", "B"]),
            new Column<double>("Sales", [100.0, 200.0, 150.0, 250.0])
        );

        var pivoted = df.Pivot(index: "Date", columns: "Product", values: "Sales");

        pivoted.RowCount.Should().Be(2);
        pivoted.ColumnNames.Should().Contain("A");
        pivoted.ColumnNames.Should().Contain("B");
        pivoted.GetColumn<double>("A")[0].Should().Be(100.0);
        pivoted.GetColumn<double>("B")[1].Should().Be(250.0);
    }

    [Fact]
    public void Melt_ConvertsWideToLong()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<double>("Math", [90.0, 85.0]),
            new Column<double>("English", [80.0, 95.0])
        );

        var melted = df.Melt(idVars: ["Name"], valueVars: ["Math", "English"],
            varName: "Subject", valueName: "Score");

        melted.RowCount.Should().Be(4); // 2 names * 2 subjects
        melted.ColumnNames.Should().Equal(["Name", "Subject", "Score"]);
        melted.GetStringColumn("Subject")[0].Should().Be("Math");
        melted.GetStringColumn("Subject")[1].Should().Be("English");
    }

    [Fact]
    public void Melt_DefaultValueVars_UsesAllNonIdColumns()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["A"]),
            new Column<int>("X", [1]),
            new Column<int>("Y", [2])
        );

        var melted = df.Melt(idVars: ["Id"]);

        melted.RowCount.Should().Be(2);
        melted.ColumnNames.Should().Contain("variable");
        melted.ColumnNames.Should().Contain("value");
    }

    [Fact]
    public void GetDummies_CreatesOneHotEncoding()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice"]),
            new StringColumn("Color", ["Red", "Blue", "Green"])
        );

        var dummies = df.GetDummies("Color");

        dummies.ColumnNames.Should().Contain("Name"); // preserved
        dummies.ColumnNames.Should().NotContain("Color"); // replaced
        dummies.ColumnNames.Should().Contain("Color_Red");
        dummies.ColumnNames.Should().Contain("Color_Blue");
        dummies.ColumnNames.Should().Contain("Color_Green");

        dummies.GetColumn<bool>("Color_Red")[0].Should().Be(true);
        dummies.GetColumn<bool>("Color_Red")[1].Should().Be(false);
        dummies.GetColumn<bool>("Color_Blue")[1].Should().Be(true);
    }

    [Fact]
    public void GetDummies_CustomPrefix()
    {
        var df = new DataFrame(
            new StringColumn("Color", ["Red", "Blue"])
        );

        var dummies = df.GetDummies("Color", prefix: "c");

        dummies.ColumnNames.Should().Contain("c_Red");
        dummies.ColumnNames.Should().Contain("c_Blue");
    }
}
