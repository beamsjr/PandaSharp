using FluentAssertions;
using Cortex.Column;
using Cortex.Reshape;

namespace Cortex.Tests.Unit.Reshape;

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

    // -- Melt typed fast paths --

    [Fact]
    public void Melt_AllDoubleValueColumns_UsesTypedFastPath()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Charlie"]),
            new Column<double>("Math", [90.0, 85.0, 70.0]),
            new Column<double>("Science", [88.0, 92.0, 75.0]),
            new Column<double>("English", [80.0, 95.0, 65.0])
        );

        var melted = df.Melt(idVars: ["Name"], valueVars: ["Math", "Science", "English"],
            varName: "Subject", valueName: "Score");

        melted.RowCount.Should().Be(9); // 3 names * 3 subjects
        melted.ColumnNames.Should().Equal(["Name", "Subject", "Score"]);

        // Verify the value column is Column<double> (typed fast path, not boxed)
        melted["Score"].DataType.Should().Be(typeof(double));

        // Verify values: row order is (Alice,Math), (Alice,Science), (Alice,English), (Bob,Math), ...
        melted.GetColumn<double>("Score")[0].Should().Be(90.0);  // Alice Math
        melted.GetColumn<double>("Score")[1].Should().Be(88.0);  // Alice Science
        melted.GetColumn<double>("Score")[2].Should().Be(80.0);  // Alice English
        melted.GetColumn<double>("Score")[3].Should().Be(85.0);  // Bob Math
        melted.GetColumn<double>("Score")[4].Should().Be(92.0);  // Bob Science
        melted.GetColumn<double>("Score")[5].Should().Be(95.0);  // Bob English

        // Verify id column is repeated correctly
        melted.GetStringColumn("Name")[0].Should().Be("Alice");
        melted.GetStringColumn("Name")[2].Should().Be("Alice");
        melted.GetStringColumn("Name")[3].Should().Be("Bob");
        melted.GetStringColumn("Name")[6].Should().Be("Charlie");

        // Verify variable column
        melted.GetStringColumn("Subject")[0].Should().Be("Math");
        melted.GetStringColumn("Subject")[1].Should().Be("Science");
        melted.GetStringColumn("Subject")[2].Should().Be("English");
    }

    [Fact]
    public void Melt_AllIntValueColumns_UsesTypedFastPath()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["A", "B"]),
            new Column<int>("X", [1, 4]),
            new Column<int>("Y", [2, 5]),
            new Column<int>("Z", [3, 6])
        );

        var melted = df.Melt(idVars: ["Id"], valueVars: ["X", "Y", "Z"],
            varName: "var", valueName: "val");

        melted.RowCount.Should().Be(6); // 2 rows * 3 value vars
        melted["val"].DataType.Should().Be(typeof(int));

        melted.GetColumn<int>("val")[0].Should().Be(1);  // A, X
        melted.GetColumn<int>("val")[1].Should().Be(2);  // A, Y
        melted.GetColumn<int>("val")[2].Should().Be(3);  // A, Z
        melted.GetColumn<int>("val")[3].Should().Be(4);  // B, X
        melted.GetColumn<int>("val")[4].Should().Be(5);  // B, Y
        melted.GetColumn<int>("val")[5].Should().Be(6);  // B, Z
    }

    [Fact]
    public void Melt_MixedTypes_FallsBackCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("Id", ["A", "B"]),
            new Column<double>("X", [1.5, 2.5]),
            new Column<int>("Y", [10, 20])
        );

        var melted = df.Melt(idVars: ["Id"], valueVars: ["X", "Y"],
            varName: "var", valueName: "val");

        melted.RowCount.Should().Be(4); // 2 rows * 2 value vars

        // Mixed types should still produce correct values (via fallback path)
        var valCol = melted["val"];
        valCol.GetObject(0).Should().Be(1.5);  // A, X
        valCol.GetObject(1).Should().Be(10);   // A, Y
        valCol.GetObject(2).Should().Be(2.5);  // B, X
        valCol.GetObject(3).Should().Be(20);   // B, Y
    }

    [Fact]
    public void Melt_StringIdVars_PreservesValues()
    {
        var df = new DataFrame(
            new StringColumn("First", ["Alice", "Bob"]),
            new StringColumn("Last", ["Smith", "Jones"]),
            new Column<double>("Score1", [90.0, 80.0]),
            new Column<double>("Score2", [85.0, 75.0])
        );

        var melted = df.Melt(idVars: ["First", "Last"], valueVars: ["Score1", "Score2"],
            varName: "Test", valueName: "Score");

        melted.RowCount.Should().Be(4);

        // Verify both string id columns are preserved correctly
        melted.GetStringColumn("First")[0].Should().Be("Alice");
        melted.GetStringColumn("Last")[0].Should().Be("Smith");
        melted.GetStringColumn("First")[1].Should().Be("Alice");
        melted.GetStringColumn("Last")[1].Should().Be("Smith");
        melted.GetStringColumn("First")[2].Should().Be("Bob");
        melted.GetStringColumn("Last")[2].Should().Be("Jones");
        melted.GetStringColumn("First")[3].Should().Be("Bob");
        melted.GetStringColumn("Last")[3].Should().Be("Jones");

        // Verify values
        melted.GetColumn<double>("Score")[0].Should().Be(90.0);
        melted.GetColumn<double>("Score")[1].Should().Be(85.0);
        melted.GetColumn<double>("Score")[2].Should().Be(80.0);
        melted.GetColumn<double>("Score")[3].Should().Be(75.0);
    }
}
