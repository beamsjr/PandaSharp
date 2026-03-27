using FluentAssertions;
using Cortex.Column;

namespace Cortex.Tests.Unit;

public class DataFrameAdvancedTests
{
    // -- SelectDtypes / ExcludeDtypes --

    [Fact]
    public void SelectDtypes_FiltersColumns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25]),
            new Column<double>("Score", [90.0]),
            new Column<bool>("Active", [true])
        );

        var numeric = df.SelectDtypes(typeof(int), typeof(double));
        numeric.ColumnCount.Should().Be(2);
        numeric.ColumnNames.Should().Equal(["Age", "Score"]);
    }

    [Fact]
    public void ExcludeDtypes_ExcludesColumns()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice"]),
            new Column<int>("Age", [25]),
            new Column<double>("Score", [90.0])
        );

        var result = df.ExcludeDtypes(typeof(string));
        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().NotContain("Name");
    }

    // -- ApplyMap --

    [Fact]
    public void ApplyMap_TransformsEveryCell()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2]),
            new StringColumn("B", ["hello", "world"])
        );

        var result = df.ApplyMap(val => val?.ToString()?.ToUpperInvariant());

        result.GetStringColumn("A")[0].Should().Be("1");
        result.GetStringColumn("B")[0].Should().Be("HELLO");
    }

    [Fact]
    public void ApplyMap_HandlesNulls()
    {
        var df = new DataFrame(Column<int>.FromNullable("X", [1, null, 3]));
        var result = df.ApplyMap(val => val is null ? "MISSING" : val.ToString());

        result.GetStringColumn("X")[1].Should().Be("MISSING");
    }

    // -- Idxmin / Idxmax --

    [Fact]
    public void Idxmin_ReturnsColumnNameOfMin()
    {
        var df = new DataFrame(
            new Column<double>("Math", [90, 70, 85]),
            new Column<double>("Science", [80, 95, 60]),
            new Column<double>("English", [75, 80, 90])
        );

        var result = df.Idxmin();
        result[0].Should().Be("English"); // row 0: min=75 in English
        result[1].Should().Be("Math");    // row 1: min=70 in Math
        result[2].Should().Be("Science"); // row 2: min=60 in Science
    }

    [Fact]
    public void Idxmax_ReturnsColumnNameOfMax()
    {
        var df = new DataFrame(
            new Column<double>("Math", [90, 70, 85]),
            new Column<double>("Science", [80, 95, 60]),
            new Column<double>("English", [75, 80, 90])
        );

        var result = df.Idxmax();
        result[0].Should().Be("Math");    // row 0: max=90
        result[1].Should().Be("Science"); // row 1: max=95
        result[2].Should().Be("English"); // row 2: max=90
    }

    [Fact]
    public void Idxmin_WithNulls()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [null, 10.0]),
            new Column<double>("B", [5.0, 20.0])
        );

        var result = df.Idxmin();
        result[0].Should().Be("B"); // A is null, only B has a value
        result[1].Should().Be("A"); // A=10 < B=20
    }

    // -- Bool column Any/All/SumTrue --

    [Fact]
    public void BoolColumn_Any()
    {
        new Column<bool>("X", [false, false, true]).Any().Should().BeTrue();
        new Column<bool>("X", [false, false, false]).Any().Should().BeFalse();
    }

    [Fact]
    public void BoolColumn_All()
    {
        new Column<bool>("X", [true, true, true]).All().Should().BeTrue();
        new Column<bool>("X", [true, false, true]).All().Should().BeFalse();
    }

    [Fact]
    public void BoolColumn_SumTrue()
    {
        new Column<bool>("X", [true, false, true, true]).SumTrue().Should().Be(3);
    }

    [Fact]
    public void BoolColumn_Any_WithNulls()
    {
        Column<bool>.FromNullable("X", [false, null, true]).Any().Should().BeTrue();
        Column<bool>.FromNullable("X", [false, null, false]).Any().Should().BeFalse();
    }

    [Fact]
    public void BoolColumn_All_WithNulls()
    {
        // All non-null values are true → true
        Column<bool>.FromNullable("X", [true, null, true]).All().Should().BeTrue();
        Column<bool>.FromNullable("X", [true, null, false]).All().Should().BeFalse();
    }
}
