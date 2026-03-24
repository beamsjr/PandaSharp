using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit.Column;

public class CategoricalColumnTests
{
    [Fact]
    public void Constructor_DictionaryEncodesValues()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue", "Red", "Green", "Blue", "Red"]);

        col.Length.Should().Be(6);
        col.CategoryCount.Should().Be(3); // Red, Blue, Green
        col[0].Should().Be("Red");
        col[1].Should().Be("Blue");
        col[5].Should().Be("Red");
    }

    [Fact]
    public void Constructor_HandlesNulls()
    {
        var col = new CategoricalColumn("Color", ["Red", null, "Blue", null]);

        col.NullCount.Should().Be(2);
        col[0].Should().Be("Red");
        col[1].Should().BeNull();
        col.IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void MemorySavings_ComparedToStringColumn()
    {
        // 10000 values from 5 categories
        var values = new string?[10_000];
        var cats = new[] { "Alpha", "Beta", "Gamma", "Delta", "Epsilon" };
        for (int i = 0; i < values.Length; i++)
            values[i] = cats[i % cats.Length];

        var strCol = new StringColumn("Data", values);
        var catCol = new CategoricalColumn("Data", values);

        // Categorical should use much less memory
        // StringColumn: ~10000 * (5 chars * 2 bytes + ref) ≈ 180 KB
        // CategoricalColumn: 5 * ~10 bytes + 10000 * 4 bytes ≈ 40 KB
        catCol.EstimatedBytes.Should().BeLessThan(50_000);
        catCol.CategoryCount.Should().Be(5);
    }

    [Fact]
    public void Filter_PreservesDictionary()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue", "Red", "Green"]);
        bool[] mask = [true, false, true, false];

        var filtered = (CategoricalColumn)col.Filter(mask);

        filtered.Length.Should().Be(2);
        filtered[0].Should().Be("Red");
        filtered[1].Should().Be("Red");
    }

    [Fact]
    public void Slice_Works()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue", "Green", "Red"]);
        var sliced = (CategoricalColumn)col.Slice(1, 2);

        sliced.Length.Should().Be(2);
        sliced[0].Should().Be("Blue");
        sliced[1].Should().Be("Green");
    }

    [Fact]
    public void TakeRows_Works()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue", "Green"]);
        int[] indices = [2, 0];

        var taken = (CategoricalColumn)col.TakeRows(indices);

        taken.Length.Should().Be(2);
        taken[0].Should().Be("Green");
        taken[1].Should().Be("Red");
    }

    [Fact]
    public void ToStringColumn_Expands()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue", "Red"]);
        var str = col.ToStringColumn();

        str.Should().BeOfType<StringColumn>();
        str.Length.Should().Be(3);
        str[0].Should().Be("Red");
    }

    [Fact]
    public void AsCategorical_ConvertsStringColumn()
    {
        var str = new StringColumn("Color", ["A", "B", "A", "C", "B"]);
        var cat = str.AsCategorical();

        cat.CategoryCount.Should().Be(3);
        cat[0].Should().Be("A");
        cat[4].Should().Be("B");
    }

    [Fact]
    public void InDataFrame_WorksLikeAnyColumn()
    {
        var df = new DataFrame(
            new CategoricalColumn("Dept", ["Sales", "Eng", "Sales", "Eng"]),
            new Column<double>("Salary", [50_000, 60_000, 70_000, 80_000])
        );

        df.RowCount.Should().Be(4);
        df["Dept"].GetObject(0).Should().Be("Sales");
        df["Dept"].DataType.Should().Be(typeof(string));
        df["Dept"].Should().BeOfType<CategoricalColumn>();
    }

    [Fact]
    public void Clone_CreatesIndependentCopy()
    {
        var col = new CategoricalColumn("Color", ["Red", "Blue"]);
        var clone = (CategoricalColumn)col.Clone("NewName");

        clone.Name.Should().Be("NewName");
        clone[0].Should().Be("Red");
    }
}
