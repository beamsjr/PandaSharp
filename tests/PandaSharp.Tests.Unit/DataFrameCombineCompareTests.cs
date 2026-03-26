using System.Text;
using FluentAssertions;
using PandaSharp.Column;

namespace PandaSharp.Tests.Unit;

public class DataFrameCombineCompareTests
{
    // ── CombineFirst ──

    [Fact]
    public void CombineFirst_ComplementaryNulls_MergesCorrectly()
    {
        // Primary has nulls in positions where fallback has values, and vice versa
        var primary = new DataFrame(
            Column<double>.FromNullable("A", [1.0, null, 3.0, null]),
            Column<int>.FromNullable("B", [null, 20, null, 40])
        );
        var fallback = new DataFrame(
            Column<double>.FromNullable("A", [10.0, 2.0, null, 4.0]),
            Column<int>.FromNullable("B", [10, null, 30, null])
        );

        var result = primary.CombineFirst(fallback);

        result.RowCount.Should().Be(4);

        // A: [1.0, 2.0, 3.0, 4.0] — primary where non-null, fallback otherwise
        result.GetColumn<double>("A")[0].Should().Be(1.0);
        result.GetColumn<double>("A")[1].Should().Be(2.0);
        result.GetColumn<double>("A")[2].Should().Be(3.0);
        result.GetColumn<double>("A")[3].Should().Be(4.0);

        // B: [10, 20, 30, 40]
        result.GetColumn<int>("B")[0].Should().Be(10);
        result.GetColumn<int>("B")[1].Should().Be(20);
        result.GetColumn<int>("B")[2].Should().Be(30);
        result.GetColumn<int>("B")[3].Should().Be(40);
    }

    [Fact]
    public void CombineFirst_UnmatchedColumnsFromOther_Included()
    {
        var primary = new DataFrame(
            Column<double>.FromNullable("A", [1.0, null])
        );
        var fallback = new DataFrame(
            Column<double>.FromNullable("A", [10.0, 2.0]),
            new Column<int>("Extra", [100, 200])
        );

        var result = primary.CombineFirst(fallback);

        result.ColumnNames.Should().Contain("Extra");
        result.GetColumn<int>("Extra")[0].Should().Be(100);
        result.GetColumn<int>("Extra")[1].Should().Be(200);

        // A merges correctly
        result.GetColumn<double>("A")[0].Should().Be(1.0);
        result.GetColumn<double>("A")[1].Should().Be(2.0);
    }

    [Fact]
    public void CombineFirst_StringColumns_MergesNulls()
    {
        var primary = new DataFrame(
            new StringColumn("Name", ["Alice", null, "Charlie", null])
        );
        var fallback = new DataFrame(
            new StringColumn("Name", ["X", "Bob", "Y", "Diana"])
        );

        var result = primary.CombineFirst(fallback);

        var col = result.GetStringColumn("Name");
        col[0].Should().Be("Alice");
        col[1].Should().Be("Bob");
        col[2].Should().Be("Charlie");
        col[3].Should().Be("Diana");
    }

    // ── Compare ──

    [Fact]
    public void Compare_ThreeDifferingCells_ReturnsCorrectDiff()
    {
        var df1 = new DataFrame(
            new Column<int>("X", [1, 2, 3, 4]),
            new StringColumn("Y", ["a", "b", "c", "d"])
        );
        var df2 = new DataFrame(
            new Column<int>("X", [1, 99, 3, 4]),
            new StringColumn("Y", ["a", "b", "changed", "DIFFERENT"])
        );

        var result = df1.Compare(df2);

        // Should have rows only where differences exist
        // Row 1 (X differs), Row 2 (Y differs), Row 3 (Y differs)
        result.RowCount.Should().Be(3);

        // X columns: diff at row index 1 only
        result.ColumnNames.Should().Contain("X_self");
        result.ColumnNames.Should().Contain("X_other");
        result.GetStringColumn("X_self")[0].Should().Be("2");
        result.GetStringColumn("X_other")[0].Should().Be("99");

        // Y columns: diff at row indices 2,3
        result.ColumnNames.Should().Contain("Y_self");
        result.ColumnNames.Should().Contain("Y_other");
    }

    [Fact]
    public void Compare_IdenticalDataFrames_ReturnsEmpty()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );
        var df2 = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var result = df1.Compare(df2);

        result.ColumnCount.Should().Be(0);
    }

    [Fact]
    public void Compare_WithAlignColumn_IncludesAlignColumn()
    {
        var df1 = new DataFrame(
            new StringColumn("ID", ["a", "b", "c"]),
            new Column<int>("Val", [1, 2, 3])
        );
        var df2 = new DataFrame(
            new StringColumn("ID", ["a", "b", "c"]),
            new Column<int>("Val", [1, 99, 3])
        );

        var result = df1.Compare(df2, alignColumn: "ID");

        result.ColumnNames.Should().Contain("ID");
        result.GetStringColumn("ID")[0].Should().Be("b");
        result.RowCount.Should().Be(1);
    }

    // ── Case-insensitive string operations ──

    [Fact]
    public void ContainsIgnoreCase_MixedCaseStrings()
    {
        var col = new StringColumn("S", ["Hello World", "HELLO", "goodbye", null, "hElLo"]);

        var result = col.Str.ContainsIgnoreCase("hello");

        result[0].Should().BeTrue();
        result[1].Should().BeTrue();
        result[2].Should().BeFalse();
        result.IsNull(3).Should().BeTrue();
        result[4].Should().BeTrue();
    }

    [Fact]
    public void StartsWithIgnoreCase_MixedCaseStrings()
    {
        var col = new StringColumn("S", ["Hello World", "HELLO", "goodbye", null]);

        var result = col.Str.StartsWithIgnoreCase("hello");

        result[0].Should().BeTrue();
        result[1].Should().BeTrue();
        result[2].Should().BeFalse();
        result.IsNull(3).Should().BeTrue();
    }

    [Fact]
    public void EndsWithIgnoreCase_MixedCaseStrings()
    {
        var col = new StringColumn("S", ["Hello WORLD", "test world", "goodbye", null]);

        var result = col.Str.EndsWithIgnoreCase("world");

        result[0].Should().BeTrue();
        result[1].Should().BeTrue();
        result[2].Should().BeFalse();
        result.IsNull(3).Should().BeTrue();
    }

    [Fact]
    public void ReplaceIgnoreCase_ReplacesAllCaseVariants()
    {
        var col = new StringColumn("S", ["Hello hello HELLO", "no match", null]);

        var result = col.Str.ReplaceIgnoreCase("hello", "hi");

        result[0].Should().Be("hi hi hi");
        result[1].Should().Be("no match");
        result.IsNull(2).Should().BeTrue();
    }

    // ── Unicode normalization ──

    [Fact]
    public void NormalizeUnicode_ComposedVsDecomposed()
    {
        // U+00E9 (e-acute composed) vs U+0065 U+0301 (e + combining acute)
        string composed = "\u00E9";        // e-acute as single char
        string decomposed = "e\u0301";     // e + combining acute

        var col = new StringColumn("S", [composed, decomposed, "ascii", null]);

        // Normalize to FormC (composed)
        var resultC = col.Str.NormalizeUnicode(NormalizationForm.FormC);
        resultC[0].Should().Be(composed);
        resultC[1].Should().Be(composed); // decomposed → composed
        resultC[2].Should().Be("ascii");
        resultC.IsNull(3).Should().BeTrue();

        // Normalize to FormD (decomposed)
        var resultD = col.Str.NormalizeUnicode(NormalizationForm.FormD);
        resultD[0].Should().Be(decomposed); // composed → decomposed
        resultD[1].Should().Be(decomposed);
        resultD[2].Should().Be("ascii");
        resultD.IsNull(3).Should().BeTrue();
    }
}
