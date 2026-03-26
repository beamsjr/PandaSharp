using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Joins;
using PandaSharp.GroupBy;
using PandaSharp.Concat;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class CoreEdgeCaseTests
{
    // ===================================================================
    // Bug 1: RenameColumn to an existing column name silently creates
    //        a DataFrame that throws on construction due to duplicate names.
    //        It should throw a clear ArgumentException instead.
    // ===================================================================
    [Fact]
    public void RenameColumn_ToExistingName_ShouldThrowClearError()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2]),
            new Column<int>("B", [3, 4])
        );

        var act = () => df.RenameColumn("A", "B");

        act.Should().Throw<ArgumentException>()
            .WithMessage("*'B'*already exists*");
    }

    // ===================================================================
    // Bug 2: GroupBy on string column conflates null and empty string "".
    //        BuildStringGroups uses `vals[r] ?? ""` which maps null to "".
    //        Null and "" should be separate groups.
    // ===================================================================
    [Fact]
    public void GroupBy_StringColumn_NullAndEmptyString_ShouldBeSeparateGroups()
    {
        var df = new DataFrame(
            new StringColumn("Key", [null, "", "A", null, ""]),
            new Column<int>("Val", [1, 2, 3, 4, 5])
        );

        var grouped = df.GroupBy("Key");

        // Should have 3 groups: null, "", "A"
        grouped.GroupCount.Should().Be(3,
            "null and empty string should be distinct groups");
    }

    // ===================================================================
    // Bug 3: Generic join path — overlap detection misses case where a
    //        left key column name collides with a right non-key column.
    //        leftNonKey intersect rightNonKey misses this case, causing
    //        duplicate column names in the result.
    // ===================================================================
    [Fact]
    public void Join_LeftKeyName_CollidesWithRightNonKeyName_ShouldAddSuffix()
    {
        // Use nullable int key to force the generic join path (not the typed fast path).
        // Left has key column "id" and value column "value".
        // Right has key column "rid" and non-key column "id".
        // After left join, both "id" columns should appear — one suffixed.
        var left = new DataFrame(
            Column<int>.FromNullable("id", [1, 2, 3]),
            new Column<double>("value", [10.0, 20.0, 30.0])
        );
        var right = new DataFrame(
            Column<int>.FromNullable("rid", [1, 2, 3]),
            new StringColumn("id", ["x", "y", "z"])  // non-key column named "id"
        );

        // Force generic path by using Left join (typed fast path only handles Inner)
        var result = left.Join(right, "id", "rid", how: JoinType.Left);

        // Should not throw "Duplicate column name" and should have suffixed columns
        result.ColumnNames.Should().OnlyHaveUniqueItems(
            "column names should be unique after join");
    }

    // ===================================================================
    // Bug 4: Head/Tail with negative count produces cryptic error.
    //        Should throw a clear ArgumentException.
    // ===================================================================
    [Fact]
    public void Head_NegativeCount_ShouldThrowArgumentException()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var act = () => df.Head(-1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Tail_NegativeCount_ShouldThrowArgumentException()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var act = () => df.Tail(-1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ===================================================================
    // Bug 5: String inner join conflates null and empty string keys.
    //        TypedStringInnerJoin uses `?? ""` for both left and right keys,
    //        meaning null keys will match empty string keys.
    // ===================================================================
    [Fact]
    public void StringJoin_NullKey_ShouldNotMatchEmptyStringKey()
    {
        var left = new DataFrame(
            new StringColumn("key", [null, "A", ""]),
            new Column<int>("lval", [1, 2, 3])
        );
        var right = new DataFrame(
            new StringColumn("key", ["", "A", null]),
            new Column<int>("rval", [10, 20, 30])
        );

        var result = left.Join(right, "key");

        // "A" matches "A" -> 1 row
        // "" matches "" -> 1 row
        // null matches null -> 1 row (debatable, but should NOT match "")
        // The bug is: null also matches "" giving extra rows
        // With the fix, null-to-null matching is acceptable
        // The key point: we should get exactly 3 rows, not more
        result.RowCount.Should().BeInRange(1, 3,
            "null keys should not match empty string keys");
    }

    // ===================================================================
    // Bug 6: Sort on double column with NaN - NaN goes to front in ascending
    //        because the typed fast path uses CompareTo where NaN < everything.
    //        Standard behavior (pandas, SQL, etc.) puts NaN at the end.
    // ===================================================================
    [Fact]
    public void Sort_DoubleColumn_WithNaN_NaNShouldSortLast_Ascending()
    {
        var df = new DataFrame(
            new Column<double>("val", [3.0, double.NaN, 1.0, 2.0])
        );

        var sorted = df.Sort("val", ascending: true);
        var vals = sorted.GetColumn<double>("val");

        // NaN should be at the end for ascending sort
        double.IsNaN(vals[3]!.Value).Should().BeTrue(
            "NaN should sort to the end in ascending order");
        vals[0].Should().Be(1.0);
        vals[1].Should().Be(2.0);
        vals[2].Should().Be(3.0);
    }

    [Fact]
    public void Sort_DoubleColumn_WithNaN_NaNShouldSortLast_Descending()
    {
        var df = new DataFrame(
            new Column<double>("val", [3.0, double.NaN, 1.0, 2.0])
        );

        var sorted = df.Sort("val", ascending: false);
        var vals = sorted.GetColumn<double>("val");

        // NaN should be at the end for descending sort too
        double.IsNaN(vals[3]!.Value).Should().BeTrue(
            "NaN should sort to the end in descending order");
        vals[0].Should().Be(3.0);
        vals[1].Should().Be(2.0);
        vals[2].Should().Be(1.0);
    }

    // ===================================================================
    // Non-bug edge cases that should work correctly (sanity checks)
    // ===================================================================

    [Fact]
    public void Filter_AllFalseMask_ReturnsEmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new StringColumn("B", ["x", "y", "z"])
        );

        var result = df.Filter(new bool[] { false, false, false });

        result.RowCount.Should().Be(0);
        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Equal("A", "B");
    }

    [Fact]
    public void Head_Zero_ReturnsEmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var result = df.Head(0);

        result.RowCount.Should().Be(0);
        result.ColumnCount.Should().Be(1);
    }

    [Fact]
    public void Tail_Zero_ReturnsEmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var result = df.Tail(0);

        result.RowCount.Should().Be(0);
        result.ColumnCount.Should().Be(1);
    }

    [Fact]
    public void Head_GreaterThanRowCount_ReturnsAllRows()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var result = df.Head(100);

        result.RowCount.Should().Be(3);
    }

    [Fact]
    public void Batch_GreaterThanRowCount_ReturnsSingleBatch()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var batches = df.Batch(100).ToList();

        batches.Should().HaveCount(1);
        batches[0].RowCount.Should().Be(3);
    }

    [Fact]
    public void Batch_Zero_ShouldThrow()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var act = () => df.Batch(0).ToList();

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Sort_EmptyDataFrame_DoesNotThrow()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>())
        );

        var result = df.Sort("A");

        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Concat_WithEmptyDataFrame_WorksCorrectly()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2])
        );
        var df2 = new DataFrame(
            new Column<int>("A", Array.Empty<int>())
        );

        var result = DataFrame.Concat(df1, df2);

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void GroupBy_ZeroRowDataFrame_ReturnsZeroGroups()
    {
        var df = new DataFrame(
            new StringColumn("Key", Array.Empty<string?>()),
            new Column<int>("Val", Array.Empty<int>())
        );

        var grouped = df.GroupBy("Key");

        grouped.GroupCount.Should().Be(0);
    }

    [Fact]
    public void RenameColumn_NonExistentColumn_ShouldThrow()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2])
        );

        var act = () => df.RenameColumn("Z", "B");

        act.Should().Throw<KeyNotFoundException>();
    }

    [Fact]
    public void ColumnIntArithmetic_AddIntColumns_Works()
    {
        var a = new Column<int>("a", [1, 2, 3]);
        var b = new Column<int>("b", [4, 5, 6]);

        var result = a + b;

        result[0].Should().Be(5);
        result[1].Should().Be(7);
        result[2].Should().Be(9);
    }

    [Fact]
    public void ColumnDivision_ByZero_ReturnsNull()
    {
        var a = new Column<int>("a", [10, 20, 30]);
        var b = new Column<int>("b", [2, 0, 5]);

        var result = a / b;

        result[0].Should().Be(5);
        result[1].Should().BeNull("division by zero should produce null");
        result[2].Should().Be(6);
    }

    [Fact]
    public void FromMatrix_ZeroByZero_ReturnsEmptyDataFrame()
    {
        var matrix = new double[0, 0];
        var result = DataFrame.FromMatrix(matrix);

        result.RowCount.Should().Be(0);
        result.ColumnCount.Should().Be(0);
    }

    [Fact]
    public void FromMatrix_ColumnNamesMismatch_ShouldThrow()
    {
        var matrix = new double[2, 3] { { 1, 2, 3 }, { 4, 5, 6 } };

        var act = () => DataFrame.FromMatrix(matrix, ["A", "B"]);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Filter_MaskLengthMismatch_ShouldThrow()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3])
        );

        var act = () => df.Filter(new bool[] { true, false });

        act.Should().Throw<ArgumentException>()
            .WithMessage("*mask*");
    }

    [Fact]
    public void ColumnArithmetic_DifferentLengths_ShouldThrow()
    {
        var a = new Column<int>("a", [1, 2, 3]);
        var b = new Column<int>("b", [4, 5]);

        var act = () => { var _ = a + b; };

        // After fix: ArgumentException propagates directly
        // (no longer wrapped in TargetInvocationException)
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Concat_MismatchedColumns_FillsNulls()
    {
        var df1 = new DataFrame(
            new Column<int>("A", [1, 2]),
            new Column<int>("B", [3, 4])
        );
        var df2 = new DataFrame(
            new Column<int>("A", [5]),
            new Column<int>("C", [6])
        );

        var result = DataFrame.Concat(df1, df2);

        result.RowCount.Should().Be(3);
        result.ColumnNames.Should().Contain("A");
        result.ColumnNames.Should().Contain("B");
        result.ColumnNames.Should().Contain("C");

        // B should be null for row from df2
        result["B"].GetObject(2).Should().BeNull();
        // C should be null for rows from df1
        result["C"].GetObject(0).Should().BeNull();
    }
}
