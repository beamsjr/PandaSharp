using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Expressions;
using PandaSharp.GroupBy;
using PandaSharp.Joins;
using PandaSharp.Window;
using static PandaSharp.Expressions.Expr;

namespace PandaSharp.Tests.Unit;

/// <summary>
/// Tests filling coverage gaps identified in code review.
/// </summary>
public class CoverageGapTests
{
    // -- Join with null keys --

    [Fact]
    public void Join_NullInJoinKey_HandledGracefully()
    {
        var left = new DataFrame(
            Column<int>.FromNullable("Id", [1, null, 3]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );
        var right = new DataFrame(
            Column<int>.FromNullable("Id", [1, 3]),
            new StringColumn("Role", ["Dev", "QA"])
        );

        // Inner join should only match non-null keys
        var result = left.Join(right, "Id");
        result.RowCount.Should().Be(2); // 1 and 3 match, null doesn't
    }

    [Fact]
    public void Join_LeftJoin_NullKey_Preserved()
    {
        var left = new DataFrame(
            Column<int>.FromNullable("Id", [1, null, 3]),
            new StringColumn("Name", ["Alice", "Bob", "Charlie"])
        );
        var right = new DataFrame(
            new Column<int>("Id", [1, 3]),
            new StringColumn("Role", ["Dev", "QA"])
        );

        var result = left.Join(right, "Id", how: JoinType.Left);
        result.RowCount.Should().Be(3); // all left rows kept
        result.GetStringColumn("Role")[1].Should().BeNull(); // null key → no match
    }

    // -- Window with nulls --

    [Fact]
    public void RollingMean_AllNulls_ReturnsNulls()
    {
        var col = Column<double>.FromNullable("X", [null, null, null, null]);
        var result = col.Rolling(3, minPeriods: 1).Mean();
        // With all nulls, no valid windows
        for (int i = 0; i < 4; i++)
            result[i].Should().BeNull();
    }

    [Fact]
    public void Rolling_WindowLargerThanArray()
    {
        var col = new Column<double>("X", [1.0, 2.0]);
        var result = col.Rolling(5, minPeriods: 1).Mean();
        result[0].Should().Be(1.0); // only 1 value in window
        result[1].Should().Be(1.5); // 2 values
    }

    [Fact]
    public void Expanding_SingleElement()
    {
        var col = new Column<double>("X", [42.0]);
        col.Expanding().Sum()[0].Should().Be(42.0);
        col.Expanding().Mean()[0].Should().Be(42.0);
    }

    // -- Expression type coercion --

    [Fact]
    public void Expr_IntPlusDouble_Works()
    {
        var df = new DataFrame(
            new Column<int>("A", [1, 2, 3]),
            new Column<double>("B", [0.5, 1.5, 2.5])
        );

        var result = (Col("A") + Col("B")).Evaluate(df);
        result.GetObject(0).Should().Be(1.5);
        result.GetObject(2).Should().Be(5.5);
    }

    [Fact]
    public void Expr_CompareIntToDouble_Works()
    {
        var df = new DataFrame(
            new Column<int>("Age", [20, 30, 40])
        );

        var result = df.Filter(Col("Age") > Lit(25.0));
        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void Expr_StringEq_Works()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob", "Alice"])
        );

        var result = df.Filter(Col("Name").Eq(Lit("Alice")));
        result.RowCount.Should().Be(2);
    }

    // -- GroupBy Sum type consistency --

    [Fact]
    public void GroupBySum_IntColumn_ReturnsDouble()
    {
        var df = new DataFrame(
            new StringColumn("Key", ["A", "A", "B"]),
            new Column<int>("Val", [10, 20, 30])
        );

        var result = GroupByExtensions.GroupBy(df, "Key").Sum();
        // Document: int sum returns double (typed fast path)
        result["Val"].DataType.Should().Be(typeof(double));
    }

    [Fact]
    public void GroupBySum_DoubleColumn_ReturnsDouble()
    {
        var df = new DataFrame(
            new StringColumn("Key", ["A", "A"]),
            new Column<double>("Val", [1.5, 2.5])
        );

        var result = GroupByExtensions.GroupBy(df, "Key").Sum();
        result["Val"].DataType.Should().Be(typeof(double));
        result.GetColumn<double>("Val")[0].Should().Be(4.0);
    }

    // -- CSV encoding edge cases --

    [Fact]
    public void CsvRead_NullPath_Throws()
    {
        var act = () => PandaSharp.IO.CsvReader.Read((string)null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void CsvRoundTrip_WithSpecialChars()
    {
        var df = new DataFrame(
            new StringColumn("Text", ["hello, world", "tabs\there", "quote\"mark"])
        );

        using var ms = new MemoryStream();
        PandaSharp.IO.CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var loaded = PandaSharp.IO.CsvReader.Read(ms);

        loaded.GetStringColumn("Text")[0].Should().Be("hello, world");
    }

    // -- ArrowBackedBuffer bounds check --

    [Fact]
    public void TakeRows_OutOfBounds_ThrowsWithContext()
    {
        var col = new Column<int>("X", [1, 2, 3]);
        var act = () => col.TakeRows(new int[] { 0, 99 });
        act.Should().Throw<ArgumentOutOfRangeException>()
            .WithMessage("*99*");
    }

}
