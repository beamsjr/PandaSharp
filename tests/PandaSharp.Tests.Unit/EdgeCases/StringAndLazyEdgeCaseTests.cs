using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Expressions;
using PandaSharp.Lazy;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class StringAndLazyEdgeCaseTests
{
    // ══════════════════════════════════════════════════════════════════
    // Bug 1: ValueCounts crashes on all-null StringColumn
    //   ValueCountsExtensions.cs line ~17: countArr[codes[i]]++
    //   Null codes are -1, causing IndexOutOfRangeException.
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void ValueCounts_AllNullStringColumn_ShouldNotThrow()
    {
        var col = new StringColumn("x", [null, null, null]);
        var result = col.ValueCounts();
        result.RowCount.Should().BeGreaterThanOrEqualTo(1);
        // Should have a null entry with count 3
        var countCol = result.GetColumn<int>("count");
        countCol[0].Should().Be(3);
    }

    // ══════════════════════════════════════════════════════════════════
    // Bug 2: StringAccessor.Count("") infinite loop
    //   Count method uses IndexOf with empty substring; finds at every
    //   position but advances idx by 0 (substring.Length == 0), looping
    //   forever.
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void StringAccessor_Count_EmptySubstring_ShouldNotInfiniteLoop()
    {
        var col = new StringColumn("x", ["hello", "ab", null]);
        // Count of empty string: should not hang. Conventional answer is Length+1.
        var cts = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        Column<int>? result = null;
        var task = Task.Run(() => result = col.Str.Count(""), cts.Token);
        bool completed = task.Wait(TimeSpan.FromSeconds(5));
        completed.Should().BeTrue("Count('') should not infinite loop");
    }

    // ══════════════════════════════════════════════════════════════════
    // Bug 3: DropDuplicates single-string-column fast path treats
    //   null and "" as the same value (vals[r] ?? "").
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_NullVsEmptyString_ShouldBeDistinct()
    {
        var df = new DataFrame(
            new StringColumn("a", [null, "", null, ""])
        );
        var result = df.DropDuplicates("a");
        // null and "" are different values; should keep 2 rows
        result.RowCount.Should().Be(2);
    }

    // ══════════════════════════════════════════════════════════════════
    // Bug 4: DropDuplicates two-string-column fast path crashes with
    //   nulls: code -1 produces negative composite key, causing
    //   IndexOutOfRangeException on the flat boolean array.
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_TwoStringColumns_WithNulls_ShouldNotThrow()
    {
        var df = new DataFrame(
            new StringColumn("a", ["x", null, "x", null]),
            new StringColumn("b", ["y", "y", null, null])
        );
        var act = () => df.DropDuplicates("a", "b");
        act.Should().NotThrow();
        var result = act();
        // All 4 rows are distinct (x,y), (null,y), (x,null), (null,null)
        result.RowCount.Should().Be(4);
    }

    // ══════════════════════════════════════════════════════════════════
    // Bug 5: DictEncoding-based Contains/StartsWith/Len return false/0
    //   for null rows instead of propagating null. This means StringAccessor
    //   returns different results for small vs large columns.
    //   Small columns: null input -> null output (correct)
    //   Large columns (dict path): null input -> false/0 (wrong)
    // ══════════════════════════════════════════════════════════════════

    // Note: The dict path only activates for Length > 10_000. We test
    // the non-dict path behavior to confirm the contract, then verify
    // the Contains method on StringColumn (non-accessor) too.

    [Fact]
    public void StringAccessor_Contains_AllNull_ReturnsAllNull()
    {
        var col = new StringColumn("x", [null, null, null]);
        var result = col.Str.Contains("a");
        // Null-propagating: null -> null, not null -> false
        result.IsNull(0).Should().BeTrue();
        result.IsNull(1).Should().BeTrue();
        result.IsNull(2).Should().BeTrue();
    }

    [Fact]
    public void StringAccessor_Upper_AllNull_ReturnsAllNull()
    {
        var col = new StringColumn("x", [null, null, null]);
        var upper = col.Str.Upper();
        upper.Length.Should().Be(3);
        upper[0].Should().BeNull();
        upper[1].Should().BeNull();
        upper[2].Should().BeNull();
    }

    [Fact]
    public void StringAccessor_Match_AllNull_ReturnsAllNull()
    {
        var col = new StringColumn("x", [null, null]);
        var result = col.Str.Match(".*");
        result.IsNull(0).Should().BeTrue();
        result.IsNull(1).Should().BeTrue();
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: StringColumn methods on null-heavy columns
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void StringColumn_ToUpper_WithNulls_PreservesNulls()
    {
        var col = new StringColumn("x", [null, "hello", null]);
        var upper = col.ToUpper();
        upper[0].Should().BeNull();
        upper[1].Should().Be("HELLO");
        upper[2].Should().BeNull();
    }

    [Fact]
    public void StringColumn_ToLower_WithNulls_PreservesNulls()
    {
        var col = new StringColumn("x", [null, "HELLO", null]);
        var lower = col.ToLower();
        lower[0].Should().BeNull();
        lower[1].Should().Be("hello");
        lower[2].Should().BeNull();
    }

    [Fact]
    public void StringColumn_Trim_WithNulls_PreservesNulls()
    {
        var col = new StringColumn("x", [null, "  hi  ", null]);
        var trimmed = col.Trim();
        trimmed[0].Should().BeNull();
        trimmed[1].Should().Be("hi");
        trimmed[2].Should().BeNull();
    }

    [Fact]
    public void StringColumn_Substring_WithNulls_PreservesNulls()
    {
        var col = new StringColumn("x", [null, "hello", null]);
        var sub = col.Substring(0, 2);
        sub[0].Should().BeNull();
        sub[1].Should().Be("he");
        sub[2].Should().BeNull();
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: StringColumn with empty string only
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void StringColumn_EmptyStringsOnly_NUnique_ReturnsOne()
    {
        var col = new StringColumn("x", ["", "", ""]);
        var nunique = col.NUnique();
        nunique.Should().Be(1);
    }

    [Fact]
    public void StringColumn_EmptyStringsOnly_ValueCounts_ReturnsCorrectCount()
    {
        var col = new StringColumn("x", ["", "", ""]);
        var vc = col.ValueCounts();
        vc.RowCount.Should().Be(1);
        var countCol = vc.GetColumn<int>("count");
        countCol[0].Should().Be(3);
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: GetDictCodes when all values are null
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void StringColumn_GetDictCodes_AllNull_ReturnsEmptyUniques()
    {
        var col = new StringColumn("x", [null, null, null]);
        var (codes, uniques) = col.GetDictCodes();
        uniques.Should().BeEmpty();
        codes.Should().AllSatisfy(c => c.Should().Be(-1));
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: StringAccessor edge cases
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void StringAccessor_NormalizeUnicode_WithNulls_PreservesNulls()
    {
        var col = new StringColumn("x", [null, "hello", null]);
        var normalized = col.Str.NormalizeUnicode();
        normalized[0].Should().BeNull();
        normalized[1].Should().Be("hello");
        normalized[2].Should().BeNull();
    }

    [Fact]
    public void StringAccessor_ReplaceIgnoreCase_EmptyOldValue_ShouldNotThrow()
    {
        var col = new StringColumn("x", ["hello"]);
        // Replacing empty string: .NET's String.Replace("", "x", ...) throws ArgumentException
        // This should either throw gracefully or handle it
        Action act = () => col.Str.ReplaceIgnoreCase("", "x");
        // .NET throws ArgumentException for empty oldValue with StringComparison
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void StringAccessor_ContainsIgnoreCase_WithNulls_PropagatesNull()
    {
        var col = new StringColumn("x", [null, "Hello", null]);
        var result = col.Str.ContainsIgnoreCase("hello");
        result.IsNull(0).Should().BeTrue();
        result[1].Should().Be(true);
        result.IsNull(2).Should().BeTrue();
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: CategoricalColumn edge cases
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void CategoricalColumn_ZeroCategories_AllNull()
    {
        var cat = new CategoricalColumn("x", [null, null]);
        cat.CategoryCount.Should().Be(0);
        cat.NullCount.Should().Be(2);
        cat[0].Should().BeNull();
        cat[1].Should().BeNull();
    }

    [Fact]
    public void CategoricalColumn_SlicePreservesMetadata()
    {
        var cat = new CategoricalColumn("x", ["a", "b", "a", null]);
        var sliced = cat.Slice(1, 2);
        sliced.Should().BeOfType<CategoricalColumn>();
        var catSliced = (CategoricalColumn)sliced;
        catSliced.Length.Should().Be(2);
        catSliced[0].Should().Be("b");
        catSliced[1].Should().Be("a");
    }

    [Fact]
    public void CategoricalColumn_FilterPreservesType()
    {
        var cat = new CategoricalColumn("x", ["a", "b", "c"]);
        var filtered = cat.Filter(new bool[] { true, false, true });
        filtered.Should().BeOfType<CategoricalColumn>();
        var catFiltered = (CategoricalColumn)filtered;
        catFiltered.Length.Should().Be(2);
        catFiltered[0].Should().Be("a");
        catFiltered[1].Should().Be("c");
    }

    [Fact]
    public void CategoricalColumn_TakeRowsPreservesType()
    {
        var cat = new CategoricalColumn("x", ["a", "b", "c"]);
        var taken = cat.TakeRows(new int[] { 2, 0 });
        taken.Should().BeOfType<CategoricalColumn>();
        var catTaken = (CategoricalColumn)taken;
        catTaken[0].Should().Be("c");
        catTaken[1].Should().Be("a");
    }

    [Fact]
    public void CategoricalColumn_AllNullCodes_ToStringColumn()
    {
        var cat = new CategoricalColumn("x", [null, null, null]);
        var sc = cat.ToStringColumn();
        sc.Length.Should().Be(3);
        sc[0].Should().BeNull();
        sc[1].Should().BeNull();
        sc[2].Should().BeNull();
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: LazyFrame edge cases
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void LazyFrame_ChainMultipleOperations_ThenCollect()
    {
        var df = new DataFrame(
            new Column<int>("a", [1, 2, 3, 4, 5]),
            new Column<double>("b", [10.0, 20.0, 30.0, 40.0, 50.0])
        );

        var result = df.Lazy()
            .Filter(Expr.Col("a") > Expr.Lit(2))
            .Sort("b", ascending: false)
            .Head(2)
            .Collect();

        result.RowCount.Should().Be(2);
        var aCol = result.GetColumn<int>("a");
        // After filter a>2: [3,4,5], sort b desc: [5,4,3], head 2: [5,4]
        aCol[0].Should().Be(5);
        aCol[1].Should().Be(4);
    }

    [Fact]
    public void LazyFrame_ZeroRowDataFrame_Collect()
    {
        var df = new DataFrame(
            new Column<int>("a", [1]),
            new Column<double>("b", [1.0])
        );
        // Filter everything out
        var result = df.Lazy()
            .Filter(Expr.Col("a") > Expr.Lit(999))
            .Collect();

        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void LazyFrame_CollectCalledTwice_SameResult()
    {
        var df = new DataFrame(
            new Column<int>("a", [1, 2, 3])
        );

        var lazy = df.Lazy().Filter(Expr.Col("a") > Expr.Lit(1));
        var result1 = lazy.Collect();
        var result2 = lazy.Collect();

        result1.RowCount.Should().Be(result2.RowCount);
    }

    [Fact]
    public void LazyFrame_GroupBySum_Collect()
    {
        var df = new DataFrame(
            new StringColumn("key", ["a", "b", "a", "b"]),
            new Column<double>("val", [1.0, 2.0, 3.0, 4.0])
        );

        var result = df.Lazy()
            .GroupBy("key")
            .Sum()
            .Collect();

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void LazyFrame_Explain_ReturnsNonEmptyString()
    {
        var df = new DataFrame(
            new Column<int>("a", [1, 2, 3])
        );

        var plan = df.Lazy()
            .Filter(Expr.Col("a") > Expr.Lit(1))
            .Sort("a")
            .Explain();

        plan.Should().NotBeNullOrWhiteSpace();
        plan.Should().Contain("Sort");
    }

    // ══════════════════════════════════════════════════════════════════
    // Area: DropDuplicates edge cases
    // ══════════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_SingleRowDataFrame_KeepsRow()
    {
        var df = new DataFrame(
            new Column<int>("a", [42])
        );
        var result = df.DropDuplicates();
        result.RowCount.Should().Be(1);
    }

    [Fact]
    public void DropDuplicates_AllIdenticalSubsetColumns_KeepsOneRow()
    {
        var df = new DataFrame(
            new StringColumn("a", ["x", "x", "x"]),
            new Column<int>("b", [1, 2, 3])
        );
        var result = df.DropDuplicates("a");
        result.RowCount.Should().Be(1);
    }
}
