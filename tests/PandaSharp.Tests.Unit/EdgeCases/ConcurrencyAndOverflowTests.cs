using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Missing;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.EdgeCases;

/// <summary>
/// Tests for concurrency, numeric overflow, sort stability, and interpolation edge cases.
/// </summary>
public class ConcurrencyAndOverflowTests
{
    // =========================================================================
    // Bug 1: Sort stability — equal values should preserve original order
    // The SpanComparer<T> doesn't break ties by index, and Array.Sort is unstable.
    // =========================================================================

    [Fact]
    public void Sort_ShouldBeStable_PreservesOriginalOrderForEqualValues()
    {
        // Use enough elements with equal keys that IntroSort is likely to reorder.
        // The sort column has groups of equal values; the order column reveals original position.
        // We interleave groups: [2,1,2,1,2,1,...] so that a stable sort of "group" ascending
        // must produce all 1s first (in original order) then all 2s (in original order).
        int n = 64;
        var groups = new int[n];
        var orders = new int[n];
        for (int i = 0; i < n; i++)
        {
            groups[i] = i % 2 == 0 ? 2 : 1; // alternating 2, 1, 2, 1, ...
            orders[i] = i;
        }

        var df = DataFrame.FromDictionary(new()
        {
            ["group"] = groups,
            ["order"] = orders
        });

        var sorted = df.Sort("group");
        var orderCol = sorted.GetColumn<int>("order");

        // First n/2 rows should be group=1, with original indices 1, 3, 5, 7, ...
        // in that exact order (stability = preserve original relative order)
        for (int i = 0; i < n / 2; i++)
        {
            int expectedOriginalIdx = 2 * i + 1; // 1, 3, 5, 7, ...
            orderCol[i].Should().Be(expectedOriginalIdx,
                $"stable sort should preserve original order for equal keys (row {i})");
        }

        // Last n/2 rows should be group=2, with original indices 0, 2, 4, 6, ...
        for (int i = 0; i < n / 2; i++)
        {
            int expectedOriginalIdx = 2 * i; // 0, 2, 4, 6, ...
            orderCol[n / 2 + i].Should().Be(expectedOriginalIdx,
                $"stable sort should preserve original order for equal keys (row {n / 2 + i})");
        }
    }

    [Fact]
    public void Sort_ShouldBeStable_DoubleColumn()
    {
        // Use enough duplicate values to trigger instability with IntroSort
        int n = 64;
        var vals = new double[n];
        var seqs = new int[n];
        for (int i = 0; i < n; i++)
        {
            vals[i] = i < n / 2 ? 1.0 : 2.0;
            seqs[i] = i;
        }
        // Reverse the first half so it's not already sorted by index
        Array.Reverse(seqs, 0, n / 2);
        Array.Reverse(vals, 0, n / 2);

        var df = DataFrame.FromDictionary(new()
        {
            ["val"] = vals,
            ["seq"] = seqs
        });

        var sorted = df.Sort("val");
        var seqCol = sorted.GetColumn<int>("seq");

        // Within the group of val=1.0, the original order should be preserved
        // Original order of val=1.0: indices n/2-1, n/2-2, ..., 1, 0 (reversed)
        for (int i = 1; i < n / 2; i++)
        {
            seqCol[i]!.Value.Should().BeLessThan(seqCol[i - 1]!.Value,
                "stable sort should preserve the original relative order within group val=1.0");
        }
    }

    // =========================================================================
    // Bug 2: Column<int>.Sum() silently overflows
    // Uses int accumulator — should use long for int columns to avoid overflow
    // =========================================================================

    [Fact]
    public void IntColumn_Sum_ShouldNotSilentlyOverflow()
    {
        // int.MaxValue + 1 would overflow a 32-bit int to int.MinValue
        var col = new Column<int>("x", new[] { int.MaxValue, 1 });

        // The mathematically correct sum is 2147483648L which doesn't fit in int.
        // The fix uses checked arithmetic, so this should throw OverflowException
        // instead of silently returning int.MinValue.
        var act = () => col.Sum();
        act.Should().Throw<OverflowException>(
            "summing int.MaxValue + 1 should throw rather than silently overflow");
    }

    // =========================================================================
    // Bug 3: Column<int>.CumSum() silently overflows
    // Same issue — cumulative sum with int accumulator wraps around
    // =========================================================================

    [Fact]
    public void IntColumn_CumSum_ShouldNotSilentlyOverflow()
    {
        var col = new Column<int>("x", new[] { int.MaxValue, 1 });

        // CumSum at index 1 would be int.MaxValue + 1, which overflows int.
        // The fix uses checked arithmetic, so this should throw OverflowException.
        var act = () => col.CumSum();
        act.Should().Throw<OverflowException>(
            "cumsum should throw rather than silently overflow");
    }

    // =========================================================================
    // Bug 4: PctChange where previous value is 0 returns null, should return Infinity
    // Pandas returns inf for 0 -> positive, -inf for 0 -> negative
    // =========================================================================

    [Fact]
    public void PctChange_WhenPreviousIsZero_ShouldReturnInfinity()
    {
        var col = new Column<double>("x", new[] { 0.0, 5.0 });
        var pct = col.PctChange();

        // Pandas: (5 - 0) / 0 = inf, not null
        pct[1].Should().NotBeNull("division by zero in pct_change should produce Infinity, not null");
        double.IsPositiveInfinity(pct[1]!.Value).Should().BeTrue(
            "0 -> 5 should produce positive infinity");
    }

    [Fact]
    public void PctChange_WhenPreviousIsZero_Negative_ShouldReturnNegativeInfinity()
    {
        var col = new Column<double>("x", new[] { 0.0, -3.0 });
        var pct = col.PctChange();

        pct[1].Should().NotBeNull("division by zero in pct_change should produce -Infinity, not null");
        double.IsNegativeInfinity(pct[1]!.Value).Should().BeTrue(
            "0 -> -3 should produce negative infinity");
    }

    [Fact]
    public void PctChange_WhenBothZero_ShouldReturnNaN()
    {
        var col = new Column<double>("x", new[] { 0.0, 0.0 });
        var pct = col.PctChange();

        // 0/0 is NaN in IEEE754, not null
        pct[1].Should().NotBeNull("0/0 in pct_change should produce NaN, not null");
        double.IsNaN(pct[1]!.Value).Should().BeTrue("0 -> 0 should produce NaN");
    }

    // =========================================================================
    // Bug 5: Interpolate on all-null Column<double> silently converts nulls to 0.0
    // The double-specific Interpolate overload returns new Column<double>(name, result)
    // which loses null information
    // =========================================================================

    [Fact]
    public void Interpolate_AllNullDoubleColumn_ShouldPreserveNulls()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var interpolated = col.Interpolate();

        // All-null column cannot be interpolated — nulls should remain null
        interpolated[0].Should().BeNull("all-null column cannot interpolate, should stay null");
        interpolated[1].Should().BeNull("all-null column cannot interpolate, should stay null");
        interpolated[2].Should().BeNull("all-null column cannot interpolate, should stay null");
    }

    [Fact]
    public void Interpolate_NullsAtBothEnds_ShouldExtrapolateCorrectly()
    {
        // [null, null, 10, 20, null, null]
        var col = Column<double>.FromNullable("x", new double?[] { null, null, 10.0, 20.0, null, null });
        var interpolated = col.Interpolate();

        // Leading nulls extrapolate from first known value
        interpolated[0].Should().Be(10.0);
        interpolated[1].Should().Be(10.0);
        // Known values
        interpolated[2].Should().Be(10.0);
        interpolated[3].Should().Be(20.0);
        // Trailing nulls extrapolate from last known value
        interpolated[4].Should().Be(20.0);
        interpolated[5].Should().Be(20.0);
    }

    [Fact]
    public void Interpolate_SingleNonNullValue_ShouldFillWithThatValue()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, 42.0, null, null });
        var interpolated = col.Interpolate();

        // Single known value should be used for all extrapolation
        interpolated[0].Should().Be(42.0);
        interpolated[1].Should().Be(42.0);
        interpolated[2].Should().Be(42.0);
        interpolated[3].Should().Be(42.0);
        interpolated[4].Should().Be(42.0);
    }

    // =========================================================================
    // Additional edge case tests (confirming correct behavior)
    // =========================================================================

    [Fact]
    public void FillNa_OnColumnWithNoNulls_ReturnsIdenticalColumn()
    {
        var col = new Column<double>("x", new[] { 1.0, 2.0, 3.0 });
        var filled = col.FillNa(0.0);

        filled[0].Should().Be(1.0);
        filled[1].Should().Be(2.0);
        filled[2].Should().Be(3.0);
    }

    [Fact]
    public void ForwardFill_WhenFirstValuesAreNull_LeavesThemNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, 5.0, null });
        var filled = col.FillNa(FillStrategy.Forward);

        // No previous value to fill from
        filled[0].Should().BeNull();
        filled[1].Should().BeNull();
        filled[2].Should().Be(5.0);
        filled[3].Should().Be(5.0);
    }

    [Fact]
    public void BackwardFill_WhenLastValuesAreNull_LeavesThemNull()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, 5.0, null, null });
        var filled = col.FillNa(FillStrategy.Backward);

        filled[0].Should().Be(5.0);
        filled[1].Should().Be(5.0);
        // No next value to fill from
        filled[2].Should().BeNull();
        filled[3].Should().BeNull();
    }

    [Fact]
    public void Describe_OnlyStringColumns_ShouldNotThrow()
    {
        var df = new DataFrame(
            new StringColumn("name", new[] { "Alice", "Bob", "Carol" }),
            new StringColumn("city", new[] { "NYC", "LA", "NYC" })
        );

        var result = df.Describe();

        // Should return a DataFrame with just the "stat" column (no numeric columns to describe)
        result.ColumnCount.Should().BeGreaterThanOrEqualTo(1);
        result.RowCount.Should().Be(8); // count, mean, std, min, 25%, 50%, 75%, max
    }

    [Fact]
    public void Describe_SingleRow_ShouldWork()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["value"] = new double[] { 42.0 }
        });

        var result = df.Describe();

        // count=1, mean=42, std=0, min=max=42
        var valueCol = result.GetColumn<double>("value");
        valueCol[0].Should().Be(1.0);  // count
        valueCol[1].Should().Be(42.0); // mean
        valueCol[2].Should().Be(0.0);  // std (with 1 element)
        valueCol[3].Should().Be(42.0); // min
        valueCol[7].Should().Be(42.0); // max
    }

    [Fact]
    public void Describe_AllNullNumericColumn_ShouldReturnNaNStats()
    {
        var col = Column<double>.FromNullable("x", new double?[] { null, null, null });
        var df = new DataFrame(col);

        var result = df.Describe();
        var xCol = result.GetColumn<double>("x");

        xCol[0].Should().Be(0);           // count = 0
        double.IsNaN(xCol[1]!.Value).Should().BeTrue(); // mean = NaN
    }

    [Fact]
    public void ProfileToDataFrame_EmptyDataFrame_ShouldNotThrow()
    {
        var df = new DataFrame(new Column<int>("x", Array.Empty<int>()));
        var profile = df.ProfileToDataFrame();

        profile.RowCount.Should().Be(14); // number of stat rows
        profile.ColumnCount.Should().Be(2); // stat + x
    }

    [Fact]
    public void DataFrame_WithDuplicateColumnNames_ShouldThrow()
    {
        var act = () => new DataFrame(
            new Column<int>("A", new[] { 1, 2 }),
            new Column<int>("A", new[] { 3, 4 })
        );

        act.Should().Throw<ArgumentException>().WithMessage("*Duplicate*");
    }

    [Fact]
    public void DataFrame_WithDifferentLengthColumns_ShouldThrow()
    {
        var act = () => new DataFrame(
            new Column<int>("A", new[] { 1, 2, 3 }),
            new Column<int>("B", new[] { 4, 5 })
        );

        act.Should().Throw<ArgumentException>().WithMessage("*rows*");
    }

    [Fact]
    public void DataFrame_ZeroColumns_HasZeroRowCount()
    {
        var df = new DataFrame();
        df.RowCount.Should().Be(0);
        df.ColumnCount.Should().Be(0);
    }

    [Fact]
    public void FromDictionary_EmptyDictionary_ShouldCreateEmptyDataFrame()
    {
        var df = DataFrame.FromDictionary(new Dictionary<string, Array>());
        df.RowCount.Should().Be(0);
        df.ColumnCount.Should().Be(0);
    }

    [Fact]
    public void GetColumn_WrongType_ShouldThrowInvalidCast()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["x"] = new int[] { 1, 2, 3 }
        });

        // Getting Column<int> as Column<double> should throw
        var act = () => df.GetColumn<double>("x");
        act.Should().Throw<InvalidCastException>();
    }

    [Fact]
    public void GetStringColumn_OnNumericColumn_ShouldThrowInvalidCast()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["x"] = new int[] { 1, 2, 3 }
        });

        var act = () => df.GetStringColumn("x");
        act.Should().Throw<InvalidCastException>();
    }

    [Fact]
    public void DoubleMaxValue_Times2_ProducesInfinity()
    {
        var col = new Column<double>("x", new[] { double.MaxValue });
        var result = col + col;

        double.IsPositiveInfinity(result[0]!.Value).Should().BeTrue(
            "double.MaxValue + double.MaxValue should be +Infinity");
    }

    [Fact]
    public void Rank_WithIntMaxAndMinValues_ShouldWorkCorrectly()
    {
        var col = new Column<int>("x", new[] { int.MaxValue, int.MinValue, 0 });
        var ranked = col.Rank();

        // Ascending: MinValue=1, 0=2, MaxValue=3
        ranked[0].Should().Be(3.0); // MaxValue gets rank 3
        ranked[1].Should().Be(1.0); // MinValue gets rank 1
        ranked[2].Should().Be(2.0); // 0 gets rank 2
    }
}
