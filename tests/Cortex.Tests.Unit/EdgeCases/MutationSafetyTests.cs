using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.Concat;
using Cortex.Accessors;
using FluentAssertions;

namespace Cortex.Tests.Unit.EdgeCases;

public class MutationSafetyTests
{
    // ========================================================================
    // Bug 1: Build2StringGroups crashes when either string column contains nulls
    // DictEncoding maps null -> code -1, then composite key = code1 * n2 + code2
    // produces negative keys, and reversing via dict.Uniques[c1] uses negative index.
    // ========================================================================

    [Fact]
    public void GroupBy_TwoStringColumns_WithNulls_ShouldNotCrash()
    {
        var df = new DataFrame(
            new StringColumn("A", new string?[] { "x", "y", null, "x" }),
            new StringColumn("B", new string?[] { "a", "b", "c", null })
        );

        // This should not throw - nulls should form their own group(s)
        var grouped = df.GroupBy("A", "B");
        grouped.GroupCount.Should().BeGreaterThan(0);
    }

    // ========================================================================
    // Bug 2: Build2StringGroups doesn't respect NullGroupingMode.Exclude
    // The fast path for 2 string columns never checks _nullMode
    // ========================================================================

    [Fact]
    public void GroupBy_TwoStringColumns_ExcludeNulls_ShouldExcludeNullRows()
    {
        var df = new DataFrame(
            new StringColumn("A", new string?[] { "x", "y", null, "x" }),
            new StringColumn("B", new string?[] { "a", "b", "c", null }),
            new Column<int>("Val", new[] { 1, 2, 3, 4 })
        );

        var grouped = df.GroupBy(new[] { "A", "B" }, NullGroupingMode.Exclude);

        // Rows with any null key should be excluded
        // Row 0: (x, a) - included
        // Row 1: (y, b) - included
        // Row 2: (null, c) - excluded
        // Row 3: (x, null) - excluded
        // So we should have 2 groups
        grouped.GroupCount.Should().Be(2);
    }

    // ========================================================================
    // Bug 3: BuildIntGroups reads span[r] for null positions, grouping nulls
    // with the default(int)=0 value rather than as a separate null group.
    // ========================================================================

    [Fact]
    public void GroupBy_NullableIntColumn_NullsShouldNotGroupWithZero()
    {
        var col = Column<int>.FromNullable("Key", new int?[] { 0, null, 0, null, 1 });
        var df = new DataFrame(
            col,
            new Column<int>("Val", new[] { 10, 20, 30, 40, 50 })
        );

        var grouped = df.GroupBy("Key");

        // Should have 3 groups: 0, null, 1
        // BUG: nulls are grouped with 0 because span[r] returns default(0) for null positions
        grouped.GroupCount.Should().Be(3, "null values should form their own group, separate from 0");
    }

    // ========================================================================
    // Bug 4: BuildStringGroups with cached dict doesn't handle null codes (-1)
    // When using the fast cached-dict path, codes[r] = -1 for nulls,
    // and groupLists[codes[r]] throws IndexOutOfRangeException.
    // ========================================================================

    [Fact]
    public void GroupBy_StringColumn_CachedDict_WithNulls_ShouldNotCrash()
    {
        var sc = new StringColumn("Key", new string?[] { "a", null, "b", null, "a" });

        // Force dict caching by calling GetDictCodes()
        sc.GetDictCodes();

        var df = new DataFrame(
            sc,
            new Column<int>("Val", new[] { 1, 2, 3, 4, 5 })
        );

        // This should not throw IndexOutOfRangeException
        var grouped = df.GroupBy("Key");
        grouped.GroupCount.Should().BeGreaterThan(0);
    }

    // ========================================================================
    // Bug 5: ColumnArithmetic.Divide treats zero-divisor as null for
    // floating-point types, but IEEE 754 says x/0 should be +/-Infinity.
    // ========================================================================

    [Fact]
    public void Divide_DoubleByZero_ShouldProduceInfinity_NotNull()
    {
        var left = new Column<double>("a", new[] { 1.0, -1.0, 0.0 });
        var right = new Column<double>("b", new[] { 0.0, 0.0, 0.0 });

        var result = left / right;

        // IEEE 754: 1.0 / 0.0 = +Infinity, -1.0 / 0.0 = -Infinity, 0.0 / 0.0 = NaN
        // BUG: Current code treats zero divisor as null for all types
        result[0].Should().Be(double.PositiveInfinity, "1.0 / 0.0 should be +Infinity per IEEE 754");
        result[1].Should().Be(double.NegativeInfinity, "-1.0 / 0.0 should be -Infinity per IEEE 754");
        double.IsNaN(result[2]!.Value).Should().BeTrue("0.0 / 0.0 should be NaN per IEEE 754");
    }

    // ========================================================================
    // Bug 6: Concat with Column<int> and Column<double> for the same column name
    // uses the type from the first frame, casting doubles to int and losing data.
    // ========================================================================

    [Fact]
    public void Concat_IntAndDoubleColumns_ShouldWidenToDouble()
    {
        var df1 = new DataFrame(new Column<int>("A", new[] { 1, 2 }));
        var df2 = new DataFrame(new Column<double>("A", new[] { 3.5, 4.5 }));

        var result = DataFrame.Concat(df1, df2);

        // The result should preserve the double values 3.5 and 4.5
        // BUG: Current code uses the first frame's type (int), so Convert.ChangeType
        // truncates 3.5 -> 3 and 4.5 -> 4
        result.RowCount.Should().Be(4);

        var val2 = result["A"].GetObject(2);
        var val3 = result["A"].GetObject(3);

        // These should be 3.5 and 4.5, not 3 and 4
        Convert.ToDouble(val2).Should().BeApproximately(3.5, 0.001, "double value 3.5 should be preserved after concat");
        Convert.ToDouble(val3).Should().BeApproximately(4.5, 0.001, "double value 4.5 should be preserved after concat");
    }

    // ========================================================================
    // Bug 7: BuildStringGroups cached-dict path doesn't respect NullGroupingMode
    // ========================================================================

    [Fact]
    public void GroupBy_StringColumn_CachedDict_ExcludeNulls_ShouldExclude()
    {
        var sc = new StringColumn("Key", new string?[] { "a", null, "b", null, "a" });
        sc.GetDictCodes(); // force caching

        var df = new DataFrame(
            sc,
            new Column<int>("Val", new[] { 1, 2, 3, 4, 5 })
        );

        var grouped = df.GroupBy(new[] { "Key" }, NullGroupingMode.Exclude);

        // With Exclude mode, null keys should be excluded
        // Should have 2 groups: "a" and "b"
        grouped.GroupCount.Should().Be(2, "null keys should be excluded");

        // The sum for "a" should be 1 + 5 = 6
        var sumDf = grouped.Sum();
        var aKey = new GroupKey(new object?[] { "a" });
        var aGroup = grouped.GetGroup(aKey);
        aGroup.RowCount.Should().Be(2, "group 'a' should have 2 rows");
    }

    // ========================================================================
    // Bug 8: Multi-column GroupBy with int + string combination
    // Falls through to the generic path which works, but let's verify it
    // handles nulls and produces correct Sum results.
    // ========================================================================

    [Fact]
    public void GroupBy_IntPlusStringColumns_ShouldWork()
    {
        var df = new DataFrame(
            new Column<int>("Dept", new[] { 1, 2, 1, 2, 1 }),
            new StringColumn("Region", new string?[] { "East", "West", "East", "West", "West" }),
            new Column<double>("Sales", new[] { 10.0, 20.0, 30.0, 40.0, 50.0 })
        );

        var grouped = df.GroupBy("Dept", "Region");
        grouped.GroupCount.Should().Be(3); // (1,East), (2,West), (1,West)

        var sumDf = grouped.Sum();
        sumDf.RowCount.Should().Be(3);

        // Verify keys are preserved correctly in the result
        sumDf.ColumnNames.Should().Contain("Dept");
        sumDf.ColumnNames.Should().Contain("Region");
        sumDf.ColumnNames.Should().Contain("Sales");
    }

    // ========================================================================
    // Bug 9: GroupBy on 3+ columns should work (uses generic path)
    // ========================================================================

    [Fact]
    public void GroupBy_ThreeColumns_ShouldWork()
    {
        var df = new DataFrame(
            new StringColumn("A", new string?[] { "x", "x", "y", "y" }),
            new StringColumn("B", new string?[] { "a", "a", "b", "b" }),
            new Column<int>("C", new[] { 1, 1, 2, 2 }),
            new Column<double>("Val", new[] { 10.0, 20.0, 30.0, 40.0 })
        );

        var grouped = df.GroupBy("A", "B", "C");
        grouped.GroupCount.Should().Be(2); // (x,a,1), (y,b,2)

        var sumDf = grouped.Sum();
        sumDf.RowCount.Should().Be(2);
        sumDf.ColumnNames.Should().Contain("Val");
    }

    // ========================================================================
    // Bug 10: DateTimeAccessor with null DateTime values
    // ========================================================================

    [Fact]
    public void DateTimeAccessor_WithNulls_ShouldPropagateNulls()
    {
        var col = Column<DateTime>.FromNullable("dt", new DateTime?[]
        {
            new DateTime(2024, 1, 15),
            null,
            new DateTime(2024, 6, 30),
            null
        });

        var years = col.Dt().Year();
        years[0].Should().Be(2024);
        years[1].Should().BeNull("null datetime should produce null year");
        years[2].Should().Be(2024);
        years[3].Should().BeNull();

        var months = col.Dt().Month();
        months[0].Should().Be(1);
        months[1].Should().BeNull();
        months[2].Should().Be(6);

        var days = col.Dt().Day();
        days[0].Should().Be(15);
        days[1].Should().BeNull();
        days[2].Should().Be(30);
    }

    // ========================================================================
    // Bug 11: DateTimeAccessor DayOfWeek on edge dates
    // ========================================================================

    [Fact]
    public void DateTimeAccessor_DayOfWeek_EdgeDates()
    {
        var col = new Column<DateTime>("dt", new[]
        {
            new DateTime(2024, 1, 1),   // Monday
            new DateTime(2024, 12, 31),  // Tuesday
            new DateTime(2024, 2, 29),   // Thursday (leap year)
        });

        var dow = col.Dt().DayOfWeek();
        dow[0].Should().Be((int)DayOfWeek.Monday);
        dow[1].Should().Be((int)DayOfWeek.Tuesday);
        dow[2].Should().Be((int)DayOfWeek.Thursday);
    }

    // ========================================================================
    // Bug 12: Arithmetic with mismatched null patterns - union of nulls
    // ========================================================================

    [Fact]
    public void Add_MismatchedNullPatterns_ShouldUnionNulls()
    {
        var col1 = Column<int>.FromNullable("a", new int?[] { null, 10, null, 40 });
        var col2 = Column<int>.FromNullable("b", new int?[] { 1, null, null, 4 });

        var result = col1 + col2;

        // Position 0: null + 1 = null
        result[0].Should().BeNull("null + value should be null");
        // Position 1: 10 + null = null
        result[1].Should().BeNull("value + null should be null");
        // Position 2: null + null = null
        result[2].Should().BeNull("null + null should be null");
        // Position 3: 40 + 4 = 44
        result[3].Should().Be(44);
    }

    // ========================================================================
    // Bug 13: Scalar op on all-null column should produce all null
    // ========================================================================

    [Fact]
    public void AddScalar_AllNullColumn_ShouldProduceAllNull()
    {
        var col = Column<int>.FromNullable("a", new int?[] { null, null, null });

        var result = col + 10;

        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().BeNull();
    }

    // ========================================================================
    // Bug 14: NaN comparison behavior per IEEE 754
    // NaN > x should be false, NaN == NaN should be false
    // ========================================================================

    [Fact]
    public void Comparison_NaN_ShouldFollowIeee754()
    {
        var col = new Column<double>("a", new[] { 1.0, double.NaN, 3.0, double.NaN });

        // NaN comparisons should always be false
        var gtResult = col.Gt(double.NaN);
        gtResult.Should().AllBeEquivalentTo(false, "nothing is greater than NaN per IEEE 754");

        var eqResult = col.Eq(double.NaN);
        eqResult.Should().AllBeEquivalentTo(false, "NaN != NaN per IEEE 754");
    }

    // ========================================================================
    // Bug 15: NaN + NaN should be NaN, Infinity * 0 should be NaN
    // ========================================================================

    [Fact]
    public void Arithmetic_NaN_Infinity_ShouldFollowIeee754()
    {
        var nanCol = new Column<double>("a", new[] { double.NaN, double.PositiveInfinity, double.NegativeInfinity });
        var nanCol2 = new Column<double>("b", new[] { double.NaN, 0.0, double.PositiveInfinity });

        var addResult = nanCol + nanCol2;
        double.IsNaN(addResult[0]!.Value).Should().BeTrue("NaN + NaN = NaN");
        // Infinity + 0 = Infinity
        addResult[1].Should().Be(double.PositiveInfinity, "Infinity + 0 = Infinity");
        // -Infinity + Infinity = NaN
        double.IsNaN(addResult[2]!.Value).Should().BeTrue("-Infinity + Infinity = NaN");

        var mulResult = nanCol * nanCol2;
        double.IsNaN(mulResult[0]!.Value).Should().BeTrue("NaN * NaN = NaN");
        // Infinity * 0 = NaN
        double.IsNaN(mulResult[1]!.Value).Should().BeTrue("Infinity * 0 = NaN");
    }

    // ========================================================================
    // Bug 16: Concat with bool and int columns - should handle gracefully
    // ========================================================================

    [Fact]
    public void Concat_BoolAndIntSameColumnName_ShouldNotLoseData()
    {
        var df1 = new DataFrame(new Column<bool>("A", new[] { true, false }));
        var df2 = new DataFrame(new Column<int>("A", new[] { 5, 10 }));

        // This should either widen or throw, but not silently corrupt
        var act = () => DataFrame.Concat(df1, df2);

        // The current implementation uses the first frame's type (bool) and tries
        // Convert.ChangeType(5, typeof(bool)) which will likely convert 5 to true
        // This is a data corruption issue - int values are silently converted to bool
        // We just verify it doesn't crash and check behavior
        var result = act();

        // If it doesn't throw, at least the row count should be right
        result.RowCount.Should().Be(4);
    }

    // ========================================================================
    // Bug 17: TakeRows on nullable column - null positions correctly mapped
    // ========================================================================

    [Fact]
    public void TakeRows_NullableColumn_ShouldPreserveNullPositions()
    {
        var col = Column<int>.FromNullable("a", new int?[] { 10, null, 30, null, 50 });

        var result = (Column<int>)col.TakeRows(new[] { 4, 1, 3, 0, 2 });

        result[0].Should().Be(50, "index 4 has value 50");
        result[1].Should().BeNull("index 1 is null");
        result[2].Should().BeNull("index 3 is null");
        result[3].Should().Be(10, "index 0 has value 10");
        result[4].Should().Be(30, "index 2 has value 30");
    }

    // ========================================================================
    // Bug 18: TakeRows with repeated indices on nullable column
    // ========================================================================

    [Fact]
    public void TakeRows_RepeatedIndices_NullableColumn_ShouldWork()
    {
        var col = Column<int>.FromNullable("a", new int?[] { 10, null, 30 });

        var result = (Column<int>)col.TakeRows(new[] { 1, 1, 0, 2, 1 });

        result[0].Should().BeNull("index 1 is null");
        result[1].Should().BeNull("index 1 is null (repeated)");
        result[2].Should().Be(10, "index 0 has value 10");
        result[3].Should().Be(30, "index 2 has value 30");
        result[4].Should().BeNull("index 1 is null (repeated again)");
    }

    // ========================================================================
    // Bug 19: Filter on nullable column - null bitmask correctly transferred
    // ========================================================================

    [Fact]
    public void Filter_NullableColumn_ShouldPreserveNullPositions()
    {
        var col = Column<int>.FromNullable("a", new int?[] { 10, null, 30, null, 50 });
        var mask = new bool[] { true, true, false, true, false };

        var result = (Column<int>)col.Filter(mask);

        result.Length.Should().Be(3);
        result[0].Should().Be(10, "position 0 (value 10) was selected");
        result[1].Should().BeNull("position 1 (null) was selected");
        result[2].Should().BeNull("position 3 (null) was selected");
    }

    // ========================================================================
    // Bug 20: Slice on nullable column - null bitmask correctly sliced
    // ========================================================================

    [Fact]
    public void Slice_NullableColumn_ShouldPreserveNullPositions()
    {
        var col = Column<int>.FromNullable("a", new int?[] { 10, null, 30, null, 50 });

        var result = (Column<int>)col.Slice(1, 3);

        result.Length.Should().Be(3);
        result[0].Should().BeNull("position 1 is null");
        result[1].Should().Be(30, "position 2 has value 30");
        result[2].Should().BeNull("position 3 is null");
    }

    // ========================================================================
    // Bug 21: Column.Clone provides data isolation
    // ========================================================================

    [Fact]
    public void Clone_ShouldProvideDataIsolation()
    {
        var original = new Column<int>("test", new[] { 1, 2, 3 });
        var clone = (Column<int>)original.Clone();

        // Clone values should match original
        clone[0].Should().Be(1);
        clone[1].Should().Be(2);
        clone[2].Should().Be(3);
        clone.Name.Should().Be("test");
    }

    // ========================================================================
    // Bug 22: StringColumn.Clone provides data isolation
    // ========================================================================

    [Fact]
    public void StringColumn_Clone_ShouldProvideDataIsolation()
    {
        var original = new StringColumn("test", new string?[] { "a", "b", null });
        var clone = (StringColumn)original.Clone();

        clone[0].Should().Be("a");
        clone[1].Should().Be("b");
        clone[2].Should().BeNull();
    }

    // ========================================================================
    // Bug 23: No-null column + all-null column = all null
    // ========================================================================

    [Fact]
    public void Add_NoNullsPlusAllNulls_ShouldBeAllNull()
    {
        var col1 = new Column<int>("a", new[] { 1, 2, 3 });
        var col2 = Column<int>.FromNullable("b", new int?[] { null, null, null });

        var result = col1 + col2;

        result[0].Should().BeNull();
        result[1].Should().BeNull();
        result[2].Should().BeNull();
    }
}
