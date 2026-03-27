using FluentAssertions;
using Cortex.Column;
using Cortex.Concat;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Statistics;
using Xunit;

namespace Cortex.Tests.Unit.EdgeCases;

public class RegressionRound7Tests
{
    // ═══════════════════════════════════════════════════════════════
    // Bug 91: GroupBy.Sum typed fast path doesn't skip NaN values.
    //         The Sum fast path for Column<double> only checks
    //         Nulls.IsNull(idx) but not NaN, so NaN values corrupt
    //         the sum with NaN.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void GroupBy_Sum_DoubleColumnWithNaN_ShouldSkipNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "A", "A", "B" },
            ["Val"] = new double[] { 1.0, double.NaN, 3.0, 5.0 }
        });

        var result = df.GroupBy("Key").Sum();

        // Group A: 1.0 + NaN + 3.0 should be 4.0 (skip NaN), not NaN
        var aRow = GetGroupRow(result, "Key", "A");
        var aSum = (double)result["Val"].GetObject(aRow)!;
        double.IsNaN(aSum).Should().BeFalse("NaN should be skipped in GroupBy Sum");
        aSum.Should().Be(4.0);

        // Group B: just 5.0
        var bRow = GetGroupRow(result, "Key", "B");
        var bSum = (double)result["Val"].GetObject(bRow)!;
        bSum.Should().Be(5.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 92: GroupBy.Mean typed fast path doesn't skip NaN values.
    //         Same issue as Sum — only checks IsNull, not NaN.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void GroupBy_Mean_DoubleColumnWithNaN_ShouldSkipNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "A", "A", "B" },
            ["Val"] = new double[] { 2.0, double.NaN, 4.0, 6.0 }
        });

        var result = df.GroupBy("Key").Mean();

        // Group A: mean of 2.0 and 4.0 = 3.0 (skip NaN)
        var aRow = GetGroupRow(result, "Key", "A");
        var aMean = (double)result["Val"].GetObject(aRow)!;
        double.IsNaN(aMean).Should().BeFalse("NaN should be skipped in GroupBy Mean");
        aMean.Should().Be(3.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 93: GroupBy.Std typed fast path doesn't skip NaN values.
    //         The Std fast path for Column<double> also only checks
    //         Nulls.IsNull(idx) and not NaN.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void GroupBy_Std_DoubleColumnWithNaN_ShouldSkipNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "A", "A", "B", "B" },
            ["Val"] = new double[] { 2.0, double.NaN, 4.0, 3.0, 5.0 }
        });

        var result = df.GroupBy("Key").Std();

        var aRow = GetGroupRow(result, "Key", "A");
        var aStd = (double)result["Val"].GetObject(aRow)!;
        double.IsNaN(aStd).Should().BeFalse("NaN should be skipped in GroupBy Std");
        // Std of [2.0, 4.0] with ddof=1 = sqrt(2) ≈ 1.4142
        aStd.Should().BeApproximately(Math.Sqrt(2.0), 0.001);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 94: GroupBy.Min/Max typed fast path doesn't skip NaN.
    //         Min only checks IsNull, so NaN < double.MaxValue
    //         may produce NaN in min comparison.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void GroupBy_MinMax_DoubleColumnWithNaN_ShouldSkipNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "A", "A" },
            ["Val"] = new double[] { 2.0, double.NaN, 4.0 }
        });

        var minResult = df.GroupBy("Key").Min();
        var maxResult = df.GroupBy("Key").Max();

        var minVal = (double)minResult["Val"].GetObject(0)!;
        var maxVal = (double)maxResult["Val"].GetObject(0)!;

        double.IsNaN(minVal).Should().BeFalse("NaN should be skipped in GroupBy Min");
        double.IsNaN(maxVal).Should().BeFalse("NaN should be skipped in GroupBy Max");
        minVal.Should().Be(2.0);
        maxVal.Should().Be(4.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 95: Rank doesn't handle NaN in floating-point columns.
    //         It checks Nulls.IsNull(i) but not NaN, so NaN values
    //         get ranked and their comparison behavior is undefined.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Rank_ColumnWithNaN_ShouldTreatNaNAsMissing()
    {
        var col = Column<double>.FromNullable("x", new double?[]
        {
            3.0, double.NaN, 1.0, null, 2.0
        });

        var ranked = col.Rank();

        // Only 3 valid values: 1.0 (rank 1), 2.0 (rank 2), 3.0 (rank 3)
        // NaN and null should both produce null rank
        ranked[0].Should().Be(3.0); // 3.0 → rank 3
        ranked[1].Should().BeNull("NaN should produce null rank");
        ranked[2].Should().Be(1.0); // 1.0 → rank 1
        ranked[3].Should().BeNull("null should produce null rank");
        ranked[4].Should().Be(2.0); // 2.0 → rank 2
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 96: Describe reports stale "count" for NaN-containing
    //         columns. The count field includes NaN values because
    //         it's computed as (n - NullCount) before NaN filtering.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Describe_ColumnWithNaN_CountShouldExcludeNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 }
        });

        var desc = df.Describe();

        // "count" row should be 3 (only real values), not 5
        var countCol = desc.GetColumn<double>("A");
        var countVal = countCol[0]; // first row is "count"
        countVal.Should().Be(3.0, "count should exclude NaN values");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 97: Profile null% doesn't account for NaN values.
    //         ProfileExtensions.ComputeColumnStats uses col.NullCount
    //         which doesn't include NaN. So null% is understated.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Profile_ColumnWithNaN_NullPercentShouldIncludeNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 }
        });

        var profile = df.ProfileToDataFrame();

        // null% should be 40% (2 NaN out of 5 values), not 0%
        var nullPctStr = (string)profile["A"].GetObject(8)!; // index 8 is "null%"
        var nullPct = double.Parse(nullPctStr);
        nullPct.Should().BeApproximately(40.0, 0.01, "null% should include NaN values");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 98: Profile numeric stats include NaN in computations.
    //         ExtractNonNullDoubles for Column<double> only filters
    //         nulls, not NaN. NaN values corrupt mean, std, etc.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Profile_NumericStats_ShouldExcludeNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 }
        });

        var profile = df.ProfileToDataFrame();

        // mean should be 3.0 (average of 1, 3, 5), not NaN
        var meanStr = (string)profile["A"].GetObject(1)!; // index 1 is "mean"
        var mean = double.Parse(meanStr);
        double.IsNaN(mean).Should().BeFalse("mean should not be NaN when NaN values are present");
        mean.Should().BeApproximately(3.0, 0.01);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 99: Profiler.ProfileNumeric includes NaN in computations.
    //         The Profile() method (not ProfileToDataFrame) also
    //         doesn't filter NaN from numeric values, corrupting
    //         mean, std, min, max, quartiles.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Profiler_NumericStats_ShouldExcludeNaN()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, double.NaN, 3.0, double.NaN, 5.0 }
        });

        var profile = df.Profile();

        var colProfile = profile.Columns[0];
        double.IsNaN(colProfile.Mean).Should().BeFalse("Profile mean should not be NaN");
        colProfile.Mean.Should().BeApproximately(3.0, 0.01);
        colProfile.Min.Should().Be(1.0);
        colProfile.Max.Should().Be(5.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 100: Sort on StringColumn with nulls crashes with
    //          IndexOutOfRangeException. The dict-code fast path
    //          uses rank[codes[i]] but codes[i] = -1 for nulls,
    //          causing rank[-1] to crash.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Sort_StringColumnWithNulls_ShouldNotCrash()
    {
        var df = new DataFrame(
            new StringColumn("Name", new string?[] { "Charlie", null, "Alice", null, "Bob" }),
            new Column<int>("Val", new int[] { 3, 99, 1, 98, 2 })
        );

        // This should not throw IndexOutOfRangeException
        var sorted = df.Sort("Name");

        // Non-null values should be sorted alphabetically, nulls at end
        sorted.RowCount.Should().Be(5);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 101: Multi-column sort on StringColumn with nulls also
    //          crashes from the same dict-code fast path bug.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MultiSort_StringColumnWithNulls_ShouldNotCrash()
    {
        var df = new DataFrame(
            new StringColumn("Name", new string?[] { "Charlie", null, "Alice", null, "Bob" }),
            new Column<int>("Val", new int[] { 3, 99, 1, 98, 2 })
        );

        // This should not throw IndexOutOfRangeException
        var sorted = df.Sort(("Name", true), ("Val", true));
        sorted.RowCount.Should().Be(5);
    }

    // ═══════════════════════════════════════════════════════════════
    // Verification tests: ensure previous fixes still work
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CumSum_WithNullAndNaN_BothProduceNull()
    {
        var col = Column<double>.FromNullable("x", new double?[]
        {
            1.0, double.NaN, null, 3.0
        });

        var cumsum = col.CumSum();

        cumsum[0].Should().Be(1.0);
        cumsum[1].Should().BeNull("NaN should produce null in CumSum");
        cumsum[2].Should().BeNull("null should produce null in CumSum");
        cumsum[3].Should().Be(4.0); // 1.0 + 3.0
    }

    [Fact]
    public void Sum_ColumnWithNullsAndNaN_SkipsBoth()
    {
        var col = Column<double>.FromNullable("x", new double?[]
        {
            1.0, double.NaN, null, 3.0
        });

        var sum = col.Sum();
        sum.Should().Be(4.0); // 1.0 + 3.0, skip NaN and null
    }

    [Fact]
    public void Mean_ColumnWithNaN_SkipsNaN()
    {
        var col = Column<double>.FromNullable("x", new double?[]
        {
            2.0, double.NaN, 4.0
        });

        var mean = col.Mean();
        mean.Should().Be(3.0); // (2.0 + 4.0) / 2
    }

    [Fact]
    public void MinMax_ColumnOnlyNaN_ReturnsNull()
    {
        var col = new Column<double>("x", new double[] { double.NaN, double.NaN });

        var min = col.Min();
        var max = col.Max();

        // All values are NaN (treated as missing) — no valid values
        min.Should().BeNull("Min of all-NaN should be null");
        max.Should().BeNull("Max of all-NaN should be null");
    }

    [Fact]
    public void SortDescending_WithNaN_NaNStillLast()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, double.NaN, 3.0, 2.0 }
        });

        var sorted = df.Sort("A", ascending: false);
        var col = sorted.GetColumn<double>("A");

        // Descending: 3.0, 2.0, 1.0, NaN (NaN always last)
        col[0].Should().Be(3.0);
        col[1].Should().Be(2.0);
        col[2].Should().Be(1.0);
        double.IsNaN(col[3]!.Value).Should().BeTrue();
    }

    [Fact]
    public void IntegerDivision_ByZero_ProducesNull()
    {
        var num = new Column<int>("a", new int[] { 10, 20, 30 });
        var den = new Column<int>("b", new int[] { 2, 0, 5 });

        var result = num / den;

        result[0].Should().Be(5);
        result[1].Should().BeNull("integer division by zero should produce null");
        result[2].Should().Be(6);
    }

    [Fact]
    public void DoubleDivision_ByZero_ProducesInfinity()
    {
        var num = new Column<double>("a", new double[] { 10.0, 20.0, -30.0 });
        var den = new Column<double>("b", new double[] { 2.0, 0.0, 0.0 });

        var result = num / den;

        result[0].Should().Be(5.0);
        double.IsPositiveInfinity(result[1]!.Value).Should().BeTrue();
        double.IsNegativeInfinity(result[2]!.Value).Should().BeTrue();
    }

    [Fact]
    public void Concat_IntAndLong_WidensToLong()
    {
        var df1 = DataFrame.FromDictionary(new() { ["A"] = new int[] { 1, 2 } });
        var df2 = DataFrame.FromDictionary(new() { ["A"] = new long[] { 3L, 4L } });

        var result = ConcatExtensions.Concat(df1, df2);

        result["A"].DataType.Should().Be(typeof(long));
        result.RowCount.Should().Be(4);
    }

    [Fact]
    public void Concat_FloatAndDouble_WidensToDouble()
    {
        var df1 = DataFrame.FromDictionary(new() { ["A"] = new float[] { 1.0f, 2.0f } });
        var df2 = DataFrame.FromDictionary(new() { ["A"] = new double[] { 3.0, 4.0 } });

        var result = ConcatExtensions.Concat(df1, df2);

        result["A"].DataType.Should().Be(typeof(double));
        result.RowCount.Should().Be(4);
    }

    [Fact]
    public void CsvRoundTrip_AllColumnTypes()
    {
        var df = new DataFrame(
            new Column<int>("IntCol", new int[] { 1, 2, 3 }),
            new Column<double>("DblCol", new double[] { 1.5, 2.5, 3.5 }),
            new StringColumn("StrCol", new string?[] { "hello, world", "foo\"bar", "plain" }),
            new Column<bool>("BoolCol", new bool[] { true, false, true }),
            new Column<DateTime>("DateCol", new DateTime[]
            {
                new(2024, 1, 15), new(2024, 6, 30), new(2024, 12, 25)
            })
        );

        var path = Path.Combine(Path.GetTempPath(), $"round7_test_{Guid.NewGuid()}.csv");
        try
        {
            CsvWriter.Write(df, path);
            var loaded = CsvReader.Read(path);

            loaded.RowCount.Should().Be(3);
            loaded.ColumnCount.Should().Be(5);

            // String with comma should survive round-trip
            ((string)loaded["StrCol"].GetObject(0)!).Should().Be("hello, world");
            // String with quote should survive round-trip
            ((string)loaded["StrCol"].GetObject(1)!).Should().Be("foo\"bar");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ChainedArithmetic_ProducesCorrectResult()
    {
        var col1 = new Column<double>("a", new double[] { 1.0, 2.0, 3.0 });
        var col2 = new Column<double>("b", new double[] { 4.0, 5.0, 6.0 });
        var col3 = new Column<double>("c", new double[] { 2.0, 3.0, 4.0 });
        var col4 = new Column<double>("d", new double[] { 0.5, 1.0, 1.5 });

        // (col1 + col2) * col3 - col4
        var result = (col1 + col2) * col3 - col4;

        // (1+4)*2 - 0.5 = 9.5
        // (2+5)*3 - 1.0 = 20.0
        // (3+6)*4 - 1.5 = 34.5
        result[0].Should().Be(9.5);
        result[1].Should().Be(20.0);
        result[2].Should().Be(34.5);
    }

    [Fact]
    public void SortStability_EqualValues_PreserveInsertionOrder()
    {
        var df = new DataFrame(
            new Column<int>("Key", new int[] { 1, 1, 1, 1 }),
            new Column<int>("Order", new int[] { 10, 20, 30, 40 })
        );

        var sorted = df.Sort("Key");

        // All keys equal — order should be preserved (stable sort)
        var orderCol = sorted.GetColumn<int>("Order");
        orderCol[0].Should().Be(10);
        orderCol[1].Should().Be(20);
        orderCol[2].Should().Be(30);
        orderCol[3].Should().Be(40);
    }

    [Fact]
    public void GroupBy_NullExclude_MultiColumnKey_OnlyOneKeyNull()
    {
        var df = new DataFrame(
            new StringColumn("A", new string?[] { "x", null, "x", "y" }),
            new StringColumn("B", new string?[] { "p", "q", "p", "q" }),
            new Column<int>("Val", new int[] { 1, 2, 3, 4 })
        );

        var result = df.GroupBy(new[] { "A", "B" }, NullGroupingMode.Exclude).Sum();

        // Row 1 (A=null, B="q") should be excluded even though B is not null
        // Only (x,p) and (y,q) groups should remain
        result.RowCount.Should().Be(2);
    }

    // ═══════════════════════════════════════════════════════════════
    // Helper method
    // ═══════════════════════════════════════════════════════════════

    private static int GetGroupRow(DataFrame df, string keyCol, string keyValue)
    {
        for (int i = 0; i < df.RowCount; i++)
        {
            var val = df[keyCol].GetObject(i)?.ToString();
            if (val == keyValue) return i;
        }
        throw new InvalidOperationException($"Group '{keyValue}' not found in column '{keyCol}'");
    }
}
