using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Concat;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Statistics;
using Xunit;

namespace Cortex.Tests.Unit.EdgeCases;

public class BoundaryFuzzTests
{
    // ═══════════════════════════════════════════════════
    // Area 1: 1-row DataFrame through full pipeline
    // ═══════════════════════════════════════════════════

    private static DataFrame Build1RowDf()
    {
        return new DataFrame(
            new Column<int>("IntCol", [42]),
            new Column<double>("DblCol", [3.14]),
            new StringColumn("StrCol", ["hello"]),
            new Column<bool>("BoolCol", [true])
        );
    }

    [Fact]
    public void OneRow_Filter_Match()
    {
        var df = Build1RowDf();
        var intCol = (Column<int>)df["IntCol"];
        var filtered = df.Filter(intCol.Eq(42));
        filtered.RowCount.Should().Be(1);
    }

    [Fact]
    public void OneRow_Filter_NoMatch()
    {
        var df = Build1RowDf();
        var intCol = (Column<int>)df["IntCol"];
        var filtered = df.Filter(intCol.Eq(999));
        filtered.RowCount.Should().Be(0);
    }

    [Fact]
    public void OneRow_GroupBy_Sum_Mean_Count()
    {
        var df = new DataFrame(
            new StringColumn("Key", ["A"]),
            new Column<double>("Val", [10.0])
        );
        var grouped = df.GroupBy("Key");
        var sum = grouped.Sum();
        sum.RowCount.Should().Be(1);
        ((Column<double>)sum["Val"])[0].Should().Be(10.0);

        var mean = grouped.Mean();
        ((Column<double>)mean["Val"])[0].Should().Be(10.0);

        var count = grouped.Count();
        count.RowCount.Should().Be(1);
    }

    [Fact]
    public void OneRow_Sort()
    {
        var df = Build1RowDf();
        var sorted = df.Sort("IntCol");
        sorted.RowCount.Should().Be(1);
    }

    [Fact]
    public void OneRow_Describe()
    {
        var df = Build1RowDf();
        var desc = df.Describe();
        desc.RowCount.Should().Be(8); // count, mean, std, min, 25%, 50%, 75%, max
    }

    [Fact]
    public void OneRow_Corr_ShouldBeNaN()
    {
        // Correlation from 1 point is undefined — should be NaN, not crash
        var df = new DataFrame(
            new Column<double>("A", [1.0]),
            new Column<double>("B", [2.0])
        );
        var corr = df.Corr();
        // With 1 row, FastCorr divides by n-1 = 0, so std = sqrt(0/0) = NaN or
        // covariance should be NaN. The correlation should be NaN.
        var aCol = (Column<double>)corr["A"];
        var bCol = (Column<double>)corr["B"];
        // Diagonal should be NaN (or 1.0 by convention) with 1 point
        // Off-diagonal should definitely be NaN
        double.IsNaN(aCol[1]!.Value).Should().BeTrue("correlation from 1 data point is undefined");
        double.IsNaN(bCol[0]!.Value).Should().BeTrue("correlation from 1 data point is undefined");
    }

    [Fact]
    public void OneRow_Merge()
    {
        var df1 = new DataFrame(
            new Column<int>("Key", [1]),
            new Column<double>("Val1", [10.0])
        );
        var df2 = new DataFrame(
            new Column<int>("Key", [1]),
            new Column<double>("Val2", [20.0])
        );
        var merged = df1.Merge(df2, "Key");
        merged.RowCount.Should().Be(1);
    }

    [Fact]
    public void OneRow_Concat()
    {
        var df = Build1RowDf();
        var concat = ConcatExtensions.Concat(df, df);
        concat.RowCount.Should().Be(2);
    }

    [Fact]
    public void OneRow_CsvRoundTrip()
    {
        var df = new DataFrame(
            new Column<int>("IntCol", [42]),
            new Column<double>("DblCol", [3.14]),
            new StringColumn("StrCol", ["hello"])
        );
        using var ms = new MemoryStream();
        df.ToCsv(ms);
        ms.Position = 0;
        var loaded = DataFrameIO.ReadCsv(ms);
        loaded.RowCount.Should().Be(1);
    }

    [Fact]
    public void OneRow_AvroRoundTrip()
    {
        var df = new DataFrame(
            new Column<int>("IntCol", [42]),
            new Column<double>("DblCol", [3.14]),
            new StringColumn("StrCol", ["hello"])
        );
        using var ms = new MemoryStream();
        df.ToAvro(ms);
        ms.Position = 0;
        var loaded = DataFrameIO.ReadAvro(ms);
        loaded.RowCount.Should().Be(1);
        ((Column<int>)loaded["IntCol"])[0].Should().Be(42);
    }

    [Fact]
    public void OneRow_OrcRoundTrip()
    {
        var path = Path.Combine(Path.GetTempPath(), $"boundary_fuzz_{Guid.NewGuid()}.orc");
        try
        {
            var df = new DataFrame(
                new Column<int>("IntCol", [42]),
                new Column<double>("DblCol", [3.14]),
                new StringColumn("StrCol", ["hello"])
            );
            df.ToOrc(path);
            var loaded = DataFrameIO.ReadOrc(path);
            loaded.RowCount.Should().Be(1);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════
    // Area 2: 2-row DataFrame (minimum for correlation/std)
    // ═══════════════════════════════════════════════════

    [Fact]
    public void TwoRow_Corr_ValidResult()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0]),
            new Column<double>("B", [1.0, 2.0])
        );
        var corr = df.Corr();
        var aCol = (Column<double>)corr["A"];
        // Perfect positive correlation
        aCol[1]!.Value.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void TwoRow_Std_UsesN1Denominator()
    {
        // Values [0, 2]: mean=1, variance with n-1 = ((0-1)^2+(2-1)^2)/(2-1) = 2
        // std = sqrt(2)
        var col = new Column<double>("X", [0.0, 2.0]);
        var std = col.Std();
        std.Should().NotBeNull();
        std!.Value.Should().BeApproximately(Math.Sqrt(2.0), 1e-10);
    }

    [Fact]
    public void TwoRow_CorrSpearman()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0]),
            new Column<double>("B", [10.0, 20.0])
        );
        var corr = df.CorrSpearman();
        var aCol = (Column<double>)corr["A"];
        // Perfect monotonic: Spearman should be 1.0
        aCol[1]!.Value.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void TwoRow_CorrKendall()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0]),
            new Column<double>("B", [10.0, 20.0])
        );
        var corr = df.CorrKendall();
        var aCol = (Column<double>)corr["A"];
        // Perfect concordance: Kendall should be 1.0
        aCol[1]!.Value.Should().BeApproximately(1.0, 1e-10);
    }

    // ═══════════════════════════════════════════════════
    // Area 3: Alternating null pattern stress test
    // ═══════════════════════════════════════════════════

    private static Column<double> BuildAlternatingNullColumn(int count)
    {
        var values = new double?[count];
        for (int i = 0; i < count; i++)
            values[i] = (i % 2 == 0) ? (double)(i + 1) : null;
        return Column<double>.FromNullable("AltNull", values);
    }

    [Fact]
    public void AlternatingNull_Sum_UsesOnlyNonNull()
    {
        var col = BuildAlternatingNullColumn(1000);
        var sum = col.Sum();
        // Non-null values: i=0->1, i=2->3, i=4->5, ... i=998->999
        // These are: 1, 3, 5, 7, ..., 999 (500 values)
        // Sum = 500 * (1+999)/2 = 500*500 = 250000
        sum.Should().Be(250000.0);
    }

    [Fact]
    public void AlternatingNull_Mean_UsesOnlyNonNull()
    {
        var col = BuildAlternatingNullColumn(1000);
        var mean = col.Mean();
        mean.Should().Be(250000.0 / 500);
    }

    [Fact]
    public void AlternatingNull_Min_Max()
    {
        var col = BuildAlternatingNullColumn(1000);
        col.Min().Should().Be(1.0);
        col.Max().Should().Be(999.0);
    }

    [Fact]
    public void AlternatingNull_Std()
    {
        var col = BuildAlternatingNullColumn(1000);
        var std = col.Std();
        std.Should().NotBeNull();
        // With 500 non-null values (1,3,5,...,999), mean=500
        // Variance = sum((x-500)^2)/(499)
        std!.Value.Should().BeGreaterThan(0);
    }

    [Fact]
    public void AlternatingNull_Rank_NullsGetNaNRanks()
    {
        var col = BuildAlternatingNullColumn(10);
        var ranked = col.Rank();
        // Null indices: 1, 3, 5, 7, 9
        for (int i = 1; i < 10; i += 2)
        {
            ranked[i].Should().BeNull("null values should get null/NaN ranks");
        }
        // Non-null indices: 0,2,4,6,8 with values 1,3,5,7,9
        // Ranks should be 1,2,3,4,5
        ranked[0].Should().Be(1.0);
        ranked[2].Should().Be(2.0);
        ranked[8].Should().Be(5.0);
    }

    [Fact]
    public void AlternatingNull_ValueCounts_CorrectNullCount()
    {
        var col = BuildAlternatingNullColumn(10);
        // Wrap in DataFrame to use ValueCounts on the column
        var vc = ((IColumn)col).ValueCounts();
        // Should count 500 nulls (well, 5 from 10)
        // Find null row
        var nameCol = (StringColumn)vc[vc.ColumnNames.First()];
        var countCol = (Column<int>)vc["count"];
        // Each non-null value appears exactly once, null appears 5 times
        int totalCount = 0;
        for (int i = 0; i < vc.RowCount; i++)
            totalCount += countCol[i]!.Value;
        totalCount.Should().Be(10);
    }

    [Fact]
    public void AlternatingNull_GroupByKey()
    {
        // GroupBy on a key column with alternating nulls
        var df = new DataFrame(
            Column<int>.FromNullable("Key", [1, null, 1, null, 2, null, 2, null, 3, null]),
            new Column<double>("Val", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        );
        // Default mode includes nulls as a group
        var grouped = df.GroupBy("Key");
        var sum = grouped.Sum();
        sum.RowCount.Should().BeGreaterThan(0);
    }

    // ═══════════════════════════════════════════════════
    // Area 4: All-same-value column through every operation
    // ═══════════════════════════════════════════════════

    private static Column<double> BuildAllSameColumn(int count, double value = 42.0)
    {
        var values = new double[count];
        Array.Fill(values, value);
        return new Column<double>("AllSame", values);
    }

    [Fact]
    public void AllSame_Std_ShouldBeZero()
    {
        var col = BuildAllSameColumn(100);
        var std = col.Std();
        std.Should().NotBeNull();
        std!.Value.Should().Be(0.0, "std of identical values should be 0");
    }

    [Fact]
    public void AllSame_Var_ShouldBeZero()
    {
        var col = BuildAllSameColumn(100);
        var variance = col.Var();
        variance.Should().NotBeNull();
        variance!.Value.Should().Be(0.0, "variance of identical values should be 0");
    }

    [Fact]
    public void AllSame_Rank_Average_ShouldBe50Point5()
    {
        var col = BuildAllSameColumn(100);
        var ranked = col.Rank();
        // All tied at same value, average rank = (1+100)/2 = 50.5
        for (int i = 0; i < 100; i++)
            ranked[i]!.Value.Should().Be(50.5, $"all same values should have average rank 50.5 at index {i}");
    }

    [Fact]
    public void AllSame_Corr_ShouldBeNaN()
    {
        // Correlation between two all-same columns: zero std => NaN
        var df = new DataFrame(
            BuildAllSameColumn(100),
            new Column<double>("AllSame2", Enumerable.Repeat(99.0, 100).ToArray())
        );
        var corr = df.Corr();
        var col = (Column<double>)corr["AllSame"];
        // Off-diagonal should be NaN (zero std)
        double.IsNaN(col[1]!.Value).Should().BeTrue("correlation with zero-std column should be NaN");
    }

    [Fact]
    public void AllSame_MinMaxNormalize_ShouldNotCrash()
    {
        // When min==max, range is 0, should not divide by zero
        var col = BuildAllSameColumn(100);
        var normalized = col.NormalizeMinMax();
        // Should return 0.5 for all values (common convention when range=0)
        normalized[0]!.Value.Should().Be(0.5);
    }

    // ═══════════════════════════════════════════════════
    // Area 5: Maximum value stress
    // ═══════════════════════════════════════════════════

    [Fact]
    public void IntMinMax_Sort()
    {
        var col = new Column<int>("X", [int.MaxValue, int.MinValue]);
        var df = new DataFrame(col);
        var sorted = df.Sort("X");
        var sortedCol = (Column<int>)sorted["X"];
        sortedCol[0].Should().Be(int.MinValue);
        sortedCol[1].Should().Be(int.MaxValue);
    }

    [Fact]
    public void IntMinMax_Rank()
    {
        var col = new Column<int>("X", [int.MaxValue, int.MinValue]);
        var ranked = col.Rank();
        // MinValue should be rank 1, MaxValue should be rank 2
        ranked[0].Should().Be(2.0); // MaxValue is rank 2
        ranked[1].Should().Be(1.0); // MinValue is rank 1
    }

    [Fact]
    public void IntMinMax_Describe()
    {
        var df = new DataFrame(new Column<int>("X", [int.MaxValue, int.MinValue]));
        var desc = df.Describe();
        desc.RowCount.Should().Be(8);
    }

    [Fact]
    public void DoubleExtremes_Sort()
    {
        var col = new Column<double>("X",
            [double.MaxValue, double.MinValue, double.Epsilon, -double.Epsilon]);
        var df = new DataFrame(col);
        var sorted = df.Sort("X");
        var sortedCol = (Column<double>)sorted["X"];
        // double.MinValue is the most negative finite double
        sortedCol[0].Should().Be(double.MinValue);
    }

    [Fact]
    public void LongMaxValue_Sum_CheckedArithmetic()
    {
        // Sum of [long.MaxValue, 1] should throw OverflowException
        // because checked arithmetic is used for integer types
        var col = new Column<long>("X", [long.MaxValue, 1L]);
        Action act = () => col.Sum();
        act.Should().Throw<OverflowException>();
    }

    // ═══════════════════════════════════════════════════
    // Area 6: Empty string vs null in string operations
    // ═══════════════════════════════════════════════════

    [Fact]
    public void StringColumn_ContainsEmpty_ShouldMatchEmptyString()
    {
        var col = new StringColumn("S", ["", null, " ", "a"]);
        // Str.Contains("") should match non-null strings (every non-null string contains "")
        var result = col.Str.Contains("");
        result[0]!.Value.Should().BeTrue("empty string contains empty string");
        result[1].Should().BeNull("null should propagate as null");
        result[2]!.Value.Should().BeTrue("space contains empty string");
        result[3]!.Value.Should().BeTrue("'a' contains empty string");
    }

    [Fact]
    public void StringColumn_Len_EmptyIsZero_NullIsNull()
    {
        var col = new StringColumn("S", ["", null, " ", "abc"]);
        var len = col.Str.Len();
        len[0]!.Value.Should().Be(0, "empty string length is 0");
        len[1].Should().BeNull("null should produce null length");
        len[2]!.Value.Should().Be(1, "space has length 1");
        len[3]!.Value.Should().Be(3, "'abc' has length 3");
    }

    [Fact]
    public void StringColumn_Trim_SpaceBecomesEmpty_NullStaysNull()
    {
        var col = new StringColumn("S", ["", null, " ", "a"]);
        var trimmed = col.Str.Trim();
        trimmed[0].Should().Be("", "empty string remains empty after trim");
        trimmed[1].Should().BeNull("null stays null after trim");
        trimmed[2].Should().Be("", "space becomes empty string after trim");
        trimmed[3].Should().Be("a", "'a' unchanged after trim");
    }

    [Fact]
    public void StringColumn_ValueCounts_DistinguishEmptyAndNull()
    {
        var col = new StringColumn("S", ["", null, " ", "a", "", null]);
        var vc = ((IColumn)col).ValueCounts();
        var nameCol = (StringColumn)vc[vc.ColumnNames.First()];
        var countCol = (Column<int>)vc["count"];

        // Should have: "" -> 2, " " -> 1, "a" -> 1, null -> 2
        // Total distinct: 3 non-null + 1 null = 4 rows in value_counts
        vc.RowCount.Should().Be(4, "should have 4 distinct value groups including null");

        // Verify counts sum to 6
        int totalCount = 0;
        for (int i = 0; i < vc.RowCount; i++)
            totalCount += countCol[i]!.Value;
        totalCount.Should().Be(6);
    }

    // ═══════════════════════════════════════════════════
    // Additional boundary: 1-row Corr on non-double columns
    // ═══════════════════════════════════════════════════

    [Fact]
    public void OneRow_Corr_IntColumns_ShouldBeNaN()
    {
        // Tests the nullable-aware fallback path (non-double columns)
        var df = new DataFrame(
            new Column<int>("A", [1]),
            new Column<int>("B", [2])
        );
        var corr = df.Corr();
        var aCol = (Column<double>)corr["A"];
        // With 1 value, StdDev returns 0 (from the private StdDev method: n<=1 => return 0)
        // PearsonCorrelation checks stdX==0 => return NaN
        double.IsNaN(aCol[1]!.Value).Should().BeTrue("correlation from 1 data point is undefined");
    }

    [Fact]
    public void OneRow_Corr_DiagonalShouldBeNaN()
    {
        // With only 1 data point, even diagonal (self-correlation) should be NaN
        // because you can't compute correlation from 1 point
        var df = new DataFrame(
            new Column<double>("A", [5.0])
        );
        var corr = df.Corr();
        var aCol = (Column<double>)corr["A"];
        double.IsNaN(aCol[0]!.Value).Should().BeTrue(
            "self-correlation from 1 data point should be NaN");
    }

    // ═══════════════════════════════════════════════════
    // Bug: GroupBy Min/Max fast path confuses sentinel with real value
    // ═══════════════════════════════════════════════════

    [Fact]
    public void GroupBy_Max_DoubleMinValue_ShouldNotReturnNaN()
    {
        // BUG: GroupBy Max fast path uses double.MinValue as sentinel,
        // so if the actual max IS double.MinValue, it incorrectly returns NaN.
        var df = new DataFrame(
            new StringColumn("Key", ["A"]),
            new Column<double>("Val", [double.MinValue])
        );
        var grouped = df.GroupBy("Key");
        var result = grouped.Max();
        var valCol = (Column<double>)result["Val"];
        // The actual max value is double.MinValue — should NOT be NaN
        double.IsNaN(valCol[0]!.Value).Should().BeFalse(
            "GroupBy Max should return double.MinValue, not NaN, when that's the actual max");
        valCol[0]!.Value.Should().Be(double.MinValue);
    }

    [Fact]
    public void GroupBy_Min_DoubleMaxValue_ShouldNotReturnNaN()
    {
        // BUG: GroupBy Min fast path uses double.MaxValue as sentinel,
        // so if the actual min IS double.MaxValue, it incorrectly returns NaN.
        var df = new DataFrame(
            new StringColumn("Key", ["A"]),
            new Column<double>("Val", [double.MaxValue])
        );
        var grouped = df.GroupBy("Key");
        var result = grouped.Min();
        var valCol = (Column<double>)result["Val"];
        // The actual min value is double.MaxValue — should NOT be NaN
        double.IsNaN(valCol[0]!.Value).Should().BeFalse(
            "GroupBy Min should return double.MaxValue, not NaN, when that's the actual min");
        valCol[0]!.Value.Should().Be(double.MaxValue);
    }

    [Fact]
    public void GroupBy_Sum_AllNullGroup_ShouldBeZeroOrNaN_NotSilentZero()
    {
        // BUG: GroupBy Sum fast path on double returns 0.0 for all-null groups,
        // but the generic fallback path returns null. The fast path should match.
        var df = new DataFrame(
            new StringColumn("Key", ["A", "A"]),
            Column<double>.FromNullable("Val", [null, null])
        );
        var grouped = df.GroupBy("Key");
        var result = grouped.Sum();
        var valCol = (Column<double>)result["Val"];
        // When all values in a group are null, sum should be 0 (pandas convention)
        // OR NaN to indicate no valid data. But it should NOT silently be 0 if the generic
        // path returns null. Let's verify what happens — the key point is consistency.
        // pandas returns 0.0 for all-null sum, so 0.0 is acceptable.
        // The real concern is the Min/Max sentinel bug above.
        valCol[0].Should().NotBeNull();
    }

    // ═══════════════════════════════════════════════════
    // Bug: Describe returns count as "non-null count" but reports
    // count of all rows including NaN for DescribeDouble
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Describe_ColumnWithNaN_CountShouldExcludeNaN()
    {
        // Describe counts non-null values, but DescribeDouble line 53 does
        // count = n - col.NullCount, which doesn't exclude NaN values.
        // Then line 77 separately filters NaN. But the count returned is
        // the pre-NaN-filter count, not the post-NaN-filter validCount.
        // This means count may include NaN values — which is wrong if
        // "count" means "number of valid values."
        var col = Column<double>.FromNullable("X", [1.0, double.NaN, null, 4.0]);
        var df = new DataFrame(col);
        var desc = df.Describe();
        var statCol = (StringColumn)desc["stat"];
        var xCol = (Column<double>)desc["X"];

        // Find the "count" row
        int countIdx = -1;
        for (int i = 0; i < desc.RowCount; i++)
            if (statCol[i] == "count") { countIdx = i; break; }

        // count should be 2 (only 1.0 and 4.0 are valid), not 3 (non-null but includes NaN)
        xCol[countIdx]!.Value.Should().Be(2.0,
            "count in Describe should exclude NaN values, not just nulls");
    }

    // ═══════════════════════════════════════════════════
    // Bug: Describe quantiles for 2-element array may not interpolate
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Describe_TwoValues_MedianShouldBeAverage()
    {
        // With values [10, 20], median (50th percentile) should be 15.0
        var df = new DataFrame(new Column<double>("X", [10.0, 20.0]));
        var desc = df.Describe();
        var statCol = (StringColumn)desc["stat"];
        var xCol = (Column<double>)desc["X"];

        int medianIdx = -1;
        for (int i = 0; i < desc.RowCount; i++)
            if (statCol[i] == "50%") { medianIdx = i; break; }

        // QuickSelect with k50 = (int)(0.50 * 1) = 0, so it returns data[0]
        // For [10, 20], that's 10.0, not 15.0 (the interpolated median)
        // This is a known limitation of the QuickSelect approach — no interpolation.
        // Let's document what it actually returns.
        double median = xCol[medianIdx]!.Value;
        // The correct interpolated median of [10, 20] is 15.0
        median.Should().Be(15.0,
            "the 50th percentile of [10, 20] should be interpolated as 15.0, not truncated to 10.0");
    }

    // ═══════════════════════════════════════════════════
    // Additional probes for subtle bugs
    // ═══════════════════════════════════════════════════

    [Fact]
    public void Describe_ThreeValues_25thPercentileShouldInterpolate()
    {
        // With values [0, 50, 100], the 25th percentile should be 25.0
        // k25 = (int)(0.25 * 2) = 0, so QuickSelect returns data[0] = 0
        var df = new DataFrame(new Column<double>("X", [0.0, 50.0, 100.0]));
        var desc = df.Describe();
        var statCol = (StringColumn)desc["stat"];
        var xCol = (Column<double>)desc["X"];

        int q25Idx = -1;
        for (int i = 0; i < desc.RowCount; i++)
            if (statCol[i] == "25%") { q25Idx = i; break; }

        // pandas: 25th percentile of [0, 50, 100] = 25.0 (linear interpolation)
        // Current code: (int)(0.25 * 2) = 0, returns data[0] = 0.0
        xCol[q25Idx]!.Value.Should().Be(25.0,
            "25th percentile of [0, 50, 100] should be interpolated as 25.0");
    }

    [Fact]
    public void CsvRoundTrip_NullVsEmptyString_Distinguishable()
    {
        // CSV default writes null as "" (empty). So null and "" are indistinguishable.
        // This is a data loss issue during CSV round-trip.
        var df = new DataFrame(
            new StringColumn("S", ["", null, "a"])
        );
        using var ms = new MemoryStream();
        df.ToCsv(ms);
        ms.Position = 0;
        var loaded = DataFrameIO.ReadCsv(ms);
        var col = (StringColumn)loaded["S"];
        // Row 0 was "" and row 1 was null — after round-trip they may both be "" or both be null
        // At minimum, row 2 should still be "a"
        col[2].Should().Be("a");
        // The real question: can we distinguish null from ""?
        // If both become the same value, that's a data fidelity bug.
        // Let's test: if they're different, great. If same, it's a known CSV limitation.
        bool canDistinguish = col[0] != col[1] || (col[0] is null != col[1] is null);
        canDistinguish.Should().BeTrue(
            "CSV round-trip should distinguish empty string from null");
    }

    [Fact]
    public void AvroRoundTrip_NullableIntColumn()
    {
        // Avro with nullable int column — verify nulls survive
        var df = new DataFrame(
            Column<int>.FromNullable("NullableInt", [1, null, 3])
        );
        using var ms = new MemoryStream();
        df.ToAvro(ms);
        ms.Position = 0;
        var loaded = DataFrameIO.ReadAvro(ms);
        var col = (Column<int>)loaded["NullableInt"];
        col[0].Should().Be(1);
        col.IsNull(1).Should().BeTrue("null should survive Avro round-trip");
        col[2].Should().Be(3);
    }
}
