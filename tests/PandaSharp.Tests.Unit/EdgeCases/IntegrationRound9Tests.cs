using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Concat;
using PandaSharp.GroupBy;
using PandaSharp.IO;
using PandaSharp.Missing;
using PandaSharp.Statistics;
using Xunit;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class IntegrationRound9Tests
{
    // ═══════════════════════════════════════════════════════════════
    // Bug 128: DropDuplicates typed comparers ignore null bitmask.
    //          A nullable Column<double> with a null row (buffer=0)
    //          is treated as equal to a real 0.0 row, causing
    //          incorrect dedup.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_NullableDoubleColumn_NullNotEqualToZero()
    {
        // Row 0: null (bitmask null, buffer value = 0 by default)
        // Row 1: 0.0 (real value)
        // Row 2: 1.0
        var col = Column<double>.FromNullable("Val", new double?[] { null, 0.0, 1.0 });
        var df = new DataFrame(col);

        var deduped = df.DropDuplicates("Val");

        // All 3 rows should remain: null != 0.0
        deduped.RowCount.Should().Be(3,
            "null and 0.0 are distinct values — DropDuplicates should not treat them as equal");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 129: DropDuplicates typed hashers ignore null bitmask.
    //          Same root cause as Bug 128 — the hasher hashes the
    //          buffer value (0) for null rows, same as real 0.
    //          Also affects int and long columns.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_NullableIntColumn_NullNotEqualToZero()
    {
        var col = Column<int>.FromNullable("Val", new int?[] { null, 0, 1 });
        var df = new DataFrame(col);

        var deduped = df.DropDuplicates("Val");

        deduped.RowCount.Should().Be(3,
            "null and 0 are distinct values in a nullable int column");
    }

    // ═══════════════════════════════════════════════════════════════
    // Regression test: GroupBy.First preserves null for nullable int
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void GroupByFirst_NullableInt_PreservesNull()
    {
        var dfWithNull = new DataFrame(
            new StringColumn("Key", new[] { "A", "B" }),
            Column<int>.FromNullable("Val", new int?[] { null, 5 })
        );

        var result = dfWithNull.GroupBy("Key").First();

        // Group A's first value is null — should remain null, not become 0
        var aRow = GetGroupRow(result, "Key", "A");
        result["Val"].IsNull(aRow).Should().BeTrue(
            "GroupBy.First on a group where all values are null should produce null, not 0");
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 131: Interpolate<T> for Column<float> ignores NaN values.
    //          The generic Interpolate<T> checks NullCount == 0 and
    //          returns early, but NaN in float columns is not counted
    //          in NullCount. The Column<double> overload checks for
    //          NaN explicitly but the generic one doesn't.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Interpolate_FloatColumnWithNaN_ShouldInterpolate()
    {
        var col = new Column<float>("Val", new float[] { 1.0f, float.NaN, 3.0f });

        var interpolated = col.Interpolate();

        // NaN at index 1 should be interpolated to 2.0
        var result = interpolated[1];
        result.Should().NotBeNull();
        float.IsNaN(result!.Value).Should().BeFalse(
            "NaN should be interpolated, not left unchanged");
        result.Value.Should().BeApproximately(2.0f, 0.001f);
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 132: ORC writer does not support DateTime columns.
    //          GetOrcTypeCode throws NotSupportedException for
    //          DateTime, so any DataFrame with DateTime columns
    //          cannot be written to ORC format.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void OrcRoundTrip_DateTimeColumn_ShouldWork()
    {
        var dt1 = new DateTime(2024, 1, 15, 10, 30, 0, DateTimeKind.Utc);
        var dt2 = new DateTime(2024, 6, 20, 14, 45, 30, DateTimeKind.Utc);
        var df = new DataFrame(
            new Column<DateTime>("Timestamp", new[] { dt1, dt2 })
        );

        var path = Path.Combine(Path.GetTempPath(), $"orc_datetime_test_{Guid.NewGuid()}.orc");
        try
        {
            df.ToOrc(path);
            var loaded = DataFrameIO.ReadOrc(path);

            loaded.RowCount.Should().Be(2);
            loaded.ColumnNames.Should().Contain("Timestamp");

            // Verify values round-trip correctly (to millisecond precision at minimum)
            var v0 = loaded["Timestamp"].GetObject(0);
            var v1 = loaded["Timestamp"].GetObject(1);
            v0.Should().NotBeNull();
            v1.Should().NotBeNull();

            // ORC stores DateTime as ticks (long), round-trip produces DateTime
            if (v0 is DateTime loadedDt0)
                loadedDt0.Should().Be(dt1);
            else if (v0 is long ticks0)
                new DateTime(ticks0, DateTimeKind.Utc).Should().Be(dt1);
            else
                throw new Exception($"Unexpected type for DateTime column: {v0?.GetType().Name}");

            if (v1 is DateTime loadedDt1)
                loadedDt1.Should().Be(dt2);
            else if (v1 is long ticks1)
                new DateTime(ticks1, DateTimeKind.Utc).Should().Be(dt2);
            else
                throw new Exception($"Unexpected type for DateTime column: {v1?.GetType().Name}");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 133: Avro writer does not support DateTime columns.
    //          GetAvroType throws NotSupportedException for DateTime.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void AvroRoundTrip_DateTimeColumn_ShouldWork()
    {
        var dt1 = new DateTime(2024, 1, 15, 10, 30, 0, DateTimeKind.Utc);
        var dt2 = new DateTime(2024, 6, 20, 14, 45, 30, DateTimeKind.Utc);
        var df = new DataFrame(
            new Column<DateTime>("Timestamp", new[] { dt1, dt2 })
        );

        var path = Path.Combine(Path.GetTempPath(), $"avro_datetime_test_{Guid.NewGuid()}.avro");
        try
        {
            df.ToAvro(path);
            var loaded = DataFrameIO.ReadAvro(path);

            loaded.RowCount.Should().Be(2);
            loaded.ColumnNames.Should().Contain("Timestamp");

            var v0 = loaded["Timestamp"].GetObject(0);
            var v1 = loaded["Timestamp"].GetObject(1);
            v0.Should().NotBeNull();
            v1.Should().NotBeNull();

            if (v0 is DateTime loadedDt0)
                loadedDt0.Should().Be(dt1);
            else if (v0 is long ticks0)
                new DateTime(ticks0, DateTimeKind.Utc).Should().Be(dt1);
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 134: Sort generic path puts NaN FIRST for nullable double
    //          columns, but the fast path puts NaN LAST.
    //          When dc.NullCount > 0, Sort falls to the generic
    //          comparator where NaN.CompareTo(anything) returns -1,
    //          so NaN sorts before all real values. The fast path
    //          explicitly sorts NaN last. This is inconsistent.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Sort_NullableDoubleWithNaN_NaNShouldBeLast()
    {
        // Create a nullable double column with both null and NaN
        var col = Column<double>.FromNullable("Val",
            new double?[] { 3.0, null, double.NaN, 1.0 });
        var df = new DataFrame(col);

        var sorted = df.Sort("Val", ascending: true);

        // Expected order: 1.0, 3.0, NaN, null (reals sorted ascending, then NaN, then null)
        var v0 = sorted["Val"].GetObject(0);
        var v1 = sorted["Val"].GetObject(1);
        var v2 = sorted["Val"].GetObject(2);
        var v3 = sorted["Val"].GetObject(3);

        // First two should be real values in ascending order
        v0.Should().Be(1.0);
        v1.Should().Be(3.0);

        // NaN and null should be last (order between them is implementation-defined,
        // but neither should appear before real values)
        bool v2IsNaNOrNull = v2 is null || (v2 is double d2 && double.IsNaN(d2));
        bool v3IsNaNOrNull = v3 is null || (v3 is double d3 && double.IsNaN(d3));
        v2IsNaNOrNull.Should().BeTrue("NaN and null should sort after real values");
        v3IsNaNOrNull.Should().BeTrue("NaN and null should sort after real values");
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: Filter → GroupBy → Sum chain
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void FilterGroupBySum_Chain_CorrectResult()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "A", "B", "B", "A" },
            ["Val"] = new double[] { 10, 20, 5, 15, 30 }
        });

        // Filter val > 10, then GroupBy, then Sum
        var filtered = df.WhereDouble("Val", v => v > 10);
        var result = filtered.GroupBy("Key").Sum();

        // A: 20 + 30 = 50, B: 15
        var aRow = GetGroupRow(result, "Key", "A");
        ((double)result["Val"].GetObject(aRow)!).Should().Be(50.0);

        var bRow = GetGroupRow(result, "Key", "B");
        ((double)result["Val"].GetObject(bRow)!).Should().Be(15.0);
    }

    [Fact]
    public void FilterToEmpty_ThenGroupBy_ShouldNotCrash()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "B" },
            ["Val"] = new double[] { 1.0, 2.0 }
        });

        // Filter to 0 rows
        var empty = df.WhereDouble("Val", v => v > 100);
        empty.RowCount.Should().Be(0);

        // GroupBy on empty should work without crashing
        var result = empty.GroupBy("Key").Sum();
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void FilterSortHead_Chain_DataIntegrity()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "E", "D", "C", "B", "A", "F", "G", "H", "I", "J" },
            ["Val"] = new double[] { 50, 40, 30, 20, 10, 60, 70, 80, 90, 100 }
        });

        // Filter val >= 30, Sort by Val ascending, Head(3)
        var result = df.WhereDouble("Val", v => v >= 30)
            .Sort("Val", ascending: true)
            .Head(3);

        result.RowCount.Should().Be(3);
        ((double)result["Val"].GetObject(0)!).Should().Be(30.0);
        ((double)result["Val"].GetObject(1)!).Should().Be(40.0);
        ((double)result["Val"].GetObject(2)!).Should().Be(50.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: RenameColumn → GroupBy chain
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void RenameColumn_ThenGroupBy_FindsRenamedColumn()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["OldKey"] = new[] { "A", "A", "B" },
            ["Val"] = new double[] { 1.0, 2.0, 3.0 }
        });

        var renamed = df.RenameColumn("OldKey", "NewKey");
        var result = renamed.GroupBy("NewKey").Sum();

        result.ColumnNames.Should().Contain("NewKey");
        var aRow = GetGroupRow(result, "NewKey", "A");
        ((double)result["Val"].GetObject(aRow)!).Should().Be(3.0);
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: Type preservation through operations
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void IntColumn_AfterFilter_StillIntColumn()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Val"] = new[] { 1, 2, 3, 4, 5 }
        });

        var filtered = df.WhereInt("Val", v => v > 2);
        filtered["Val"].DataType.Should().Be(typeof(int),
            "Column<int> should remain Column<int> after filter, not become Column<double>");
        filtered["Val"].Should().BeOfType<Column<int>>();
    }

    [Fact]
    public void StringColumn_AfterFilter_StillStringColumn()
    {
        var df = new DataFrame(
            new StringColumn("Name", new[] { "Alice", "Bob", "Charlie" }),
            new Column<int>("Age", new[] { 25, 30, 35 })
        );

        var filtered = df.WhereInt("Age", a => a > 25);
        filtered["Name"].Should().BeOfType<StringColumn>();
        filtered["Name"].DataType.Should().Be(typeof(string));
    }

    [Fact]
    public void IntColumn_AfterSort_StillIntColumn()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Val"] = new[] { 3, 1, 2 }
        });

        var sorted = df.Sort("Val");
        sorted["Val"].Should().BeOfType<Column<int>>();
        ((int)sorted["Val"].GetObject(0)!).Should().Be(1);
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: Head(n).DescribeAll chain
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Head_ThenDescribeAll_ShouldWork()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }
        });

        var sliced = df.Head(5);
        var desc = sliced.DescribeAll();

        desc.RowCount.Should().BeGreaterThan(0);
        // count should be "5" for a 5-row slice
        var countRow = -1;
        for (int i = 0; i < desc.RowCount; i++)
            if (desc["stat"].GetObject(i)?.ToString() == "count") { countRow = i; break; }

        countRow.Should().BeGreaterThanOrEqualTo(0);
        desc["A"].GetObject(countRow)?.ToString().Should().Be("5");
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: GroupBy on result of Concat
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Concat_ThenGroupBy_CorrectGroupKeys()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "B" },
            ["Val"] = new double[] { 1, 2 }
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["Key"] = new[] { "A", "C" },
            ["Val"] = new double[] { 3, 4 }
        });

        var combined = DataFrame.Concat(df1, df2);
        var result = combined.GroupBy("Key").Sum();

        result.RowCount.Should().Be(3); // A, B, C
        var aRow = GetGroupRow(result, "Key", "A");
        ((double)result["Val"].GetObject(aRow)!).Should().Be(4.0); // 1 + 3
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: Thread safety reading/filtering DataFrame
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ConcurrentFilter_SameDataFrame_EachGetsIndependentResult()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Val"] = Enumerable.Range(0, 1000).Select(i => (double)i).ToArray()
        });

        var results = new DataFrame[10];
        System.Threading.Tasks.Parallel.For(0, 10, i =>
        {
            double threshold = i * 100;
            results[i] = df.WhereDouble("Val", v => v >= threshold);
        });

        for (int i = 0; i < 10; i++)
        {
            int expected = 1000 - (i * 100);
            results[i].RowCount.Should().Be(expected,
                $"thread {i} with threshold {i * 100} should get {expected} rows");
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: CSV round-trip with NaN, null, empty string
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void CsvRoundTrip_NaNAndNull_PreservedCorrectly()
    {
        var df = new DataFrame(
            new Column<double>("Dbl", new double[] { 1.5, double.NaN, 3.0 }),
            new StringColumn("Str", new string?[] { "hello", null, "" })
        );

        var path = Path.Combine(Path.GetTempPath(), $"csv_nan_test_{Guid.NewGuid()}.csv");
        try
        {
            df.ToCsv(path);
            var loaded = DataFrameIO.ReadCsv(path);

            loaded.RowCount.Should().Be(3);

            // NaN should round-trip
            var dbl1 = loaded["Dbl"].GetObject(1);
            dbl1.Should().NotBeNull();
            if (dbl1 is double d1)
                double.IsNaN(d1).Should().BeTrue("NaN should survive CSV round-trip");

            // null string should remain null
            loaded["Str"].IsNull(1).Should().BeTrue("null string should survive CSV round-trip");

            // empty string should remain empty string (not null)
            var str2 = loaded["Str"].GetObject(2);
            str2.Should().NotBeNull("empty string should not become null after CSV round-trip");
            str2.Should().Be("");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: ORC round-trip with mixed types
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void OrcRoundTrip_MixedTypes_PreservedCorrectly()
    {
        var df = new DataFrame(
            new Column<int>("IntCol", new[] { 1, 2, 3 }),
            new Column<double>("DblCol", new[] { 1.5, double.NaN, 3.0 }),
            new StringColumn("StrCol", new string?[] { "hello", null, "" }),
            new Column<bool>("BoolCol", new[] { true, false, true })
        );

        var path = Path.Combine(Path.GetTempPath(), $"orc_mixed_test_{Guid.NewGuid()}.orc");
        try
        {
            df.ToOrc(path);
            var loaded = DataFrameIO.ReadOrc(path);

            loaded.RowCount.Should().Be(3);
            loaded.ColumnCount.Should().Be(4);

            // Int column
            ((int)loaded["IntCol"].GetObject(0)!).Should().Be(1);

            // Double column with NaN
            var dbl1 = (double)loaded["DblCol"].GetObject(1)!;
            double.IsNaN(dbl1).Should().BeTrue();

            // String null preserved
            loaded["StrCol"].IsNull(1).Should().BeTrue();
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: Avro round-trip with mixed types
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void AvroRoundTrip_MixedTypes_PreservedCorrectly()
    {
        var df = new DataFrame(
            new Column<int>("IntCol", new[] { 1, 2, 3 }),
            new Column<double>("DblCol", new[] { 1.5, 2.5, 3.0 }),
            new StringColumn("StrCol", new string?[] { "hello", null, "" }),
            new Column<bool>("BoolCol", new[] { true, false, true })
        );

        var path = Path.Combine(Path.GetTempPath(), $"avro_mixed_test_{Guid.NewGuid()}.avro");
        try
        {
            df.ToAvro(path);
            var loaded = DataFrameIO.ReadAvro(path);

            loaded.RowCount.Should().Be(3);
            ((int)loaded["IntCol"].GetObject(0)!).Should().Be(1);
            loaded["StrCol"].IsNull(1).Should().BeTrue();
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Integration test: DropDuplicates with NaN
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void DropDuplicates_DoubleColumn_NaN_NaN_Real_ShouldKeepTwo()
    {
        // [NaN, NaN, 1.0] — should deduplicate to [NaN, 1.0] (2 rows)
        var df = new DataFrame(
            new Column<double>("Val", new[] { double.NaN, double.NaN, 1.0 })
        );

        var deduped = df.DropDuplicates("Val");
        deduped.RowCount.Should().Be(2, "two NaN values should be considered duplicates");
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
