using FluentAssertions;
using Cortex.Column;
using Cortex.Concat;
using Cortex.IO;
using Cortex.Lazy;
using Cortex.Statistics;
using Xunit;
using static Cortex.Expressions.Expr;

namespace Cortex.Tests.Unit.EdgeCases;

public class UncoveredPathsTests
{
    // ═══════════════════════════════════════════════════════════════
    // Bug 60: DataFrame.Clip silently produces nonsensical results
    //         when lower > upper instead of throwing.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Clip_LowerGreaterThanUpper_ShouldThrow()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new double[] { 1.0, 5.0, 10.0 }
        });

        // lower=10 > upper=5 is nonsensical — should throw
        var act = () => df.Clip(10.0, 5.0);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ColumnClip_LowerGreaterThanUpper_ShouldThrow()
    {
        var col = new Column<double>("A", [1.0, 5.0, 10.0]);

        var act = () => col.Clip(10.0, 5.0);
        act.Should().Throw<ArgumentException>();
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 61: Parquet round-trip loses DateTime column type.
    //         ToParquetType maps DateTime to string (fallback),
    //         so DateTime columns silently become StringColumn after
    //         write + read.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Parquet_DateTime_RoundTrip_PreservesType()
    {
        var dates = new DateTime[]
        {
            new(2024, 1, 15),
            new(2024, 6, 30),
            new(2024, 12, 25)
        };
        var df = new DataFrame(
            new Column<int>("Id", [1, 2, 3]),
            new Column<DateTime>("Date", dates)
        );

        var path = Path.Combine(Path.GetTempPath(), $"datetime_test_{Guid.NewGuid()}.parquet");
        try
        {
            df.ToParquet(path);
            var loaded = DataFrameIO.ReadParquet(path);

            loaded["Date"].DataType.Should().Be(typeof(DateTime),
                "DateTime columns should survive Parquet round-trip without becoming strings");
            loaded["Date"].GetObject(0).Should().Be(new DateTime(2024, 1, 15));
            loaded["Date"].GetObject(2).Should().Be(new DateTime(2024, 12, 25));
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // Bug 62: CSV ReadWithSchema doesn't handle DateTime columns.
    //         When type inference determines DateTime, the schema-based
    //         parser has no DateTime handler and stores it as string.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ReadCsv_WithDateTimeColumn_ShouldParseAsDateTime()
    {
        var csv = "id,date\n1,2024-01-15\n2,2024-06-30\n3,2024-12-25\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var df = DataFrameIO.ReadCsv(stream);

        df["date"].DataType.Should().Be(typeof(DateTime),
            "Date strings should be inferred and parsed as DateTime, not kept as string");
        df["date"].GetObject(0).Should().Be(new DateTime(2024, 1, 15));
    }

    [Fact]
    public void ReadCsv_WithExplicitDateTimeSchema_ShouldParseAsDateTime()
    {
        var csv = "id,date\n1,2024-01-15\n2,2024-06-30\n3,2024-12-25\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var options = new CsvReadOptions
        {
            Schema = new[] { ("id", typeof(int)), ("date", typeof(DateTime)) }
        };
        var df = DataFrameIO.ReadCsv(stream, options);

        df["date"].DataType.Should().Be(typeof(DateTime),
            "Explicit DateTime schema should be honored by ReadWithSchema");
        df["date"].GetObject(0).Should().Be(new DateTime(2024, 1, 15));
    }

    // ═══════════════════════════════════════════════════════════════
    // Non-bug verifications for edge cases (previously untested paths)
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void Nlargest_WithZero_ReturnsEmptyDataFrame()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 3, 1, 4, 1, 5 }
        });

        var result = df.Nlargest(0, "A");
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Nlargest_WithN_GreaterThanRowCount_ReturnsAllRows()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 3, 1, 5 }
        });

        var result = df.Nlargest(100, "A");
        result.RowCount.Should().Be(3);
        // Should be sorted descending
        ((int)result.GetColumn<int>("A")[0]!).Should().Be(5);
    }

    [Fact]
    public void DropColumn_LastColumn_ProducesEmptyColumnDataFrame()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 1, 2, 3 }
        });

        var result = df.DropColumn("A");
        result.ColumnCount.Should().Be(0);
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Concat_PartialColumnOverlap_FillsMissingWithNulls()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 1, 2 },
            ["B"] = new int[] { 10, 20 }
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["B"] = new int[] { 30 },
            ["C"] = new int[] { 300 }
        });

        var result = ConcatExtensions.Concat(df1, df2);
        result.ColumnCount.Should().Be(3);
        result.RowCount.Should().Be(3);
        // df2 has no column A — row 2 of A should be null
        result["A"].IsNull(2).Should().BeTrue();
        // df1 has no column C — rows 0,1 of C should be null
        result["C"].IsNull(0).Should().BeTrue();
        result["C"].IsNull(1).Should().BeTrue();
    }

    [Fact]
    public void Concat_AxisOne_ColumnWise()
    {
        var df1 = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 1, 2 }
        });
        var df2 = DataFrame.FromDictionary(new()
        {
            ["B"] = new int[] { 10, 20 }
        });

        var result = ConcatExtensions.Concat(1, df1, df2);
        result.ColumnCount.Should().Be(2);
        result.ColumnNames.Should().Contain("A");
        result.ColumnNames.Should().Contain("B");
    }

    [Fact]
    public void Shift_Zero_ReturnsSameValues()
    {
        var col = new Column<int>("X", [10, 20, 30]);
        var shifted = col.Shift(0);
        shifted[0].Should().Be(10);
        shifted[1].Should().Be(20);
        shifted[2].Should().Be(30);
    }

    [Fact]
    public void Shift_NegativePeriods_LeadsValues()
    {
        var col = new Column<int>("X", [10, 20, 30]);
        var shifted = col.Shift(-1);
        shifted[0].Should().Be(20);
        shifted[1].Should().Be(30);
        shifted[2].Should().BeNull();
    }

    [Fact]
    public void Shift_GreaterThanLength_AllNulls()
    {
        var col = new Column<int>("X", [10, 20, 30]);
        var shifted = col.Shift(5);
        shifted[0].Should().BeNull();
        shifted[1].Should().BeNull();
        shifted[2].Should().BeNull();
    }

    [Fact]
    public void Between_LowerEqualsUpper_IncludesBoundary()
    {
        var col = new Column<int>("X", [1, 5, 10]);
        var mask = col.Between(5, 5);
        mask[0].Should().BeFalse();
        mask[1].Should().BeTrue();
        mask[2].Should().BeFalse();
    }

    [Fact]
    public void IsIn_EmptyValues_AllFalse()
    {
        var col = new Column<int>("X", [1, 5, 10]);
        var mask = col.IsIn(Array.Empty<int>());
        mask.Should().AllSatisfy(v => v.Should().BeFalse());
    }

    [Fact]
    public void Rank_DenseMethod_ProducesCorrectRanks()
    {
        var col = new Column<int>("X", [10, 20, 20, 30]);
        var ranked = col.Rank(RankMethod.Dense);
        ranked[0].Should().Be(1);
        ranked[1].Should().Be(2);
        ranked[2].Should().Be(2);
        ranked[3].Should().Be(3);
    }

    [Fact]
    public void Rank_FirstMethod_BreaksTiesByPosition()
    {
        var col = new Column<int>("X", [10, 20, 20, 30]);
        var ranked = col.Rank(RankMethod.First);
        ranked[0].Should().Be(1);
        // Two 20s: first occurrence gets rank 2, second gets rank 3
        ranked[1].Should().Be(2);
        ranked[2].Should().Be(3);
        ranked[3].Should().Be(4);
    }

    [Fact]
    public void Lazy_FullChain_FilterSelectGroupByCollect()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Category"] = new[] { "A", "A", "B", "B" },
            ["Value"] = new int[] { 10, 20, 30, 40 }
        });

        var result = df.Lazy()
            .Filter(Col("Value") > Lit(5))
            .GroupBy("Category")
            .Sum()
            .Collect();

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Contain("Category");
    }

    [Fact]
    public void SetIndex_ThenResetIndex_RoundTrips()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Id"] = new int[] { 1, 2, 3 },
            ["Name"] = new[] { "Alice", "Bob", "Charlie" }
        });

        var indexed = df.SetIndex("Id");
        indexed.ColumnCount.Should().Be(1); // Only "Name" remains as visible
        indexed.ColumnNames.Should().NotContain("Id");

        var reset = indexed.ResetIndex();
        reset.ColumnCount.Should().Be(2);
        reset.ColumnNames.Should().Contain("Id");
    }

    [Fact]
    public void ReadCsv_ScientificNotation_ParsesAsDouble()
    {
        var csv = "value\n1.5e-10\n2.0e3\n-3.14e2\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var df = DataFrameIO.ReadCsv(stream);

        df["value"].DataType.Should().Be(typeof(double));
        ((double)df["value"].GetObject(0)!).Should().BeApproximately(1.5e-10, 1e-20);
        ((double)df["value"].GetObject(1)!).Should().BeApproximately(2000.0, 0.01);
    }

    [Fact]
    public void ReadCsv_BooleanValues_CaseInsensitive()
    {
        var csv = "flag\ntrue\nFalse\nTRUE\n";
        using var stream = new MemoryStream(System.Text.Encoding.UTF8.GetBytes(csv));

        var df = DataFrameIO.ReadCsv(stream);

        df["flag"].DataType.Should().Be(typeof(bool));
        df["flag"].GetObject(0).Should().Be(true);
        df["flag"].GetObject(1).Should().Be(false);
        df["flag"].GetObject(2).Should().Be(true);
    }

    [Fact]
    public void Clip_OnlyNumericColumnsClipped_StringsPreserved()
    {
        var df = new DataFrame(
            new StringColumn("Name", ["Alice", "Bob"]),
            new Column<double>("Score", [1.0, 100.0])
        );

        var clipped = df.Clip(10.0, 50.0);
        clipped.GetStringColumn("Name")[0].Should().Be("Alice");
        ((double)clipped["Score"].GetObject(0)!).Should().Be(10.0);
        ((double)clipped["Score"].GetObject(1)!).Should().Be(50.0);
    }

    [Fact]
    public void MultiIndex_SetIndex_WithMultipleColumns_ThenReset()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["Year"] = new int[] { 2023, 2023, 2024 },
            ["Month"] = new int[] { 1, 2, 1 },
            ["Sales"] = new double[] { 100.0, 200.0, 150.0 }
        });

        var indexed = df.SetIndex("Year", "Month");
        indexed.MultiIndex.Should().NotBeNull();
        indexed.MultiIndex!.NLevels.Should().Be(2);
        indexed.ColumnCount.Should().Be(1); // Only Sales

        var reset = indexed.ResetIndex();
        reset.ColumnCount.Should().Be(3);
    }

    [Fact]
    public void Assign_NewColumn_LengthMismatch_Throws()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = new int[] { 1, 2, 3 }
        });

        var wrongLengthCol = new Column<int>("B", [1, 2]);
        var act = () => df.Assign("B", wrongLengthCol);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Sample_WithSeed_IsDeterministic()
    {
        var df = DataFrame.FromDictionary(new()
        {
            ["A"] = Enumerable.Range(0, 100).ToArray()
        });

        var s1 = df.Sample(10, seed: 42);
        var s2 = df.Sample(10, seed: 42);

        s1.RowCount.Should().Be(10);
        for (int i = 0; i < s1.RowCount; i++)
        {
            s1["A"].GetObject(i).Should().Be(s2["A"].GetObject(i));
        }
    }
}
