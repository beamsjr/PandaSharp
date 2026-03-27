using System.Text;
using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Statistics;
using Xunit;

namespace Cortex.Tests.Unit.EdgeCases;

public class CrossModuleEdgeCaseTests
{
    // =====================================================================
    // Bug 1: Corr() crashes with IndexOutOfRangeException when 0 numeric cols
    // =====================================================================
    [Fact]
    public void Corr_ZeroNumericColumns_ReturnsEmptyMatrix()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Name", new[] { "Alice", "Bob" })
        });

        var result = df.Corr();

        result.ColumnCount.Should().BeGreaterThanOrEqualTo(1); // at least "column" col
        result.RowCount.Should().Be(0);
    }

    // =====================================================================
    // Bug 2: GroupBy Std returns 0 for single-element groups (should be NaN)
    // =====================================================================
    [Fact]
    public void GroupBy_Std_SingleElementGroup_ReturnsNaN()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Key", new[] { "A", "B", "C" }),
            new Column<double>("Value", new[] { 10.0, 20.0, 30.0 })
        });

        var result = df.GroupBy("Key").Std();

        // Each group has 1 element -> std should be NaN, not 0
        var stdCol = result["Value"] as Column<double>;
        stdCol.Should().NotBeNull();
        for (int i = 0; i < result.RowCount; i++)
        {
            double val = stdCol!.Values[i];
            double.IsNaN(val).Should().BeTrue(
                $"Std of single-element group at row {i} should be NaN, was {val}");
        }
    }

    // =====================================================================
    // Bug 3: GroupBy Var returns 0 for single-element groups (should be NaN)
    // =====================================================================
    [Fact]
    public void GroupBy_Var_SingleElementGroup_ReturnsNaN()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Key", new[] { "A", "B", "C" }),
            new Column<double>("Value", new[] { 10.0, 20.0, 30.0 })
        });

        var result = df.GroupBy("Key").Var();

        var varCol = result["Value"] as Column<double>;
        varCol.Should().NotBeNull();
        for (int i = 0; i < result.RowCount; i++)
        {
            double val = varCol!.Values[i];
            double.IsNaN(val).Should().BeTrue(
                $"Var of single-element group at row {i} should be NaN, was {val}");
        }
    }

    // =====================================================================
    // Bug 4: CSV Writer does not quote column headers containing delimiter
    // =====================================================================
    [Fact]
    public void CsvRoundTrip_HeadersContainingDelimiter_ArePreserved()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<int>("normal", new[] { 1, 2 }),
            new Column<int>("has,comma", new[] { 3, 4 })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream);
        stream.Position = 0;
        var result = CsvReader.Read(stream);

        result.ColumnNames.Should().Contain("has,comma");
    }

    // =====================================================================
    // Bug 5: CSV Writer does not quote fields containing \r
    // =====================================================================
    [Fact]
    public void CsvRoundTrip_FieldContainingCarriageReturn_IsPreserved()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Text", new[] { "line1\rline2" })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream);
        stream.Position = 0;

        // The \r must be inside quotes; otherwise the reader sees a broken line
        var bytes = stream.ToArray();
        var content = Encoding.UTF8.GetString(bytes);
        // After header line, the data row should contain quotes around the \r
        content.Should().Contain("\"line1\rline2\"");
    }

    // =====================================================================
    // Bug 6: GroupBy.Agg Std on single-element groups returns 0 (should be NaN)
    //        This tests the Agg tuple syntax which goes through ComputeStd
    // =====================================================================
    [Fact]
    public void GroupBy_Agg_Std_SingleElement_ReturnsNaN()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Key", new[] { "A", "B" }),
            new Column<int>("Val", new[] { 10, 20 })
        });

        var result = df.GroupBy("Key").Agg(("Val", AggFunc.Std));

        // Column<int> goes through the generic ComputeStd path
        var stdCol = result["Val_std"];
        for (int i = 0; i < result.RowCount; i++)
        {
            var val = stdCol.GetObject(i);
            val.Should().NotBeNull();
            double d = Convert.ToDouble(val);
            double.IsNaN(d).Should().BeTrue(
                $"Std of single-element group at row {i} should be NaN, was {d}");
        }
    }

    // =====================================================================
    // Bug 7: CSV Writer doesn't quote headers containing quote characters
    // =====================================================================
    [Fact]
    public void CsvRoundTrip_HeadersContainingQuotes_ArePreserved()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<int>("col\"name", new[] { 1, 2 })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream);
        stream.Position = 0;
        var result = CsvReader.Read(stream);

        result.ColumnNames.Should().Contain("col\"name");
    }

    // =====================================================================
    // Non-bug verifications (these should pass without changes)
    // =====================================================================

    [Fact]
    public void Corr_SingleNumericColumn_Returns1x1Matrix()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<double>("X", new[] { 1.0, 2.0, 3.0 })
        });

        var result = df.Corr();

        result.RowCount.Should().Be(1);
        // column "X" should have value 1.0
        var xCol = result["X"] as Column<double>;
        xCol.Should().NotBeNull();
        xCol!.Values[0].Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void Corr_ConstantColumn_ReturnsNaN()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<double>("X", new[] { 5.0, 5.0, 5.0 }),
            new Column<double>("Y", new[] { 1.0, 2.0, 3.0 })
        });

        var result = df.Corr();

        // X has std=0, so correlation with anything should be NaN
        var xCol = result["X"] as Column<double>;
        // X,Y correlation
        double.IsNaN(xCol!.Values[1]).Should().BeTrue("correlation with constant column should be NaN");
    }

    [Fact]
    public void CsvRoundTrip_NullableIntColumn_PreservesNulls()
    {
        var df = new DataFrame(new List<IColumn>
        {
            Column<int>.FromNullable("Val", new int?[] { 1, null, 3 })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream, new CsvWriteOptions { NullRepresentation = "" });
        stream.Position = 0;
        var result = CsvReader.Read(stream, new CsvReadOptions
        {
            ColumnTypes = new Dictionary<string, Type> { ["Val"] = typeof(int) }
        });

        var col = result["Val"] as Column<int>;
        col.Should().NotBeNull();
        col!.IsNull(1).Should().BeTrue("null should be preserved, not become 0");
    }

    [Fact]
    public void CsvRoundTrip_FieldContainingDelimiterInsideQuotes()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Text", new[] { "hello,world", "simple" })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream);
        stream.Position = 0;
        var result = CsvReader.Read(stream);

        var col = result["Text"] as StringColumn;
        col.Should().NotBeNull();
        col!.GetValues()[0].Should().Be("hello,world");
    }

    [Fact]
    public void CsvRoundTrip_FieldContainingOnlyQuotes()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Text", new[] { "\"\"", "normal" })
        });

        using var stream = new MemoryStream();
        CsvWriter.Write(df, stream);
        stream.Position = 0;
        var result = CsvReader.Read(stream);

        var col = result["Text"] as StringColumn;
        col.Should().NotBeNull();
        col!.GetValues()[0].Should().Be("\"\"");
    }

    [Fact]
    public void CsvRoundTrip_TrailingComma_EmptyLastField()
    {
        // CSV with trailing comma means empty last field
        var csv = "A,B,C\n1,2,\n3,4,\n";
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(csv));

        var df = CsvReader.Read(stream);

        df.ColumnCount.Should().Be(3);
        df.RowCount.Should().Be(2);
    }

    [Fact]
    public void CsvRoundTrip_MixedLineEndings()
    {
        // Mix \r\n and \n in same file
        var csv = "A,B\r\n1,2\n3,4\r\n";
        using var stream = new MemoryStream(Encoding.UTF8.GetBytes(csv));

        var df = CsvReader.Read(stream);

        df.RowCount.Should().Be(2);
    }

    [Fact]
    public void CsvRoundTrip_Utf8Bom()
    {
        var bom = new byte[] { 0xEF, 0xBB, 0xBF };
        var csv = "Name,Value\nAlice,1\n";
        var bytes = bom.Concat(Encoding.UTF8.GetBytes(csv)).ToArray();
        using var stream = new MemoryStream(bytes);

        var df = CsvReader.Read(stream);

        df.ColumnNames.Should().Contain("Name");
        df.RowCount.Should().Be(1);
    }

    [Fact]
    public void ParquetRoundTrip_NullableIntColumn()
    {
        var df = new DataFrame(new List<IColumn>
        {
            Column<int>.FromNullable("Val", new int?[] { 1, null, 3, null, 5 })
        });

        var path = Path.Combine(Path.GetTempPath(), $"nullable_int_{Guid.NewGuid()}.parquet");
        try
        {
            ParquetIO.WriteParquet(df, path);
            var result = ParquetIO.ReadParquet(path);

            result.RowCount.Should().Be(5);
            var col = result["Val"];
            col.IsNull(1).Should().BeTrue();
            col.IsNull(3).Should().BeTrue();
            col.GetObject(0).Should().Be(1);
            col.GetObject(2).Should().Be(3);
            col.GetObject(4).Should().Be(5);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ParquetRoundTrip_EmptyStringVsNullString()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new StringColumn("Text", new string?[] { "hello", null, "", "world" })
        });

        var path = Path.Combine(Path.GetTempPath(), $"str_null_{Guid.NewGuid()}.parquet");
        try
        {
            ParquetIO.WriteParquet(df, path);
            var result = ParquetIO.ReadParquet(path);

            var col = result["Text"];
            col.GetObject(0).Should().Be("hello");
            col.IsNull(1).Should().BeTrue();
            (col.GetObject(2) as string).Should().Be("");
            col.GetObject(3).Should().Be("world");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void AvroRoundTrip_BooleanColumn()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<bool>("Flag", new[] { true, false, true })
        });

        var path = Path.Combine(Path.GetTempPath(), $"avro_bool_{Guid.NewGuid()}.avro");
        try
        {
            AvroWriter.Write(df, path);
            var result = AvroReader.Read(path);

            result.RowCount.Should().Be(3);
            result["Flag"].GetObject(0).Should().Be(true);
            result["Flag"].GetObject(1).Should().Be(false);
            result["Flag"].GetObject(2).Should().Be(true);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void OrcRoundTrip_LongColumn()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<long>("BigNum", new[] { long.MinValue, 0L, long.MaxValue })
        });

        var path = Path.Combine(Path.GetTempPath(), $"orc_long_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var result = OrcReader.Read(path);

            result.RowCount.Should().Be(3);
            result["BigNum"].GetObject(0).Should().Be(long.MinValue);
            result["BigNum"].GetObject(1).Should().Be(0L);
            result["BigNum"].GetObject(2).Should().Be(long.MaxValue);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void OrcRoundTrip_MixedNullsAcrossColumns()
    {
        var df = new DataFrame(new List<IColumn>
        {
            Column<int>.FromNullable("A", new int?[] { 1, null, 3 }),
            Column<double>.FromNullable("B", new double?[] { null, 2.0, null })
        });

        var path = Path.Combine(Path.GetTempPath(), $"orc_mixed_null_{Guid.NewGuid()}.orc");
        try
        {
            OrcWriter.Write(df, path);
            var result = OrcReader.Read(path);

            result["A"].IsNull(0).Should().BeFalse();
            result["A"].IsNull(1).Should().BeTrue();
            result["B"].IsNull(0).Should().BeTrue();
            result["B"].IsNull(2).Should().BeTrue();
            result["B"].GetObject(1).Should().Be(2.0);
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void LeftJoin_NoMatchingKeys_AllNullsInRightColumns()
    {
        var left = new DataFrame(new List<IColumn>
        {
            new Column<int>("Key", new[] { 1, 2, 3 }),
            new Column<int>("LVal", new[] { 10, 20, 30 })
        });
        var right = new DataFrame(new List<IColumn>
        {
            new Column<int>("Key", new[] { 100, 200 }),
            new Column<int>("RVal", new[] { 99, 88 })
        });

        var result = left.Join(right, "Key", JoinType.Left);

        result.RowCount.Should().Be(3);
        result["RVal"].IsNull(0).Should().BeTrue();
        result["RVal"].IsNull(1).Should().BeTrue();
        result["RVal"].IsNull(2).Should().BeTrue();
    }

    [Fact]
    public void GroupBy_Boolean_Column()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<bool>("Active", new[] { true, false, true, false }),
            new Column<int>("Value", new[] { 10, 20, 30, 40 })
        });

        var result = df.GroupBy("Active").Sum();

        result.RowCount.Should().Be(2);
    }

    [Fact]
    public void CorrSpearman_TwoRowDataFrame()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<double>("X", new[] { 1.0, 2.0 }),
            new Column<double>("Y", new[] { 3.0, 4.0 })
        });

        var result = df.CorrSpearman();

        result.RowCount.Should().Be(2);
        // With 2 rows, Spearman should still work
        var xyCorr = (result["Y"] as Column<double>)!.Values[0];
        // Perfect monotonic: corr = 1.0
        xyCorr.Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void CorrKendall_TwoRowDataFrame()
    {
        var df = new DataFrame(new List<IColumn>
        {
            new Column<double>("X", new[] { 1.0, 2.0 }),
            new Column<double>("Y", new[] { 3.0, 4.0 })
        });

        var result = df.CorrKendall();

        result.RowCount.Should().Be(2);
        var xyCorr = (result["Y"] as Column<double>)!.Values[0];
        // 2 observations, perfectly concordant: tau = 1.0
        xyCorr.Should().BeApproximately(1.0, 1e-10);
    }
}
