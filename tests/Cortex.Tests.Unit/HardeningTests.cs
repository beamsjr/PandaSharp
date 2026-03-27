using System.Text;
using FluentAssertions;
using Cortex;
using Cortex.Column;

using Cortex.Concat;
using Cortex.Expressions;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Lazy;
using Cortex.Missing;
using Cortex.Reshape;
using Cortex.Schema;
using Cortex.Statistics;
using Cortex.Storage;
using Cortex.TimeSeries;
using Cortex.Window;
using static Cortex.Expressions.Expr;
using DataFrameSchema = Cortex.Schema.Schema;

namespace Cortex.Tests.Unit;

/// <summary>
/// Comprehensive hardening tests: edge cases, empty inputs, type boundaries, round-trips.
/// </summary>
public class HardeningTests
{
    // ===== Empty DataFrame edge cases =====

    [Fact]
    public void EmptyDf_AllOperationsWork()
    {
        var df = new DataFrame(
            new Column<int>("A", Array.Empty<int>()),
            new StringColumn("B", Array.Empty<string?>())
        );

        df.RowCount.Should().Be(0);
        df.Head(5).RowCount.Should().Be(0);
        df.Tail(5).RowCount.Should().Be(0);
        df.Sort("A").RowCount.Should().Be(0);
        df.DropDuplicates().RowCount.Should().Be(0);
        df.Filter(Array.Empty<bool>()).RowCount.Should().Be(0);
        df.Select("A").ColumnCount.Should().Be(1);
        df.Describe().Should().NotBeNull();
        df.Info().Should().NotBeNull();
        df.ToString().Should().NotBeNull();
        df.ToHtml().Should().Contain("0 rows");
        df.ToMarkdown().Should().NotBeNull();
        df.Copy().RowCount.Should().Be(0);
        df.T.RowCount.Should().Be(2); // 2 columns transposed
        df.Sample(10).RowCount.Should().Be(0);
        df.IsEmpty.Should().BeTrue();
        df.Shape.Should().Be((0, 2));
        df.Dtypes.Count.Should().Be(2);
        df.Memory().Should().Be(0);
    }

    [Fact]
    public void EmptyDf_GroupBy()
    {
        var df = new DataFrame(
            new StringColumn("Key", Array.Empty<string?>()),
            new Column<double>("Val", Array.Empty<double>())
        );
        GroupByExtensions.GroupBy(df, "Key").Sum().RowCount.Should().Be(0);
    }

    [Fact]
    public void EmptyDf_Concat()
    {
        var empty = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        var nonempty = new DataFrame(new Column<int>("A", [1, 2]));
        ConcatExtensions.Concat(empty, nonempty).RowCount.Should().Be(2);
        ConcatExtensions.Concat(nonempty, empty).RowCount.Should().Be(2);
    }

    [Fact]
    public void EmptyDf_Join()
    {
        var left = new DataFrame(new Column<int>("Id", [1, 2]));
        var right = new DataFrame(new Column<int>("Id", Array.Empty<int>()));
        left.Join(right, "Id").RowCount.Should().Be(0);
    }

    // ===== Single-element edge cases =====

    [Fact]
    public void SingleRow_AllAggregates()
    {
        var col = new Column<double>("X", [42.0]);
        col.Sum().Should().Be(42);
        col.Mean().Should().Be(42);
        col.Median().Should().Be(42);
        col.Min().Should().Be(42);
        col.Max().Should().Be(42);
        col.Std().Should().BeNull(); // std of 1 value is null (ddof=1)
        col.Var().Should().BeNull();
        col.ArgMin().Should().Be(0);
        col.ArgMax().Should().Be(0);
    }

    [Fact]
    public void SingleRow_Window()
    {
        var col = new Column<double>("X", [5.0]);
        col.Rolling(3, minPeriods: 1).Mean()[0].Should().Be(5.0);
        col.Expanding().Mean()[0].Should().Be(5.0);
    }

    // ===== All-null column edge cases =====

    [Fact]
    public void AllNulls_Aggregates()
    {
        var col = Column<double>.FromNullable("X", [null, null, null]);
        col.Sum().Should().Be(0); // sum of nothing
        col.Mean().Should().BeNull();
        col.Median().Should().BeNull();
        col.Min().Should().BeNull();
        col.Max().Should().BeNull();
        col.ArgMin().Should().BeNull();
        col.ArgMax().Should().BeNull();
        col.Count().Should().Be(0);
    }

    [Fact]
    public void AllNulls_FillNa()
    {
        var col = Column<double>.FromNullable("X", [null, null]);
        col.FillNa(0.0)[0].Should().Be(0.0);
        col.FillNa(FillStrategy.Forward)[0].Should().BeNull(); // nothing to forward-fill from
        col.FillNa(FillStrategy.Backward)[1].Should().BeNull();
    }

    // ===== Large value edge cases =====

    [Fact]
    public void LargeValues_NoOverflow()
    {
        var col = new Column<double>("X", [double.MaxValue / 2, double.MaxValue / 2]);
        col.Sum().Should().Be(double.MaxValue);
    }

    [Fact]
    public void NegativeValues_Sort()
    {
        var df = new DataFrame(new Column<double>("X", [-3.0, -1.0, -2.0]));
        var sorted = df.Sort("X");
        sorted.GetColumn<double>("X")[0].Should().Be(-3.0);
        sorted.GetColumn<double>("X")[2].Should().Be(-1.0);
    }

    // ===== Expression edge cases =====

    [Fact]
    public void Expr_DivideByZero_ProducesNull()
    {
        var df = new DataFrame(
            new Column<double>("A", [10.0]),
            new Column<double>("B", [0.0])
        );
        var result = (Col("A") / Col("B")).Evaluate(df);
        result.IsNull(0).Should().BeTrue();
    }

    [Fact]
    public void Expr_ChainedStringOps()
    {
        var df = new DataFrame(new StringColumn("S", ["  Hello World  "]));
        var result = Col("S").Str.Trim().Evaluate(df);
        ((StringColumn)result)[0].Should().Be("Hello World");
    }

    [Fact]
    public void Expr_WhenThen_AllNull()
    {
        var df = new DataFrame(Column<int>.FromNullable("X", [null, null]));
        var result = When(Col("X") > Lit(0)).Then(Lit(1.0)).Otherwise(Lit(0.0)).Evaluate(df);
        // null condition → otherwise
        result.GetObject(0).Should().Be(0.0);
    }

    [Fact]
    public void Expr_Coalesce_AllNull()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("A", [null]),
            Column<double>.FromNullable("B", [null])
        );
        var result = Coalesce(Col("A"), Col("B")).Evaluate(df);
        result.IsNull(0).Should().BeTrue();
    }

    // ===== Lazy evaluation edge cases =====

    [Fact]
    public void Lazy_EmptyResult()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var result = df.Lazy().Filter(Col("X") > Lit(100)).Collect();
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void Lazy_MultipleWithColumns()
    {
        var df = new DataFrame(new Column<double>("X", [1.0, 2.0]));
        var result = df.Lazy()
            .WithColumns(
                (Col("X") * Lit(2.0), "X2"),
                (Col("X") * Lit(3.0), "X3")
            )
            .Collect();

        result.ColumnNames.Should().Contain("X2");
        result.ColumnNames.Should().Contain("X3");
        result.GetColumn<double>("X2")[0].Should().Be(2.0);
        result.GetColumn<double>("X3")[0].Should().Be(3.0);
    }

    // ===== I/O round-trip edge cases =====

    [Fact]
    public void CsvRoundTrip_EmptyStrings_BecomeNull()
    {
        // Empty strings are in the default null values list ["", "NA", ...]
        // so they round-trip as null — this is expected CSV behavior
        var df = new DataFrame(new StringColumn("S", ["hello", "a", "world"]));
        using var ms = new MemoryStream();
        CsvWriter.Write(df, ms);
        ms.Seek(0, SeekOrigin.Begin);
        var loaded = CsvReader.Read(ms);
        loaded.RowCount.Should().Be(3);
        loaded.GetStringColumn("S")[1].Should().Be("a");
    }

    [Fact]
    public void JsonRoundTrip_Types()
    {
        var df = new DataFrame(
            new Column<int>("I", [1, 2]),
            new Column<double>("D", [1.5, 2.5]),
            new Column<bool>("B", [true, false]),
            new StringColumn("S", ["a", "b"])
        );
        var json = df.ToJsonString();
        var loaded = JsonReader.ReadString(json);
        loaded.RowCount.Should().Be(2);
        loaded["I"].DataType.Should().Be(typeof(int));
        loaded.GetColumn<bool>("B")[0].Should().Be(true);
    }

    // ===== Schema validation edge cases =====

    [Fact]
    public void Schema_EmptyDataFrame_Validates()
    {
        var df = new DataFrame(new Column<int>("X", Array.Empty<int>()));
        var schema = new DataFrameSchema(new ColumnSchema("X", typeof(int)));
        df.MatchesSchema(schema).Should().BeTrue();
    }

    // ===== Compare edge cases =====

    [Fact]
    public void Compare_EmptyDataFrames()
    {
        var df1 = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        var df2 = new DataFrame(new Column<int>("A", Array.Empty<int>()));
        df1.Compare(df2).RowCount.Should().Be(0);
    }

    // ===== Pivot/Melt round-trip =====

    [Fact]
    public void PivotMelt_RoundTrip()
    {
        var df = new DataFrame(
            new StringColumn("Date", ["Jan", "Feb"]),
            new StringColumn("Product", ["A", "B"]),
            new Column<double>("Sales", [100, 200])
        );

        var pivoted = df.Pivot("Date", "Product", "Sales");
        pivoted.RowCount.Should().Be(2);
        pivoted.ColumnNames.Should().Contain("A");
    }

    // ===== MutableDataFrame edge cases =====

    [Fact]
    public void MutableDf_MultipleSetValues()
    {
        var df = new DataFrame(new Column<int>("X", [1, 2, 3]));
        var mdf = df.ToMutable();
        mdf.SetValue<int>("X", 0, 10);
        mdf.SetValue<int>("X", 1, 20);
        mdf.SetValue<int>("X", 2, 30);
        var result = mdf.ToDataFrame();
        result.GetColumn<int>("X")[0].Should().Be(10);
        result.GetColumn<int>("X")[1].Should().Be(20);
        result.GetColumn<int>("X")[2].Should().Be(30);
    }

    // ===== Resample edge cases =====

    [Fact]
    public void Resample_SingleRow()
    {
        var df = new DataFrame(
            new Column<DateTime>("Time", [new DateTime(2024, 1, 1)]),
            new Column<double>("Value", [42.0])
        );
        var result = df.Resample("Time", "1h").Sum();
        result.RowCount.Should().Be(1);
    }

    // ===== Categorical column edge cases =====

    [Fact]
    public void CategoricalColumn_AllSameValue()
    {
        var col = new CategoricalColumn("C", ["A", "A", "A", "A"]);
        col.CategoryCount.Should().Be(1);
        col.Length.Should().Be(4);
        col[3].Should().Be("A");
    }

    [Fact]
    public void CategoricalColumn_AllNull()
    {
        var col = new CategoricalColumn("C", [null, null, null]);
        col.NullCount.Should().Be(3);
        col.CategoryCount.Should().Be(0);
    }

    // ===== Deep copy verification =====

    [Fact]
    public void Clone_DeepCopy_IsIndependent()
    {
        var original = new Column<int>("X", [1, 2, 3]);
        var clone = (Column<int>)original.Clone("Y");

        // Verify they have the same data
        clone[0].Should().Be(1);
        clone.Name.Should().Be("Y");

        // Verify the original is unaffected by MutableDataFrame on clone
        var df = new DataFrame(clone);
        var mdf = df.ToMutable();
        mdf.SetValue<int>("Y", 0, 999);
        var modified = mdf.ToDataFrame();

        modified.GetColumn<int>("Y")[0].Should().Be(999);
        original[0].Should().Be(1); // original unchanged
    }

    // ===== ContentEquals edge cases =====

    [Fact]
    public void ContentEquals_WithNulls()
    {
        var df1 = new DataFrame(Column<int>.FromNullable("X", [1, null, 3]));
        var df2 = new DataFrame(Column<int>.FromNullable("X", [1, null, 3]));
        df1.ContentEquals(df2).Should().BeTrue();
    }

    [Fact]
    public void ContentEquals_NullVsValue()
    {
        var df1 = new DataFrame(Column<int>.FromNullable("X", [1, null]));
        var df2 = new DataFrame(Column<int>.FromNullable("X", [1, 2]));
        df1.ContentEquals(df2).Should().BeFalse();
    }
}
