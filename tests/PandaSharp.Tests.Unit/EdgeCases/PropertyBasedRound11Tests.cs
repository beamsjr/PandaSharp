using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Concat;
using PandaSharp.GroupBy;
using PandaSharp.Missing;
using PandaSharp.Statistics;
using Xunit;

namespace PandaSharp.Tests.Unit.EdgeCases;

public class PropertyBasedRound11Tests
{
    // ===============================================================
    // Bug 142: Column.PctChange(periods) with negative periods causes
    //          ArrayIndexOutOfRangeException because the loop starts
    //          at index = periods (negative), and accesses span[i]
    //          with i < 0. Negative periods means "compare to a
    //          future value" and should be valid.
    // ===============================================================

    [Fact]
    public void PctChange_NegativePeriods_ShouldNotThrow()
    {
        var col = new Column<double>("Price", new[] { 100.0, 110.0, 121.0, 133.1 });

        var result = col.PctChange(-1);

        result.Length.Should().Be(4);
        // With periods = -1, pct_change[i] = (x[i] - x[i+1]) / x[i+1]
        // pct_change[0] = (100 - 110) / 110 = -10/110
        // pct_change[3] should be null (no future value)
        result[0].Should().BeApproximately(-10.0 / 110.0, 1e-10);
        result[1].Should().BeApproximately(-11.0 / 121.0, 1e-10);
        result[2].Should().BeApproximately(-12.1 / 133.1, 1e-10);
        result[3].Should().BeNull("no future value to compare to");
    }

    // ===============================================================
    // Bug 143: Column.Diff(periods) with negative periods causes
    //          the same ArrayIndexOutOfRangeException as Bug 142.
    //          Negative periods means "diff with future value".
    // ===============================================================

    [Fact]
    public void Diff_NegativePeriods_ShouldNotThrow()
    {
        var col = new Column<double>("Val", new[] { 10.0, 20.0, 30.0, 40.0 });

        var result = col.Diff(-1);

        result.Length.Should().Be(4);
        // With periods = -1, diff[i] = x[i] - x[i+1]
        result[0].Should().Be(-10.0);
        result[1].Should().Be(-10.0);
        result[2].Should().Be(-10.0);
        result[3].Should().BeNull("no future value to diff against");
    }

    // ===============================================================
    // Bug 144: FastCorrWide is dispatched BEFORE the NaN/null check
    //          in FastCorr. When k > 100 columns and n < k rows, the
    //          wide-matrix path is used even if columns contain NaN.
    //          FastCorrWide uses (n-1) in the correlation denominator
    //          rather than (validCount-1), producing diagonal values
    //          != 1.0 when columns have different NaN patterns.
    // ===============================================================

    [Fact]
    public void Corr_WideMatrixWithNaN_DiagonalShouldBeOne()
    {
        // 101 columns, 50 rows → triggers FastCorrWide (k > 100 && n < k)
        int k = 101;
        int n = 50;

        var columns = new List<IColumn>();
        for (int c = 0; c < k; c++)
        {
            var values = new double[n];
            for (int r = 0; r < n; r++)
                values[r] = r * (c + 1) + 1.0; // non-constant, varying by column

            // Introduce NaN in half the columns at different positions
            if (c % 2 == 0 && c < 50)
                values[c % n] = double.NaN;

            columns.Add(new Column<double>($"col_{c}", values));
        }

        var df = new DataFrame(columns);
        var corr = df.Corr();

        // Diagonal of correlation matrix must be 1.0 for non-constant columns
        for (int c = 0; c < k; c++)
        {
            var colName = $"col_{c}";
            var corrCol = (Column<double>)corr[colName];
            double diagValue = corrCol.Buffer.Span[c];

            if (!double.IsNaN(diagValue))
            {
                diagValue.Should().BeApproximately(1.0, 1e-10,
                    $"Diagonal corr[{c},{c}] for non-constant column '{colName}' should be 1.0, got {diagValue}");
            }
        }
    }

    // ===============================================================
    // Property verification: algebraic identities
    // ===============================================================

    [Fact]
    public void Property_AddThenSubtract_ShouldEqualOriginal()
    {
        var col1 = new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });
        var col2 = new Column<double>("B", new[] { 10.0, 20.0, 30.0, 40.0, 50.0 });

        var result = col1.Add(col2).Subtract(col2);

        for (int i = 0; i < col1.Length; i++)
            result[i]!.Value.Should().BeApproximately(col1[i]!.Value, 1e-10,
                $"(col1 + col2) - col2 should equal col1 at index {i}");
    }

    [Fact]
    public void Property_MultiplyByOne_ShouldEqualOriginal()
    {
        var col = new Column<double>("A", new[] { 1.0, -2.5, 0.0, 100.0, -0.001 });

        var result = col.Multiply(1.0);

        for (int i = 0; i < col.Length; i++)
            result[i]!.Value.Should().Be(col[i]!.Value,
                $"col * 1.0 should equal col at index {i}");
    }

    [Fact]
    public void Property_DoubleNegate_ShouldEqualOriginal()
    {
        var col = new Column<double>("A", new[] { 1.0, -2.5, 0.0, 100.0, -0.001 });

        var result = col.Negate().Negate();

        for (int i = 0; i < col.Length; i++)
            result[i]!.Value.Should().Be(col[i]!.Value,
                $"-(-col) should equal col at index {i}");
    }

    [Fact]
    public void Property_SortIdempotent()
    {
        var col = new Column<double>("A", new[] { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0 });
        var df = new DataFrame(col);

        var sorted1 = df.Sort("A");
        var sorted2 = sorted1.Sort("A");

        var c1 = (Column<double>)sorted1["A"];
        var c2 = (Column<double>)sorted2["A"];

        for (int i = 0; i < c1.Length; i++)
            c2[i]!.Value.Should().Be(c1[i]!.Value,
                $"Sort should be idempotent at index {i}");
    }

    // ===============================================================
    // Property verification: statistical identities
    // ===============================================================

    [Fact]
    public void Property_MeanEqualsSumOverCount()
    {
        var col = new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var mean = col.Mean();
        var sum = col.Sum();
        var count = col.ValidCount();

        mean.Should().BeApproximately(sum!.Value / count, 1e-10);
    }

    [Fact]
    public void Property_VarEqualsStdSquared()
    {
        var col = new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var variance = col.Var();
        var std = col.Std();

        variance!.Value.Should().BeApproximately(std!.Value * std!.Value, 1e-10);
    }

    [Fact]
    public void Property_CumSumLastEqualSum()
    {
        var col = new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 });

        var cumSum = col.CumSum();
        var sum = col.Sum();

        cumSum[col.Length - 1]!.Value.Should().BeApproximately(sum!.Value, 1e-10);
    }

    [Fact]
    public void Property_CumMaxLastEqualMax()
    {
        var col = new Column<double>("A", new[] { 3.0, 1.0, 5.0, 2.0, 4.0 });

        var cumMax = col.CumMax();
        var max = col.Max();

        cumMax[col.Length - 1]!.Value.Should().Be(max!.Value);
    }

    [Fact]
    public void Property_CumMinLastEqualMin()
    {
        var col = new Column<double>("A", new[] { 3.0, 1.0, 5.0, 2.0, 4.0 });

        var cumMin = col.CumMin();
        var min = col.Min();

        cumMin[col.Length - 1]!.Value.Should().Be(min!.Value);
    }

    // ===============================================================
    // Property verification: invariants
    // ===============================================================

    [Fact]
    public void Property_HeadRowCount()
    {
        var col = new Column<int>("A", new[] { 1, 2, 3 });
        var df = new DataFrame(col);

        df.Head(2).RowCount.Should().Be(2);
        df.Head(5).RowCount.Should().Be(3, "Head(n) with n > RowCount should return RowCount rows");
        df.Head(0).RowCount.Should().Be(0);
    }

    [Fact]
    public void Property_TailRowCount()
    {
        var col = new Column<int>("A", new[] { 1, 2, 3 });
        var df = new DataFrame(col);

        df.Tail(2).RowCount.Should().Be(2);
        df.Tail(5).RowCount.Should().Be(3, "Tail(n) with n > RowCount should return RowCount rows");
        df.Tail(0).RowCount.Should().Be(0);
    }

    [Fact]
    public void Property_FilterAllTrue_SameRowCount()
    {
        var col = new Column<int>("A", new[] { 1, 2, 3 });
        var df = new DataFrame(col);

        var allTrue = new bool[] { true, true, true };
        df.Filter(allTrue).RowCount.Should().Be(3);
    }

    [Fact]
    public void Property_FilterAllFalse_ZeroRows()
    {
        var col = new Column<int>("A", new[] { 1, 2, 3 });
        var df = new DataFrame(col);

        var allFalse = new bool[] { false, false, false };
        df.Filter(allFalse).RowCount.Should().Be(0);
    }

    [Fact]
    public void Property_DropDuplicates_RowCountLessOrEqual()
    {
        var col = new Column<int>("A", new[] { 1, 2, 2, 3, 3, 3 });
        var df = new DataFrame(col);

        df.DropDuplicates().RowCount.Should().BeLessThanOrEqualTo(df.RowCount);
        df.DropDuplicates().RowCount.Should().Be(3);
    }

    [Fact]
    public void Property_ConcatRowCount()
    {
        var df1 = new DataFrame(new Column<int>("A", new[] { 1, 2 }));
        var df2 = new DataFrame(new Column<int>("A", new[] { 3, 4, 5 }));

        var concat = ConcatExtensions.Concat(df1, df2);
        concat.RowCount.Should().Be(5, "Concat row count should equal sum of input row counts");
    }

    [Fact]
    public void Property_SampleRowCount()
    {
        var col = new Column<int>("A", new[] { 1, 2, 3, 4, 5 });
        var df = new DataFrame(col);

        df.Sample(3, seed: 42).RowCount.Should().Be(3);
        df.Sample(10, seed: 42).RowCount.Should().Be(5, "Sample(n) with n > RowCount should cap at RowCount");
    }

    // ===============================================================
    // Property verification: correlation symmetry
    // ===============================================================

    [Fact]
    public void Property_CorrMatrixSymmetric()
    {
        var df = new DataFrame(
            new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }),
            new Column<double>("B", new[] { 5.0, 4.0, 3.0, 2.0, 1.0 }),
            new Column<double>("C", new[] { 1.0, 3.0, 2.0, 5.0, 4.0 })
        );

        var corr = df.Corr();

        // Check symmetry: corr[i,j] == corr[j,i]
        var names = new[] { "A", "B", "C" };
        for (int i = 0; i < names.Length; i++)
        {
            for (int j = i + 1; j < names.Length; j++)
            {
                var colI = (Column<double>)corr[names[i]];
                var colJ = (Column<double>)corr[names[j]];

                double corrIJ = colI.Buffer.Span[j];
                double corrJI = colJ.Buffer.Span[i];

                corrIJ.Should().BeApproximately(corrJI, 1e-10,
                    $"Corr[{names[i]},{names[j]}] should equal Corr[{names[j]},{names[i]}]");
            }
        }
    }

    [Fact]
    public void Property_CorrDiagonalIsOne()
    {
        var df = new DataFrame(
            new Column<double>("A", new[] { 1.0, 2.0, 3.0, 4.0, 5.0 }),
            new Column<double>("B", new[] { 5.0, 4.0, 3.0, 2.0, 1.0 })
        );

        var corr = df.Corr();

        var colA = (Column<double>)corr["A"];
        var colB = (Column<double>)corr["B"];

        colA.Buffer.Span[0].Should().BeApproximately(1.0, 1e-10, "Corr[A,A] should be 1.0");
        colB.Buffer.Span[1].Should().BeApproximately(1.0, 1e-10, "Corr[B,B] should be 1.0");
    }

    // ===============================================================
    // Property verification: rank properties
    // ===============================================================

    [Fact]
    public void Property_RankSumEqualsTriangularNumber()
    {
        var col = new Column<double>("A", new[] { 30.0, 10.0, 20.0, 50.0, 40.0 });

        var ranked = col.Rank();

        // All non-null → n values ranked 1..n, sum = n*(n+1)/2
        int n = col.Length;
        double expectedSum = n * (n + 1) / 2.0;
        double actualSum = 0;
        for (int i = 0; i < n; i++)
            actualSum += ranked[i]!.Value;

        actualSum.Should().BeApproximately(expectedSum, 1e-10,
            "Sum of ranks should equal n*(n+1)/2");
    }

    [Fact]
    public void Property_RankValuesInRange()
    {
        var col = new Column<double>("A", new[] { 30.0, 10.0, 20.0, 50.0, 40.0 });

        var ranked = col.Rank();
        int n = col.Length;

        for (int i = 0; i < n; i++)
        {
            ranked[i]!.Value.Should().BeGreaterThanOrEqualTo(1.0);
            ranked[i]!.Value.Should().BeLessThanOrEqualTo(n);
        }
    }

    // ===============================================================
    // Property verification: round-trip
    // ===============================================================

    [Fact]
    public void Property_RenameRoundTrip()
    {
        var df = new DataFrame(new Column<int>("A", new[] { 1, 2, 3 }));

        var roundTrip = df.RenameColumn("A", "B").RenameColumn("B", "A");

        roundTrip.ColumnNames.Should().Contain("A");
        var c = (Column<int>)roundTrip["A"];
        c[0].Should().Be(1);
        c[1].Should().Be(2);
        c[2].Should().Be(3);
    }

    [Fact]
    public void Property_FillNaThenNullCountIsZero()
    {
        var col = Column<double>.FromNullable("A", new double?[] { 1.0, null, 3.0, null, 5.0 });

        var filled = col.FillNa(0.0);

        filled.NullCount.Should().Be(0, "After FillNa, no bitmask nulls should remain");
        // Also check no NaN values
        for (int i = 0; i < filled.Length; i++)
            double.IsNaN(filled[i]!.Value).Should().BeFalse($"FillNa(0.0) should not produce NaN at index {i}");
    }

    [Fact]
    public void Property_DropNaThenNoNulls()
    {
        var col = Column<double>.FromNullable("A", new double?[] { 1.0, null, 3.0, double.NaN, 5.0 });
        var df = new DataFrame(col);

        var dropped = df.DropNa();

        dropped.RowCount.Should().Be(3, "Should drop rows with null and NaN");
        var result = (Column<double>)dropped["A"];
        for (int i = 0; i < dropped.RowCount; i++)
        {
            result.IsNull(i).Should().BeFalse($"No null at index {i} after DropNa");
            double.IsNaN(result.Buffer.Span[i]).Should().BeFalse($"No NaN at index {i} after DropNa");
        }
    }

    // ===============================================================
    // Property verification: GroupBy count sum equals original rows
    // (using non-null values)
    // ===============================================================

    [Fact]
    public void Property_GroupByCountSum_EqualsOriginalRowCount_WhenNoNulls()
    {
        var key = new StringColumn("Key", new[] { "A", "A", "B", "B", "B", "C" });
        var val = new Column<int>("Val", new[] { 1, 2, 3, 4, 5, 6 });
        var df = new DataFrame(key, val);

        var counts = df.GroupBy("Key").Count();
        var countCol = (Column<int>)counts["Val"];

        int totalCount = 0;
        for (int i = 0; i < countCol.Length; i++)
            totalCount += countCol[i]!.Value;

        totalCount.Should().Be(df.RowCount,
            "Sum of group counts should equal original row count when no nulls");
    }

    // ===============================================================
    // Property verification: ordering after sort
    // ===============================================================

    [Fact]
    public void Property_SortAscending_IsOrdered()
    {
        var col = new Column<double>("A", new[] { 5.0, 3.0, 1.0, 4.0, 2.0 });
        var df = new DataFrame(col);

        var sorted = df.Sort("A");
        var sc = (Column<double>)sorted["A"];

        for (int i = 1; i < sc.Length; i++)
            sc[i]!.Value.Should().BeGreaterThanOrEqualTo(sc[i - 1]!.Value,
                $"After ascending sort, element {i} should be >= element {i-1}");
    }

    [Fact]
    public void Property_SortDescending_IsOrdered()
    {
        var col = new Column<double>("A", new[] { 5.0, 3.0, 1.0, 4.0, 2.0 });
        var df = new DataFrame(col);

        var sorted = df.Sort("A", ascending: false);
        var sc = (Column<double>)sorted["A"];

        for (int i = 1; i < sc.Length; i++)
            sc[i]!.Value.Should().BeLessThanOrEqualTo(sc[i - 1]!.Value,
                $"After descending sort, element {i} should be <= element {i-1}");
    }

    // ===============================================================
    // Property verification: statistical identities with nulls
    // ===============================================================

    [Fact]
    public void Property_MeanEqualsSumOverCount_WithNulls()
    {
        var col = Column<double>.FromNullable("A", new double?[] { 1.0, null, 3.0, null, 5.0 });

        var mean = col.Mean();
        var sum = col.Sum();
        var count = col.ValidCount();

        count.Should().Be(3);
        mean!.Value.Should().BeApproximately(sum!.Value / count, 1e-10);
    }

    [Fact]
    public void Property_VarEqualsStdSquared_WithNulls()
    {
        var col = Column<double>.FromNullable("A", new double?[] { 1.0, null, 3.0, null, 5.0 });

        var variance = col.Var();
        var std = col.Std();

        variance!.Value.Should().BeApproximately(std!.Value * std!.Value, 1e-10);
    }
}
