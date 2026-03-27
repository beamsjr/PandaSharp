using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Xunit;

namespace Cortex.Tests.Unit.GroupBy;

/// <summary>
/// Round 5 bug hunting: cross-module data flow in GroupBy → Aggregation → further operations.
/// </summary>
public class CrossModuleRound5Tests
{
    /// <summary>
    /// GroupBy.Sum() on int columns produces Column&lt;double&gt;.
    /// Sorting the result by the sum column should work correctly.
    /// </summary>
    [Fact]
    public void GroupBy_Sum_ThenSort_ProducesCorrectOrder()
    {
        var df = new DataFrame(
            new StringColumn("Category", new string?[] { "B", "A", "B", "A", "C" }),
            new Column<int>("Value", new int[] { 10, 20, 30, 40, 5 })
        );

        var sumDf = df.GroupBy("Category").Sum();
        var sorted = sumDf.Sort("Value", ascending: true);

        var cats = sorted.GetStringColumn("Category");
        var vals = sorted.GetColumn<double>("Value");

        // C=5, A=60, B=40 → sorted ascending: C(5), B(40), A(60)
        cats[0].Should().Be("C");
        cats[1].Should().Be("B");
        cats[2].Should().Be("A");

        vals[0].Should().Be(5.0);
        vals[1].Should().Be(40.0);
        vals[2].Should().Be(60.0);
    }

    /// <summary>
    /// GroupBy.Agg() then join with another DataFrame on the key column.
    /// This tests that the key column type is preserved through aggregation.
    /// </summary>
    [Fact]
    public void GroupBy_Agg_ThenJoin_PreservesKeyType()
    {
        var df = new DataFrame(
            new StringColumn("Category", new string?[] { "A", "B", "A", "B" }),
            new Column<double>("Sales", new double[] { 100, 200, 150, 250 })
        );

        var aggDf = df.GroupBy("Category").Agg(("Sales", AggFunc.Sum));
        // aggDf has "Category" (StringColumn) and "Sales_sum" (Column<double>)

        var lookup = new DataFrame(
            new StringColumn("Category", new string?[] { "A", "B", "C" }),
            new StringColumn("Region", new string?[] { "North", "South", "East" })
        );

        // Join on Category
        var joined = aggDf.Join(lookup, "Category", JoinType.Inner);

        joined.RowCount.Should().Be(2);
        joined.ColumnNames.Should().Contain("Sales_sum");
        joined.ColumnNames.Should().Contain("Region");

        // Verify values survive the pipeline
        for (int i = 0; i < joined.RowCount; i++)
        {
            var cat = joined.GetStringColumn("Category")[i];
            var region = joined.GetStringColumn("Region")[i];
            var sales = joined.GetColumn<double>("Sales_sum")[i];

            if (cat == "A")
            {
                region.Should().Be("North");
                sales.Should().Be(250.0);
            }
            else if (cat == "B")
            {
                region.Should().Be("South");
                sales.Should().Be(450.0);
            }
        }
    }

    /// <summary>
    /// Filter → GroupBy → Sum chain should work correctly.
    /// This tests chaining operations where the data flow passes through
    /// Filter (which creates sliced columns) then GroupBy.
    /// </summary>
    [Fact]
    public void Filter_GroupBy_Sum_ChainedCorrectly()
    {
        var df = new DataFrame(
            new StringColumn("Category", new string?[] { "A", "B", "A", "B", "C" }),
            new Column<int>("Value", new int[] { 10, 20, 30, 40, 50 })
        );

        // Filter out category C, then GroupBy + Sum
        var result = df.WhereString("Category", s => s != "C")
            .GroupBy("Category")
            .Sum();

        result.RowCount.Should().Be(2);

        // Find A and B in results (order not guaranteed)
        var cats = Enumerable.Range(0, result.RowCount)
            .ToDictionary(i => result.GetStringColumn("Category")[i]!,
                          i => result.GetColumn<double>("Value")[i]);

        cats["A"].Should().Be(40.0); // 10 + 30
        cats["B"].Should().Be(60.0); // 20 + 40
    }

    /// <summary>
    /// GroupBy on an empty DataFrame (result of filtering all rows out).
    /// Should produce an empty result, not crash.
    /// </summary>
    [Fact]
    public void GroupBy_OnEmptyDataFrame_ProducesEmptyResult()
    {
        var df = new DataFrame(
            new StringColumn("Category", new string?[] { "A", "B" }),
            new Column<int>("Value", new int[] { 10, 20 })
        );

        var empty = df.WhereString("Category", s => s == "Z");
        empty.RowCount.Should().Be(0);

        var grouped = empty.GroupBy("Category");
        grouped.GroupCount.Should().Be(0);

        var sum = grouped.Sum();
        sum.RowCount.Should().Be(0);
    }

    /// <summary>
    /// CSV round-trip: GroupBy → Sum → Write CSV → Read CSV → verify values.
    /// Tests that data survives serialization/deserialization through the full pipeline.
    /// </summary>
    [Fact]
    public void CsvRoundTrip_GroupBySum_PreservesValues()
    {
        var df = new DataFrame(
            new StringColumn("Category", new string?[] { "A", "B", "A", "B" }),
            new Column<double>("Amount", new double[] { 10.5, 20.3, 30.7, 40.1 })
        );

        var sumDf = df.GroupBy("Category").Sum();

        // Write and read back
        var tmpPath = Path.GetTempFileName() + ".csv";
        try
        {
            CsvWriter.Write(sumDf, tmpPath);
            var readBack = CsvReader.Read(tmpPath);

            readBack.RowCount.Should().Be(sumDf.RowCount);

            // Values should be preserved (may be re-inferred as double)
            for (int i = 0; i < readBack.RowCount; i++)
            {
                var cat = readBack.GetStringColumn("Category")[i];
                double amount = TypeHelpers.GetDouble(readBack["Amount"], i);
                double expected = TypeHelpers.GetDouble(sumDf["Amount"],
                    Enumerable.Range(0, sumDf.RowCount)
                        .First(j => sumDf.GetStringColumn("Category")[j] == cat));
                amount.Should().BeApproximately(expected, 0.01);
            }
        }
        finally
        {
            File.Delete(tmpPath);
        }
    }
}
