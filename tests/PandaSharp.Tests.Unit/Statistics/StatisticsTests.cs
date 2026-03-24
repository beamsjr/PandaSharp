using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.Statistics;

public class StatisticsTests
{
    private static DataFrame CreateSampleDf() => new(
        new Column<int>("Age", [25, 30, 35, 28, 42]),
        new Column<double>("Salary", [50_000, 62_000, 75_000, 58_000, 91_000]),
        new StringColumn("Name", ["Alice", "Bob", "Charlie", "Diana", "Eve"])
    );

    [Fact]
    public void Describe_ReturnsStatsForNumericColumns()
    {
        var df = CreateSampleDf();
        var desc = df.Describe();

        desc.ColumnNames.Should().Contain("stat");
        desc.ColumnNames.Should().Contain("Age");
        desc.ColumnNames.Should().Contain("Salary");
        desc.ColumnNames.Should().NotContain("Name"); // non-numeric excluded

        desc.RowCount.Should().Be(8); // count, mean, std, min, 25%, 50%, 75%, max
        desc.GetStringColumn("stat")[0].Should().Be("count");
        desc.GetColumn<double>("Age")[0].Should().Be(5); // count
    }

    [Fact]
    public void Info_ReturnsSchemaInfo()
    {
        var df = CreateSampleDf();
        var info = df.Info();

        info.RowCount.Should().Be(5);
        info.ColumnCount.Should().Be(3);
        info.Columns.Should().HaveCount(3);

        var ageInfo = info.Columns.First(c => c.Name == "Age");
        ageInfo.DataType.Should().Be("int32");
        ageInfo.NonNullCount.Should().Be(5);
        ageInfo.NullCount.Should().Be(0);
    }

    [Fact]
    public void Info_ToString_FormatsNicely()
    {
        var df = CreateSampleDf();
        var output = df.Info().ToString();

        output.Should().Contain("Age");
        output.Should().Contain("int32");
        output.Should().Contain("5 rows x 3 columns");
    }

    [Fact]
    public void ValueCounts_ReturnsSortedCounts()
    {
        var col = new StringColumn("color", ["red", "blue", "red", "green", "red", "blue"]);
        var vc = col.ValueCounts();

        vc.RowCount.Should().Be(3);
        vc.GetStringColumn("color")[0].Should().Be("red"); // most frequent first
        vc.GetColumn<int>("count")[0].Should().Be(3);
    }

    [Fact]
    public void NUnique_CountsDistinctNonNull()
    {
        var col = new StringColumn("x", ["a", "b", "a", null, "c"]);
        col.NUnique().Should().Be(3);
    }

    [Fact]
    public void Rank_Average_HandlesTies()
    {
        var col = new Column<int>("x", [3, 1, 4, 1, 5]);
        var ranked = col.Rank(RankMethod.Average);

        ranked[0].Should().Be(3); // 3 is rank 3
        ranked[1].Should().Be(1.5); // two 1s share ranks 1-2
        ranked[2].Should().Be(4);
        ranked[3].Should().Be(1.5);
        ranked[4].Should().Be(5);
    }

    [Fact]
    public void Rank_Dense_NoGaps()
    {
        var col = new Column<int>("x", [3, 1, 4, 1, 5]);
        var ranked = col.Rank(RankMethod.Dense);

        ranked[0].Should().Be(2);
        ranked[1].Should().Be(1);
        ranked[2].Should().Be(3);
        ranked[3].Should().Be(1);
        ranked[4].Should().Be(4);
    }

    [Fact]
    public void Rank_Min_UsesLowestRankForTies()
    {
        var col = new Column<int>("x", [3, 1, 4, 1, 5]);
        var ranked = col.Rank(RankMethod.Min);

        ranked[1].Should().Be(1);
        ranked[3].Should().Be(1);
    }

    [Fact]
    public void Corr_ReturnsCorrelationMatrix()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0, 4.0, 5.0]),
            new Column<double>("B", [2.0, 4.0, 6.0, 8.0, 10.0]) // perfectly correlated
        );

        var corr = df.Corr();
        corr.GetColumn<double>("A")[0].Should().BeApproximately(1.0, 0.001); // A-A
        corr.GetColumn<double>("B")[0].Should().BeApproximately(1.0, 0.001); // A-B
        corr.GetColumn<double>("A")[1].Should().BeApproximately(1.0, 0.001); // B-A
    }

    [Fact]
    public void Cov_ReturnsCovarianceMatrix()
    {
        var df = new DataFrame(
            new Column<double>("A", [1.0, 2.0, 3.0]),
            new Column<double>("B", [4.0, 5.0, 6.0])
        );

        var cov = df.Cov();
        cov.GetColumn<double>("A")[0].Should().Be(1.0); // Var(A) = 1
        cov.GetColumn<double>("B")[1].Should().Be(1.0); // Var(B) = 1
        cov.GetColumn<double>("B")[0].Should().Be(1.0); // Cov(A,B) = 1
    }
}
