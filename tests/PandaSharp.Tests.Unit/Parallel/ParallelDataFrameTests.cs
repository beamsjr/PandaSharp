using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ParallelOps;

namespace PandaSharp.Tests.Unit.Parallel;

public class ParallelDataFrameTests
{
    private DataFrame CreateLargeDF(int n = 10_000)
    {
        var ids = new int[n];
        var values = new double[n];
        var categories = new string?[n];
        var rng = new Random(42);
        var cats = new[] { "A", "B", "C", "D" };

        for (int i = 0; i < n; i++)
        {
            ids[i] = i;
            values[i] = rng.NextDouble() * 1000;
            categories[i] = cats[rng.Next(cats.Length)];
        }

        return new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values),
            new StringColumn("Category", categories)
        );
    }

    // ===== ParallelFilter =====

    [Fact]
    public void ParallelFilter_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = df.Filter(row => (double)row["Value"]! > 500.0);
        var parallel = df.ParallelFilter(row => (double)row["Value"]! > 500.0);

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    [Fact]
    public void ParallelFilter_EmptyResult()
    {
        var df = CreateLargeDF();
        var result = df.ParallelFilter(row => (double)row["Value"]! > 9999.0);
        result.RowCount.Should().Be(0);
    }

    [Fact]
    public void ParallelFilter_AllMatch()
    {
        var df = CreateLargeDF();
        var result = df.ParallelFilter(row => (double)row["Value"]! >= 0.0);
        result.RowCount.Should().Be(df.RowCount);
    }

    [Fact]
    public void ParallelFilter_SingleRow()
    {
        var df = new DataFrame(new Column<int>("x", [42]));
        var result = df.ParallelFilter(row => (int)row["x"]! > 0);
        result.RowCount.Should().Be(1);
    }

    [Fact]
    public void ParallelFilter_EmptyDataFrame()
    {
        var df = new DataFrame(new Column<int>("x", Array.Empty<int>()));
        var result = df.ParallelFilter(row => true);
        result.RowCount.Should().Be(0);
    }

    // ===== ParallelWhere =====

    [Fact]
    public void ParallelWhere_MatchesSequential()
    {
        var df = CreateLargeDF();
        var sequential = df.Where("Value", val => (double)val! > 500.0);
        var parallel = df.ParallelWhere("Value", val => (double)val! > 500.0);

        parallel.RowCount.Should().Be(sequential.RowCount);
    }

    // ===== ParallelApply =====

    [Fact]
    public void ParallelApply_NumericResult_MatchesSequential()
    {
        var df = CreateLargeDF(1000);
        var sequential = df.Apply<double>(row => (double)row["Value"]! * 2, "Doubled");
        var parallel = df.ParallelApply<double>(row => (double)row["Value"]! * 2, "Doubled");

        parallel.RowCount.Should().Be(sequential.RowCount);
        for (int i = 0; i < 10; i++)
            parallel.GetColumn<double>("Doubled")[i].Should().Be(sequential.GetColumn<double>("Doubled")[i]);
    }

    [Fact]
    public void ParallelApply_StringResult()
    {
        var df = CreateLargeDF(1000);
        var result = df.ParallelApply(
            row => row["Category"]?.ToString()?.ToUpper(),
            "UpperCat");

        result.ColumnNames.Should().Contain("UpperCat");
        result.GetStringColumn("UpperCat")[0].Should().BeOneOf("A", "B", "C", "D");
    }

    [Fact]
    public void ParallelApply_WithNulls_HandlesGracefully()
    {
        var df = new DataFrame(
            Column<int>.FromNullable("X", [1, null, 3])
        );

        var result = df.ParallelApply<int>(row => row.Get<int>("X")!.Value * 10, "Y");

        result.GetColumn<int>("Y")[0].Should().Be(10);
        result["Y"].IsNull(1).Should().BeTrue();
        result.GetColumn<int>("Y")[2].Should().Be(30);
    }

    // ===== Parallel Arithmetic =====

    [Fact]
    public void ParallelAdd_MatchesSequential()
    {
        var df = CreateLargeDF();
        var col = df.GetColumn<double>("Value");

        var sequential = col.Add(col);
        var parallel = col.ParallelAdd(col);

        for (int i = 0; i < Math.Min(100, col.Length); i++)
            parallel[i].Should().Be(sequential[i]);
    }

    [Fact]
    public void ParallelMultiply_ColumnColumn_MatchesSequential()
    {
        var df = CreateLargeDF();
        var col = df.GetColumn<double>("Value");

        var sequential = col.Multiply(col);
        var parallel = col.ParallelMultiply(col);

        for (int i = 0; i < Math.Min(100, col.Length); i++)
            parallel[i].Should().Be(sequential[i]);
    }

    [Fact]
    public void ParallelMultiply_Scalar_MatchesSequential()
    {
        var df = CreateLargeDF();
        var col = df.GetColumn<double>("Value");

        var sequential = col.Multiply(2.5);
        var parallel = col.ParallelMultiply(2.5);

        for (int i = 0; i < Math.Min(100, col.Length); i++)
            parallel[i].Should().Be(sequential[i]);
    }

    [Fact]
    public void ParallelAdd_WithNulls()
    {
        var col = Column<double>.FromNullable("x", [1.0, null, 3.0, null, 5.0]);
        var result = col.ParallelAdd(col);

        result[0].Should().Be(2.0);
        result[1].Should().BeNull();
        result[2].Should().Be(6.0);
    }

    // ===== ParallelSum =====

    [Fact]
    public void ParallelSum_MatchesSequential()
    {
        var df = CreateLargeDF();
        var col = df.GetColumn<double>("Value");

        var sequential = col.Sum()!.Value;
        var parallel = col.ParallelSum();

        parallel.Should().BeApproximately(sequential, 0.001);
    }

    [Fact]
    public void ParallelSum_Empty_ReturnsZero()
    {
        var col = new Column<double>("x", Array.Empty<double>());
        col.ParallelSum().Should().Be(0);
    }

    // ===== Degree of parallelism =====

    [Fact]
    public void ParallelFilter_WithExplicitThreads_Works()
    {
        var df = CreateLargeDF();
        var result = df.ParallelFilter(row => (double)row["Value"]! > 500.0, maxDegreeOfParallelism: 2);
        result.RowCount.Should().BeGreaterThan(0);
    }
}
