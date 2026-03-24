using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.Statistics;

namespace PandaSharp.Tests.Unit.Statistics;

public class ProfileTests
{
    private DataFrame CreateTestDF()
    {
        var rng = new Random(42);
        int n = 100;
        var ids = new int[n];
        var values = new double[n];
        var categories = new string?[n];
        var cats = new[] { "A", "B", "C" };
        for (int i = 0; i < n; i++)
        {
            ids[i] = i;
            values[i] = rng.NextDouble() * 100;
            categories[i] = cats[rng.Next(cats.Length)];
        }
        return new DataFrame(
            new Column<int>("Id", ids),
            new Column<double>("Value", values),
            new StringColumn("Category", categories)
        );
    }

    [Fact]
    public void Profile_ReturnsCorrectOverview()
    {
        var df = CreateTestDF();
        var profile = df.Profile();

        profile.RowCount.Should().Be(100);
        profile.ColumnCount.Should().Be(3);
        profile.Columns.Should().HaveCount(3);
        profile.MemoryBytes.Should().BeGreaterThan(0);
    }

    [Fact]
    public void Profile_NumericColumn_HasStats()
    {
        var df = CreateTestDF();
        var profile = df.Profile();

        var valProfile = profile.Columns.First(c => c.Name == "Value");
        valProfile.IsNumeric.Should().BeTrue();
        valProfile.NonNullCount.Should().Be(100);
        valProfile.NullCount.Should().Be(0);
        valProfile.Mean.Should().BeInRange(40, 60); // ~50 for uniform [0,100)
        valProfile.Std.Should().BeGreaterThan(0);
        valProfile.Min.Should().BeGreaterThanOrEqualTo(0);
        valProfile.Max.Should().BeLessThan(100);
        valProfile.Q25.Should().BeLessThan(valProfile.Median);
        valProfile.Median.Should().BeLessThan(valProfile.Q75);
    }

    [Fact]
    public void Profile_IntColumn_HasStats()
    {
        var df = CreateTestDF();
        var profile = df.Profile();

        var idProfile = profile.Columns.First(c => c.Name == "Id");
        idProfile.IsNumeric.Should().BeTrue();
        idProfile.Min.Should().Be(0);
        idProfile.Max.Should().Be(99);
        idProfile.UniqueCount.Should().Be(100);
    }

    [Fact]
    public void Profile_StringColumn_HasStats()
    {
        var df = CreateTestDF();
        var profile = df.Profile();

        var catProfile = profile.Columns.First(c => c.Name == "Category");
        catProfile.IsString.Should().BeTrue();
        catProfile.UniqueCount.Should().Be(3);
        catProfile.MinLength.Should().Be(1);
        catProfile.MaxLength.Should().Be(1);
        catProfile.TopValues.Should().HaveCount(3);
    }

    [Fact]
    public void Profile_WithNulls_CountsCorrectly()
    {
        var df = new DataFrame(
            Column<double>.FromNullable("x", [1.0, null, 3.0, null, 5.0])
        );
        var profile = df.Profile();

        profile.TotalMissingValues.Should().Be(2);
        profile.MissingPercent.Should().BeApproximately(40, 0.1);
        profile.Columns[0].NullCount.Should().Be(2);
        profile.Columns[0].NullPercent.Should().BeApproximately(40, 0.1);
    }

    [Fact]
    public void Profile_DuplicateDetection()
    {
        var df = new DataFrame(
            new Column<int>("x", [1, 2, 1, 3, 2])
        );
        var profile = df.Profile();

        profile.DuplicateRowCount.Should().Be(2); // 1 and 2 each appear twice
    }

    [Fact]
    public void Profile_CorrelationMatrix()
    {
        var df = new DataFrame(
            new Column<double>("a", [1, 2, 3, 4, 5]),
            new Column<double>("b", [2, 4, 6, 8, 10]), // perfectly correlated with a
            new Column<double>("c", [5, 4, 3, 2, 1])   // negatively correlated with a
        );
        var profile = df.Profile();

        profile.CorrelationMatrix.Should().NotBeNull();
        profile.CorrelationColumns.Should().HaveCount(3);

        // a-b should be ~1.0, a-c should be ~-1.0
        profile.CorrelationMatrix![0, 1].Should().BeApproximately(1.0, 0.001);
        profile.CorrelationMatrix![0, 2].Should().BeApproximately(-1.0, 0.001);
    }

    [Fact]
    public void Profile_SkewnessKurtosis()
    {
        // Normal-ish distribution → skew ≈ 0, kurtosis ≈ 0
        var rng = new Random(42);
        var values = Enumerable.Range(0, 1000).Select(_ => rng.NextDouble()).ToArray();
        var df = new DataFrame(new Column<double>("x", values));
        var profile = df.Profile();

        var col = profile.Columns[0];
        col.Skew.Should().BeInRange(-0.5, 0.5);
        col.Kurtosis.Should().BeInRange(-1.5, 1.5); // uniform dist → excess kurtosis ≈ -1.2
    }

    [Fact]
    public void Profile_ZeroCount()
    {
        var df = new DataFrame(
            new Column<double>("x", [0, 1, 0, 2, 0])
        );
        var profile = df.Profile();

        profile.Columns[0].ZeroCount.Should().Be(3);
        profile.Columns[0].ZeroPercent.Should().BeApproximately(60, 0.1);
    }

    [Fact]
    public void Profile_EmptyDataFrame()
    {
        var df = new DataFrame(
            new Column<int>("x", Array.Empty<int>())
        );
        var profile = df.Profile();

        profile.RowCount.Should().Be(0);
        profile.TotalMissingValues.Should().Be(0);
    }

    [Fact]
    public void Profile_ToString_NotEmpty()
    {
        var df = CreateTestDF();
        var text = df.Profile().ToString();

        text.Should().Contain("DataFrame Profile");
        text.Should().Contain("Value");
        text.Should().Contain("Category");
        text.Should().Contain("Mean");
    }

    [Fact]
    public void Profile_ToHtml_ValidHtml()
    {
        var df = CreateTestDF();
        var html = df.Profile().ToHtml();

        html.Should().Contain("<!DOCTYPE html>");
        html.Should().Contain("DataFrame Profile Report");
        html.Should().Contain("Correlation Matrix");
        html.Should().Contain("</html>");
    }

    [Fact]
    public void Profile_TopValues_SortedByFrequency()
    {
        var df = new DataFrame(
            new StringColumn("x", ["a", "b", "a", "a", "b", "c"])
        );
        var profile = df.Profile();

        var top = profile.Columns[0].TopValues;
        top[0].Value.Should().Be("a");
        top[0].Count.Should().Be(3);
        top[1].Value.Should().Be("b");
        top[1].Count.Should().Be(2);
    }

    [Fact]
    public void Profile_StringEmptyAndLength()
    {
        var df = new DataFrame(
            new StringColumn("s", ["hello", "", "hi", ""])
        );
        var profile = df.Profile();

        var col = profile.Columns[0];
        col.EmptyStringCount.Should().Be(2);
        col.MinLength.Should().Be(0);
        col.MaxLength.Should().Be(5);
        col.MeanLength.Should().BeApproximately(1.75, 0.01); // (5+0+2+0)/4
    }
}
