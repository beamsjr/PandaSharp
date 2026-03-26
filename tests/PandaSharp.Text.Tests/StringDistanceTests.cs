using Xunit;
using FluentAssertions;
using PandaSharp.Text.StringDistance;

namespace PandaSharp.Text.Tests;

public class LevenshteinDistanceTests
{
    [Theory]
    [InlineData("kitten", "sitting", 3)]
    [InlineData("saturday", "sunday", 3)]
    [InlineData("abc", "abc", 0)]
    [InlineData("", "", 0)]
    [InlineData("abc", "", 3)]
    [InlineData("", "xyz", 3)]
    [InlineData("a", "b", 1)]
    [InlineData("flaw", "lawn", 2)]
    public void Compute_KnownPairs_ReturnsExpectedDistance(string a, string b, int expected)
    {
        LevenshteinDistance.Compute(a, b).Should().Be(expected);
    }

    [Fact]
    public void Compute_IsSymmetric()
    {
        LevenshteinDistance.Compute("kitten", "sitting")
            .Should().Be(LevenshteinDistance.Compute("sitting", "kitten"));
    }

    [Fact]
    public void Compute_NullInput_ThrowsArgumentNullException()
    {
        var act = () => LevenshteinDistance.Compute(null!, "test");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Compute_Unicode_HandlesCorrectly()
    {
        // Single character substitution in Unicode
        LevenshteinDistance.Compute("cafe", "caf\u00e9").Should().Be(1);
    }

    [Fact]
    public void BatchCompute_1000Strings_ReturnsCorrectResults()
    {
        var strings = new string?[1000];
        for (int i = 0; i < 1000; i++)
            strings[i] = $"string{i}";

        var results = LevenshteinDistance.BatchCompute(strings, "string0");

        results.Should().HaveCount(1000);
        results[0].Should().Be(0); // "string0" vs "string0"

        // All distances should be non-negative
        results.Should().OnlyContain(d => d >= 0);
    }

    [Fact]
    public void BatchCompute_NullEntries_TreatedAsEmpty()
    {
        var strings = new string?[] { null, "abc", null };
        var results = LevenshteinDistance.BatchCompute(strings, "abc");

        results[0].Should().Be(3); // "" vs "abc"
        results[1].Should().Be(0); // "abc" vs "abc"
        results[2].Should().Be(3); // "" vs "abc"
    }
}

public class JaroWinklerSimilarityTests
{
    [Fact]
    public void Compute_IdenticalStrings_ReturnsOne()
    {
        JaroWinklerSimilarity.Compute("hello", "hello").Should().Be(1.0);
    }

    [Fact]
    public void Compute_BothEmpty_ReturnsOne()
    {
        JaroWinklerSimilarity.Compute("", "").Should().Be(1.0);
    }

    [Fact]
    public void Compute_OneEmpty_ReturnsZero()
    {
        JaroWinklerSimilarity.Compute("abc", "").Should().Be(0.0);
        JaroWinklerSimilarity.Compute("", "abc").Should().Be(0.0);
    }

    [Fact]
    public void Compute_CompletelyDifferent_NearZero()
    {
        var result = JaroWinklerSimilarity.Compute("abc", "xyz");
        result.Should().BeApproximately(0.0, 0.01);
    }

    [Fact]
    public void Compute_KnownPair_Martha_Marhta()
    {
        // Classic Jaro-Winkler example: MARTHA vs MARHTA
        // Jaro = 0.944..., common prefix "MAR" (len 3)
        // JW = 0.944 + 3 * 0.1 * (1 - 0.944) = 0.944 + 0.0167 = 0.961
        var result = JaroWinklerSimilarity.Compute("MARTHA", "MARHTA");
        result.Should().BeApproximately(0.961, 0.01);
    }

    [Fact]
    public void Compute_WinklerPrefixBonus_IncreasesScore()
    {
        // Two pairs with same Jaro score but different common prefixes
        // "MARTHA"/"MARHTA" has prefix "MAR" (3) -> gets Winkler bonus
        double withPrefix = JaroWinklerSimilarity.Compute("MARTHA", "MARHTA");

        // Jaro score (no prefix bonus) should be lower
        // Use prefixScale=0 to effectively disable Winkler
        double withoutBonus = JaroWinklerSimilarity.Compute("MARTHA", "MARHTA", prefixScale: 0.0);

        withPrefix.Should().BeGreaterThan(withoutBonus);
    }

    [Fact]
    public void Compute_NullInput_ThrowsArgumentNullException()
    {
        var act = () => JaroWinklerSimilarity.Compute(null!, "test");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Compute_PrefixScaleTooLarge_ThrowsArgumentOutOfRange()
    {
        var act = () => JaroWinklerSimilarity.Compute("a", "b", prefixScale: 0.3);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Compute_Unicode_HandlesCorrectly()
    {
        // Identical Unicode strings
        JaroWinklerSimilarity.Compute("\u00e9\u00e8\u00ea", "\u00e9\u00e8\u00ea").Should().Be(1.0);
    }

    [Fact]
    public void BatchCompute_ReturnsCorrectResults()
    {
        var strings = new string?[] { "hello", "hallo", null, "world" };
        var results = JaroWinklerSimilarity.BatchCompute(strings, "hello");

        results.Should().HaveCount(4);
        results[0].Should().Be(1.0); // identical
        results[1].Should().BeGreaterThan(0.8); // similar
        results[2].Should().Be(0.0); // null (empty) vs "hello"
        results[3].Should().BeLessThan(0.5); // quite different
    }

    [Fact]
    public void Compute_ResultAlwaysBetweenZeroAndOne()
    {
        var pairs = new[] {
            ("abc", "def"), ("hello", "world"), ("test", "testing"),
            ("a", "aaaa"), ("short", "a very long string indeed")
        };

        foreach (var (a, b) in pairs)
        {
            var result = JaroWinklerSimilarity.Compute(a, b);
            result.Should().BeGreaterThanOrEqualTo(0.0);
            result.Should().BeLessThanOrEqualTo(1.0);
        }
    }
}
