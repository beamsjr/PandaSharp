using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.Text.Embeddings;
using Cortex.Text.Preprocessing;
using Cortex.Text.StringDistance;
using Cortex.Text.Tokenizers;
using Xunit;

namespace Cortex.Text.Tests.EdgeCases;

public class WhitespaceTokenizerEdgeCaseTests
{
    [Fact]
    public void Encode_Empty_String_Returns_Empty_Tokens()
    {
        var tok = new WhitespaceTokenizer();
        tok.Train(["hello world"]);

        var result = tok.Encode("");

        result.TokenIds.Should().BeEmpty();
        result.AttentionMask.Should().BeEmpty();
    }

    [Fact]
    public void Encode_Only_Spaces_Returns_Empty_Tokens()
    {
        var tok = new WhitespaceTokenizer();
        tok.Train(["hello world"]);

        var result = tok.Encode("     ");

        result.TokenIds.Should().BeEmpty();
        result.AttentionMask.Should().BeEmpty();
    }

    [Fact]
    public void Train_With_Null_Entry_In_Corpus_Throws_ArgumentNullException()
    {
        // BUG FIX: null entry in corpus previously caused NullReferenceException,
        // now throws ArgumentNullException with clear message
        var tok = new WhitespaceTokenizer(lowercase: true);

        var act = () => tok.Train(["hello", null!, "world"]);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Encode_Unicode_Emoji_And_CJK()
    {
        var tok = new WhitespaceTokenizer();
        // Train with emoji as a separate token
        tok.Train(["hello", "\ud83d\ude00", "\u4e16\u754c"]);

        // Encode text with emoji and CJK - all are known tokens
        var result = tok.Encode("hello \ud83d\ude00 \u4e16\u754c");

        result.TokenIds.Should().HaveCount(3);
        // All tokens should be recognized (not UNK)
        result.TokenIds.Should().NotContain(0, "all tokens were in training corpus");
    }
}

public class BPETokenizerEdgeCaseTests
{
    [Fact]
    public void Train_Empty_Corpus_Then_Encode_Throws()
    {
        var tok = new BPETokenizer();
        tok.Train([], vocabSize: 100);

        // After training on empty corpus, vocab has only [UNK] => Encode should throw
        var act = () => tok.Encode("hello");

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Train_Single_Char_Vocabulary()
    {
        var tok = new BPETokenizer();
        tok.Train(["a"], vocabSize: 5);

        var result = tok.Encode("a");

        result.TokenIds.Should().NotBeEmpty();
    }

    [Fact]
    public void Encode_Unknown_Chars_Returns_UNK()
    {
        var tok = new BPETokenizer();
        tok.Train(["abc"], vocabSize: 10);

        // 'z' was never in training corpus
        var result = tok.Encode("z");

        // At minimum the unknown char should be mapped to UNK id (0)
        result.TokenIds.Should().Contain(0);
    }
}

public class LevenshteinEdgeCaseTests
{
    [Fact]
    public void LongStrings_1000_Plus_Chars()
    {
        var a = new string('a', 1000) + "b";
        var b = new string('a', 1000) + "c";

        var distance = LevenshteinDistance.Compute(a, b);

        distance.Should().Be(1);
    }

    [Fact]
    public void Empty_Vs_Empty_Returns_Zero()
    {
        LevenshteinDistance.Compute("", "").Should().Be(0);
    }

    [Fact]
    public void Identical_Strings_Returns_Zero()
    {
        LevenshteinDistance.Compute("test", "test").Should().Be(0);
    }

    [Fact]
    public void Single_Char_Strings()
    {
        LevenshteinDistance.Compute("a", "b").Should().Be(1);
        LevenshteinDistance.Compute("a", "a").Should().Be(0);
    }

    [Fact]
    public void Empty_Vs_NonEmpty_Returns_Length()
    {
        LevenshteinDistance.Compute("", "abc").Should().Be(3);
        LevenshteinDistance.Compute("abc", "").Should().Be(3);
    }
}

public class JaroWinklerEdgeCaseTests
{
    [Fact]
    public void Both_Empty_Returns_One()
    {
        JaroWinklerSimilarity.Compute("", "").Should().Be(1.0);
    }

    [Fact]
    public void One_Empty_Returns_Zero()
    {
        JaroWinklerSimilarity.Compute("", "abc").Should().Be(0.0);
        JaroWinklerSimilarity.Compute("abc", "").Should().Be(0.0);
    }

    [Fact]
    public void Single_Char_Each_Same()
    {
        JaroWinklerSimilarity.Compute("a", "a").Should().Be(1.0);
    }

    [Fact]
    public void Single_Char_Each_Different()
    {
        JaroWinklerSimilarity.Compute("a", "b").Should().Be(0.0);
    }

    [Fact]
    public void Identical_Strings()
    {
        JaroWinklerSimilarity.Compute("hello", "hello").Should().Be(1.0);
    }

    [Fact]
    public void Completely_Different_Strings()
    {
        var sim = JaroWinklerSimilarity.Compute("abc", "xyz");
        sim.Should().Be(0.0);
    }
}

public class StemmerEdgeCaseTests
{
    [Fact]
    public void Stem_Empty_String_Returns_Empty()
    {
        var stemmer = new Stemmer();
        stemmer.Stem("").Should().Be("");
    }

    [Fact]
    public void Stem_Single_Letter_Returns_Same()
    {
        var stemmer = new Stemmer();
        stemmer.Stem("a").Should().Be("a");
        stemmer.Stem("z").Should().Be("z");
    }

    [Fact]
    public void Stem_NonEnglish_Characters_Does_Not_Throw()
    {
        var stemmer = new Stemmer();

        var act = () => stemmer.Stem("cafe");

        act.Should().NotThrow();
    }

    [Fact]
    public void Stem_Already_Stemmed_Word_Is_Stable()
    {
        var stemmer = new Stemmer();
        var once = stemmer.Stem("running");
        var twice = stemmer.Stem(once);

        twice.Should().Be(once, "stemming an already-stemmed word should be idempotent");
    }
}

public class StopWordRemoverEdgeCaseTests
{
    [Fact]
    public void RemoveStopWords_Text_With_Only_StopWords_Returns_Empty()
    {
        var remover = new StopWordRemover("english");
        var result = remover.RemoveStopWords("the a an is are was");

        result.Should().Be("");
    }

    [Fact]
    public void RemoveStopWords_Empty_Text_Returns_Empty()
    {
        var remover = new StopWordRemover("english");
        var result = remover.RemoveStopWords("");

        result.Should().Be("");
    }

    [Fact]
    public void Unsupported_Language_Throws()
    {
        var act = () => new StopWordRemover("klingon");

        act.Should().Throw<ArgumentException>();
    }
}

public class CosineSimilarityEdgeCaseTests
{
    [Fact]
    public void Compute_Zero_Vector_Returns_Zero()
    {
        var zero = new double[] { 0.0, 0.0, 0.0 };
        var other = new double[] { 1.0, 2.0, 3.0 };

        CosineSimilarity.Compute(zero, other).Should().Be(0.0);
    }

    [Fact]
    public void Compute_Unit_Vectors_Same_Direction()
    {
        var a = new double[] { 1.0, 0.0 };
        var b = new double[] { 1.0, 0.0 };

        CosineSimilarity.Compute(a, b).Should().BeApproximately(1.0, 1e-10);
    }

    [Fact]
    public void Compute_Single_Element_Vectors()
    {
        CosineSimilarity.Compute([3.0], [3.0]).Should().BeApproximately(1.0, 1e-10);
        CosineSimilarity.Compute([1.0], [-1.0]).Should().BeApproximately(-1.0, 1e-10);
    }

    [Fact]
    public void Compute_Mismatched_Lengths_Throws()
    {
        var a = new double[] { 1.0, 2.0 };
        var b = new double[] { 1.0, 2.0, 3.0 };

        var act = () => CosineSimilarity.Compute(a, b);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Compute_Both_Zero_Vectors_Returns_Zero()
    {
        var a = new double[] { 0.0, 0.0 };
        var b = new double[] { 0.0, 0.0 };

        // Both zero vectors -- denom = 0, returns 0.0
        var result = CosineSimilarity.Compute(a, b);
        double.IsNaN(result).Should().BeFalse("should return 0, not NaN");
    }

    [Fact]
    public void Compute_Empty_Vectors_Returns_Zero()
    {
        var a = Array.Empty<double>();
        var b = Array.Empty<double>();

        // Empty vectors have denom = 0, returns 0.0
        var result = CosineSimilarity.Compute(a, b);
        double.IsNaN(result).Should().BeFalse("should return 0, not NaN for empty vectors");
    }
}
