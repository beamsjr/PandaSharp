using Xunit;
using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Tensors;
using PandaSharp.Text.Analytics;
using PandaSharp.Text.Embeddings;
using PandaSharp.Text.Preprocessing;
using PandaSharp.Text.Tokenizers;

namespace PandaSharp.Text.Tests;

public class WhitespaceTokenizerTests
{
    [Fact]
    public void Train_and_Encode_returns_valid_IDs()
    {
        var tokenizer = new WhitespaceTokenizer();
        tokenizer.Train(["hello world", "foo bar"]);

        var result = tokenizer.Encode("hello world");

        result.TokenIds.Should().HaveCount(2);
        result.TokenIds.Should().AllSatisfy(id => id.Should().BeGreaterThanOrEqualTo(0));
        result.AttentionMask.Should().AllBeEquivalentTo(1);
    }

    [Fact]
    public void Encode_Decode_round_trips()
    {
        var tokenizer = new WhitespaceTokenizer();
        tokenizer.Train(["hello world", "foo bar"]);

        var encoded = tokenizer.Encode("hello world");
        var decoded = tokenizer.Decode(encoded.TokenIds);

        decoded.Should().Be("hello world");
    }

    [Fact]
    public void Encode_null_throws_ArgumentNullException()
    {
        var tokenizer = new WhitespaceTokenizer();
        tokenizer.Train(["hello"]);

        var act = () => tokenizer.Encode(null!);

        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void Train_null_throws_ArgumentNullException()
    {
        var tokenizer = new WhitespaceTokenizer();

        var act = () => tokenizer.Train(null!);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class RegexTokenizerTests
{
    [Fact]
    public void Train_and_Encode_returns_word_level_tokens()
    {
        var tokenizer = new RegexTokenizer();
        tokenizer.Train(["hello world", "foo bar"]);

        var result = tokenizer.Encode("hello world");

        result.TokenIds.Should().HaveCount(2);
        result.TokenIds.Should().AllSatisfy(id => id.Should().BeGreaterThanOrEqualTo(0));
    }

    [Fact]
    public void Encode_Decode_round_trips()
    {
        var tokenizer = new RegexTokenizer();
        tokenizer.Train(["hello world"]);

        var encoded = tokenizer.Encode("hello world");
        var decoded = tokenizer.Decode(encoded.TokenIds);

        decoded.Should().Be("hello world");
    }

    [Fact]
    public void Encode_null_throws_ArgumentNullException()
    {
        var tokenizer = new RegexTokenizer();
        tokenizer.Train(["hello"]);

        var act = () => tokenizer.Encode(null!);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class BPETokenizerTests
{
    [Fact]
    public void Train_and_Encode_produces_IDs()
    {
        var tokenizer = new BPETokenizer();
        var corpus = new[] { "hello world", "hello there", "world hello", "foo bar baz" };
        tokenizer.Train(corpus, vocabSize: 50);

        var result = tokenizer.Encode("hello world");

        result.TokenIds.Should().NotBeEmpty();
        result.TokenIds.Should().AllSatisfy(id => id.Should().BeGreaterThanOrEqualTo(0));
    }

    [Fact]
    public void Encode_Decode_round_trips()
    {
        var tokenizer = new BPETokenizer();
        var corpus = new[] { "hello world", "hello there", "world hello" };
        tokenizer.Train(corpus, vocabSize: 50);

        var encoded = tokenizer.Encode("hello world");
        var decoded = tokenizer.Decode(encoded.TokenIds);

        // BPE decode may contain merged subwords; the decoded text
        // should contain the original words when end-of-word markers are stripped
        var cleaned = decoded.Replace("</w>", " ").Trim();
        var words = cleaned.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        words.Should().Contain("hello");
        words.Should().Contain("world");
    }

    [Fact]
    public void Train_null_throws_ArgumentNullException()
    {
        var tokenizer = new BPETokenizer();

        var act = () => tokenizer.Train(null!, 50);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class WordPieceTokenizerTests
{
    private static WordPieceTokenizer CreateSimpleTokenizer()
    {
        var tokenizer = new WordPieceTokenizer();
        tokenizer.AddToken("[PAD]", 0);
        tokenizer.AddToken("[UNK]", 1);
        tokenizer.AddToken("[CLS]", 2);
        tokenizer.AddToken("[SEP]", 3);
        tokenizer.AddToken("[MASK]", 4);
        tokenizer.AddToken("hello", 5);
        tokenizer.AddToken("world", 6);
        tokenizer.AddToken("##ing", 7);
        tokenizer.AddToken("run", 8);
        tokenizer.AddToken("##s", 9);
        return tokenizer;
    }

    [Fact]
    public void Encode_adds_CLS_and_SEP()
    {
        var tokenizer = CreateSimpleTokenizer();

        var result = tokenizer.Encode("hello world");

        // First token should be [CLS] (id=2), last should be [SEP] (id=3)
        result.TokenIds.First().Should().Be(2);
        result.TokenIds.Last().Should().Be(3);
    }

    [Fact]
    public void Unknown_words_produce_UNK()
    {
        var tokenizer = CreateSimpleTokenizer();

        var result = tokenizer.Encode("hello xyz");

        // xyz is not in vocab, should produce [UNK]
        result.TokenIds.Should().Contain(1); // [UNK] id
    }

    [Fact]
    public void Encode_null_throws_ArgumentNullException()
    {
        var tokenizer = CreateSimpleTokenizer();

        var act = () => tokenizer.Encode(null!);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class StopWordRemoverTests
{
    [Fact]
    public void RemoveStopWords_removes_the_is_on()
    {
        var remover = new StopWordRemover();

        var result = remover.RemoveStopWords("the cat is on the mat");

        result.Should().Contain("cat");
        result.Should().Contain("mat");
        result.Should().NotContainEquivalentOf("the");
        result.Should().NotContainEquivalentOf(" is ");
        // "on" should be removed
        var words = result.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        words.Should().NotContain("on");
        words.Should().NotContain("the");
        words.Should().NotContain("is");
    }
}

public class StemmerTests
{
    [Fact]
    public void Stem_running_returns_run()
    {
        var stemmer = new Stemmer();

        var result = stemmer.Stem("running");

        result.Should().Be("run");
    }

    [Fact]
    public void Stem_cats_returns_cat()
    {
        var stemmer = new Stemmer();

        var result = stemmer.Stem("cats");

        result.Should().Be("cat");
    }

    [Fact]
    public void Stem_null_throws_ArgumentNullException()
    {
        var stemmer = new Stemmer();

        var act = () => stemmer.Stem(null!);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class LemmatizerTests
{
    [Fact]
    public void Lemmatize_went_returns_go()
    {
        var lemmatizer = new Lemmatizer();

        var result = lemmatizer.Lemmatize("went");

        result.Should().Be("go");
    }

    [Fact]
    public void Lemmatize_running_returns_run()
    {
        var lemmatizer = new Lemmatizer();

        var result = lemmatizer.Lemmatize("running");

        result.Should().Be("run");
    }

    [Fact]
    public void Lemmatize_null_throws_ArgumentNullException()
    {
        var lemmatizer = new Lemmatizer();

        var act = () => lemmatizer.Lemmatize(null!);

        act.Should().Throw<ArgumentNullException>();
    }
}

public class NGramExtractorTests
{
    [Fact]
    public void Extract_bigrams_returns_expected()
    {
        var extractor = new NGramExtractor(n: 2);

        var result = extractor.Extract("a b c");

        result.Should().BeEquivalentTo(["a b", "b c"]);
    }
}

public class TextCleanerTests
{
    [Fact]
    public void Clean_strips_HTML_and_URLs()
    {
        var cleaner = new TextCleaner();

        var result = cleaner.Clean("<p>Hello!</p> http://example.com");

        result.Should().NotContain("<p>");
        result.Should().NotContain("</p>");
        result.Should().NotContain("http://example.com");
        result.Should().Contain("hello");
    }
}

public class SentenceSplitterTests
{
    [Fact]
    public void Split_returns_two_sentences()
    {
        var splitter = new SentenceSplitter();

        var result = splitter.Split("Hello. World!");

        result.Should().HaveCount(2);
    }
}

public class CosineSimilarityTests
{
    [Fact]
    public void Orthogonal_vectors_return_zero()
    {
        var result = CosineSimilarity.Compute([1, 0], [0, 1]);

        result.Should().BeApproximately(0.0, 1e-10);
    }

    [Fact]
    public void Identical_vectors_return_one()
    {
        var result = CosineSimilarity.Compute([1, 0], [1, 0]);

        result.Should().BeApproximately(1.0, 1e-10);
    }
}

public class SemanticSearchTests
{
    [Fact]
    public void Search_returns_ranked_results()
    {
        var query = new double[] { 1, 0 };
        // doc0 is orthogonal to query, doc1 is identical to query
        var corpus = new Tensor<double>([0, 1, 1, 0, 0.5, 0.5], 3, 2);
        var docs = new[] { "orthogonal", "identical", "mixed" };

        var result = SemanticSearch.Search(query, corpus, docs, topK: 2);

        // "identical" should be the top result
        var docCol = (StringColumn)result["document"];
        docCol[0].Should().Be("identical");
    }
}

public class TextColumnAccessorTests
{
    [Fact]
    public void TokenCount_returns_correct_counts()
    {
        var col = new StringColumn("text", ["hello world", "one two three", "single"]);
        var accessor = new TextColumnAccessor(col);

        var counts = accessor.TokenCount();

        counts[0].Should().Be(2);
        counts[1].Should().Be(3);
        counts[2].Should().Be(1);
    }
}
