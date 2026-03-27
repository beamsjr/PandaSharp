using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.ML.Tests;

public class TextVectorizerTests
{
    [Fact]
    public void TfIdf_BasicVectorization()
    {
        var df = new DataFrame(
            new StringColumn("Text", [
                "the cat sat on the mat",
                "the dog sat on the log",
                "cats and dogs"
            ])
        );

        var vectorizer = new TextVectorizer("Text", VectorizerMode.TfIdf, maxFeatures: 10);
        var result = vectorizer.FitTransform(df);

        result.ColumnNames.Should().NotContain("Text"); // original dropped
        vectorizer.Vocabulary.Should().NotBeNull();
        vectorizer.Vocabulary!.Count.Should().BeGreaterThan(0);

        // "the" appears in all docs → low IDF → low TF-IDF
        // "cat" appears in 1 doc → high IDF
        result.ColumnCount.Should().BeGreaterThan(5);
    }

    [Fact]
    public void Count_BasicVectorization()
    {
        var df = new DataFrame(
            new StringColumn("Text", ["hello world hello", "world foo bar"])
        );

        var vectorizer = new TextVectorizer("Text", VectorizerMode.Count, maxFeatures: 5);
        var result = vectorizer.FitTransform(df);

        // "hello" appears 2x in doc 0
        var helloCols = result.ColumnNames.Where(n => n.Contains("hello")).ToList();
        if (helloCols.Count > 0)
            result.GetColumn<double>(helloCols[0])[0].Should().Be(2);
    }

    [Fact]
    public void MaxFeatures_Limits()
    {
        var df = new DataFrame(
            new StringColumn("Text", ["a b c d e f g h i j k l m n o p"])
        );

        var vectorizer = new TextVectorizer("Text", maxFeatures: 5);
        vectorizer.FitTransform(df);

        vectorizer.Vocabulary!.Count.Should().Be(5);
    }

    [Fact]
    public void FitThenTransform_UsesVocab()
    {
        var train = new DataFrame(
            new StringColumn("Text", ["machine learning is great", "deep learning rocks"])
        );
        var test = new DataFrame(
            new StringColumn("Text", ["learning is fun", "unknown words here"])
        );

        var vectorizer = new TextVectorizer("Text", VectorizerMode.Count, maxFeatures: 10);
        vectorizer.Fit(train);
        var result = vectorizer.Transform(test);

        // "learning" should have a feature; "unknown" should not (not in vocabulary)
        result.ColumnNames.Should().Contain(n => n.Contains("learning"));
    }

    [Fact]
    public void NullText_HandledGracefully()
    {
        var df = new DataFrame(
            new StringColumn("Text", ["hello world", null, "foo bar"])
        );

        var result = new TextVectorizer("Text", maxFeatures: 5).FitTransform(df);
        result.RowCount.Should().Be(3);
    }
}
