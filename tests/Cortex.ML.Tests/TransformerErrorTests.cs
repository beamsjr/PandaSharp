using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Transformers;

namespace Cortex.ML.Tests;

public class TransformerErrorTests
{
    private static DataFrame Df() => new(new Column<double>("X", [1, 2, 3]));

    [Fact]
    public void StandardScaler_TransformBeforeFit_Throws()
    {
        var act = () => new StandardScaler("X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void MinMaxScaler_TransformBeforeFit_Throws()
    {
        var act = () => new MinMaxScaler("X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void LabelEncoder_TransformBeforeFit_Throws()
    {
        var df = new DataFrame(new StringColumn("S", ["a", "b"]));
        var act = () => new LabelEncoder("S").Transform(df);
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void OneHotEncoder_TransformBeforeFit_Throws()
    {
        var df = new DataFrame(new StringColumn("S", ["a"]));
        var act = () => new OneHotEncoder("S").Transform(df);
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void Imputer_TransformBeforeFit_Throws()
    {
        var act = () => new Imputer(columns: "X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void RobustScaler_TransformBeforeFit_Throws()
    {
        var act = () => new RobustScaler("X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void TargetEncoder_TransformBeforeFit_Throws()
    {
        var act = () => new TargetEncoder("Y", columns: "X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void TextVectorizer_TransformBeforeFit_Throws()
    {
        var df = new DataFrame(new StringColumn("T", ["hello"]));
        var act = () => new TextVectorizer("T").Transform(df);
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void Discretizer_TransformBeforeFit_Throws()
    {
        var act = () => new Discretizer(columns: "X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }

    [Fact]
    public void PolynomialFeatures_TransformBeforeFit_Throws()
    {
        var act = () => new PolynomialFeatures(columns: "X").Transform(Df());
        act.Should().Throw<InvalidOperationException>().WithMessage("*Fit*");
    }
}
