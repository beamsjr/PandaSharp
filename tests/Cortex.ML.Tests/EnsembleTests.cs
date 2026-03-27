using FluentAssertions;
using Cortex.ML.Models;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests;

public class EnsembleTests
{
    // Linearly separable 2-class data
    private static Tensor<double> Xc => new(
        new double[] { 0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 3, 3, 2, 3, 3 }, 8, 2);
    private static Tensor<double> Yc => new(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 }, 8);

    // Simple regression data: y = 2*x1 + 3*x2 + 1
    private static Tensor<double> Xr => new(new double[] { 1, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 3 }, 6, 2);
    private static Tensor<double> Yr => new(new double[] { 3, 5, 7, 4, 7, 10 }, 6);

    // ----------------------------------------------------------------
    // VotingEnsemble - Hard
    // ----------------------------------------------------------------
    [Fact]
    public void VotingEnsemble_HardVoting_MajorityWins()
    {
        var models = new IClassifier[]
        {
            new LogisticRegression(learningRate: 0.5, maxIterations: 2000),
            new LogisticRegression(learningRate: 0.3, maxIterations: 2000),
            new KNearestNeighborsClassifier(k: 3),
        };

        var ensemble = new VotingEnsemble(models, VotingStrategy.Hard);
        ensemble.Fit(Xc, Yc);

        ensemble.IsFitted.Should().BeTrue();
        ensemble.NumClasses.Should().Be(2);

        var predictions = ensemble.Predict(Xc);
        predictions.Length.Should().Be(8);

        // Since all models should do well on this separable data, accuracy should be high
        double accuracy = ensemble.Score(Xc, Yc);
        accuracy.Should().BeGreaterThan(0.7);
    }

    // ----------------------------------------------------------------
    // VotingEnsemble - Soft
    // ----------------------------------------------------------------
    [Fact]
    public void VotingEnsemble_SoftVoting_AveragesProbabilities()
    {
        var models = new IClassifier[]
        {
            new LogisticRegression(learningRate: 0.5, maxIterations: 2000),
            new LogisticRegression(learningRate: 0.3, maxIterations: 2000),
            new KNearestNeighborsClassifier(k: 3),
        };

        var ensemble = new VotingEnsemble(models, VotingStrategy.Soft);
        ensemble.Fit(Xc, Yc);

        var proba = ensemble.PredictProba(Xc);
        proba.Shape[0].Should().Be(8);
        proba.Shape[1].Should().Be(2);

        // Probabilities should sum to ~1 per sample
        var span = proba.Span;
        for (int i = 0; i < 8; i++)
        {
            double sum = span[i * 2] + span[i * 2 + 1];
            sum.Should().BeApproximately(1.0, 0.01);
        }

        double accuracy = ensemble.Score(Xc, Yc);
        accuracy.Should().BeGreaterThan(0.7);
    }

    // ----------------------------------------------------------------
    // VotingEnsemble - Weighted
    // ----------------------------------------------------------------
    [Fact]
    public void VotingEnsemble_Weighted_RespectsWeights()
    {
        var models = new IClassifier[]
        {
            new LogisticRegression(learningRate: 0.5, maxIterations: 2000),
            new LogisticRegression(learningRate: 0.3, maxIterations: 2000),
            new KNearestNeighborsClassifier(k: 3),
        };

        var weights = new double[] { 2.0, 1.0, 1.0 };
        var ensemble = new VotingEnsemble(models, VotingStrategy.Soft, weights);
        ensemble.Fit(Xc, Yc);

        ensemble.IsFitted.Should().BeTrue();

        double accuracy = ensemble.Score(Xc, Yc);
        accuracy.Should().BeGreaterThan(0.6);
    }

    // ----------------------------------------------------------------
    // VotingEnsemble - Validation
    // ----------------------------------------------------------------
    [Fact]
    public void VotingEnsemble_ThrowsBeforeFit()
    {
        var models = new IClassifier[] { new LogisticRegression() };
        var ensemble = new VotingEnsemble(models);

        var act = () => ensemble.Predict(Xc);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void VotingEnsemble_ThrowsOnEmptyModels()
    {
        var act = () => new VotingEnsemble(Array.Empty<IClassifier>());
        act.Should().Throw<ArgumentException>();
    }

    // ----------------------------------------------------------------
    // StackingEnsemble
    // ----------------------------------------------------------------
    [Fact]
    public void StackingEnsemble_FitAndPredict_Works()
    {
        var baseModels = new IModel[]
        {
            new LinearRegression(),
            new LinearRegression(l2Penalty: 0.1),
        };
        var meta = new LinearRegression();

        var stacking = new StackingEnsemble(baseModels, meta, nFolds: 3, seed: 42);
        stacking.Fit(Xr, Yr);

        stacking.IsFitted.Should().BeTrue();

        var predictions = stacking.Predict(Xr);
        predictions.Length.Should().Be(6);

        // Score should be reasonable (linear data, linear models)
        double score = stacking.Score(Xr, Yr);
        score.Should().BeGreaterThan(0.5);
    }

    [Fact]
    public void StackingEnsemble_ThrowsBeforeFit()
    {
        var stacking = new StackingEnsemble(
            new IModel[] { new LinearRegression() },
            new LinearRegression());

        var act = () => stacking.Predict(Xr);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void StackingEnsemble_ThrowsOnEmptyBaseModels()
    {
        var act = () => new StackingEnsemble(Array.Empty<IModel>(), new LinearRegression());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void StackingEnsemble_MultipleBaseModels_ProducesMetaFeatures()
    {
        // Use 3 base models with different regularization
        var baseModels = new IModel[]
        {
            new LinearRegression(l2Penalty: 0.0),
            new LinearRegression(l2Penalty: 0.5),
            new LinearRegression(l2Penalty: 2.0),
        };
        var meta = new LinearRegression();

        // Larger dataset to support 3-fold CV
        var xData = new double[]
        {
            1, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 3,
            1, 1, 2, 1, 3, 1, 1, 2, 2, 2, 3, 3
        };
        var yData = new double[] { 3, 5, 7, 4, 7, 10, 6, 8, 10, 8, 10, 16 };

        var X = new Tensor<double>(xData, 12, 2);
        var y = new Tensor<double>(yData, 12);

        var stacking = new StackingEnsemble(baseModels, meta, nFolds: 3, seed: 7);
        stacking.Fit(X, y);

        var predictions = stacking.Predict(X);
        predictions.Length.Should().Be(12);

        double score = stacking.Score(X, y);
        score.Should().BeGreaterThan(0.3, "stacking should achieve a reasonable fit");
    }
}
