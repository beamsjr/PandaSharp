using FluentAssertions;
using Cortex.Column;
using Cortex.ML.Metrics;
using Cortex.ML.Models;
using Cortex.ML.Tensors;
using Cortex.ML.Transformers;
using Xunit;

namespace Cortex.ML.Tests.EdgeCases;

public class MLEdgeCaseRound4Tests
{
    // ═══════════════════════════════════════════════════════════
    // BUG 1: Model.Score returns 1.0 when ssTot==0 but ssRes>0
    // Multiple regression models blindly return 1.0 when all y
    // values are constant (ssTot==0), even when predictions are
    // completely wrong (ssRes>0). Only StackingEnsemble handles
    // this correctly. The fix: return ssRes==0 ? 1.0 : 0.0
    // when ssTot==0.
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void LinearRegression_Score_ConstantTarget_WrongPredictions_ShouldNotReturn1()
    {
        // Train on simple data so model is fitted
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new LinearRegression();
        model.Fit(X, y);

        // Score on constant target (all 100s) — model predicts something else
        var Xtest = new Tensor<double>(new double[] { 10, 20, 30, 40 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 100, 100 }, 2);
        var score = model.Score(Xtest, yConst);

        // ssTot == 0 because y is constant, but predictions are way off.
        // Score should NOT be 1.0 (perfect). It should be <= 0.
        score.Should().BeLessThanOrEqualTo(0.0,
            "a model that predicts wrong values on constant target cannot have R²=1.0");
    }

    [Fact]
    public void DecisionTreeRegressor_Score_ConstantTarget_WrongPredictions_ShouldNotReturn1()
    {
        // Train on non-trivial data
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
            2, 1,
        }, 5, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3, 0, 4 }, 5);
        var model = new DecisionTreeRegressor();
        model.Fit(X, y);

        // Score on constant target where prediction will likely be wrong
        var Xtest = new Tensor<double>(new double[] { 10, 10, 20, 20 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 999, 999 }, 2);
        var score = model.Score(Xtest, yConst);

        // Predictions won't be 999, so ssRes > 0, ssTot == 0 → should NOT be 1.0
        score.Should().BeLessThanOrEqualTo(0.0,
            "wrong predictions on constant target should give R² <= 0, not 1.0");
    }

    [Fact]
    public void LinearRegression_Score_ConstantTarget_PerfectPredictions_ShouldReturn1()
    {
        // Train model to predict a constant (intercept-only)
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var yConst = new Tensor<double>(new double[] { 5, 5, 5 }, 3);
        var model = new LinearRegression();
        model.Fit(X, yConst);

        // Model should predict ~5 for any input, scoring on constant target = 5
        var score = model.Score(X, yConst);
        score.Should().BeApproximately(1.0, 1e-6,
            "perfect predictions on constant target should give R² = 1.0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 2: RegressionMetrics.R2 returns 0.0 when ssTot==0
    // even when predictions are perfect (ssRes==0).
    // The correct behavior: R²=1.0 for perfect predictions.
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void RegressionMetrics_R2_PerfectPredictions_ConstantTarget_ShouldReturn1()
    {
        var yTrue = new Tensor<double>(new double[] { 5, 5, 5 }, 3);
        var yPred = new Tensor<double>(new double[] { 5, 5, 5 }, 3);

        var r2 = RegressionMetrics.R2(yTrue, yPred);

        // Perfect predictions: ssRes=0, ssTot=0. Should be R²=1.0, not 0.0.
        r2.Should().Be(1.0,
            "perfect predictions (ssRes=0) should always give R²=1.0, even when target is constant");
    }

    [Fact]
    public void RegressionMetrics_R2_WrongPredictions_ConstantTarget_ShouldNotReturn1()
    {
        var yTrue = new Tensor<double>(new double[] { 5, 5, 5 }, 3);
        var yPred = new Tensor<double>(new double[] { 1, 2, 3 }, 3);

        var r2 = RegressionMetrics.R2(yTrue, yPred);

        // Wrong predictions: ssRes>0, ssTot=0. Should not be 1.0.
        r2.Should().BeLessThanOrEqualTo(0.0,
            "wrong predictions on constant target should give R² <= 0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 3: MetricsCalculator.Regression R2 returns 0 when
    // ssTot==0, even for perfect predictions.
    // Same pattern as Bug 2 but for the Column-based API.
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void MetricsCalculator_Regression_PerfectPredictions_ConstantTarget_R2ShouldBe1()
    {
        var yTrue = new Column<double>("true", new double[] { 5, 5, 5 });
        var yPred = new Column<double>("pred", new double[] { 5, 5, 5 });

        var result = MetricsCalculator.Regression(yTrue, yPred);

        result.R2.Should().Be(1.0,
            "perfect predictions (ssRes=0) on constant target should give R²=1.0");
    }

    [Fact]
    public void MetricsCalculator_Regression_WrongPredictions_ConstantTarget_R2ShouldNotBe1()
    {
        var yTrue = new Column<double>("true", new double[] { 5, 5, 5 });
        var yPred = new Column<double>("pred", new double[] { 1, 2, 3 });

        var result = MetricsCalculator.Regression(yTrue, yPred);

        result.R2.Should().BeLessThanOrEqualTo(0.0,
            "wrong predictions on constant target should give R² <= 0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 4: RegressionMetrics.R2 with single sample returns
    // 0.0 (ssTot always == 0 for n=1). Perfect prediction on
    // 1 sample should give R²=1.0.
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void RegressionMetrics_R2_SingleSample_PerfectPrediction_ShouldReturn1()
    {
        var yTrue = new Tensor<double>(new double[] { 42 }, 1);
        var yPred = new Tensor<double>(new double[] { 42 }, 1);

        var r2 = RegressionMetrics.R2(yTrue, yPred);

        r2.Should().Be(1.0,
            "single sample with perfect prediction should give R²=1.0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 5: Multiple regression model Scores return 1.0 for
    // single sample with wrong prediction (ssTot always 0 for
    // n=1). This is a consequence of Bug 1.
    // Testing with KNearestNeighborsRegressor as another example.
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void KNearestNeighborsRegressor_Score_SingleSample_WrongPrediction_ShouldNotReturn1()
    {
        // Train on some data
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
        }, 4, 2);
        var y = new Tensor<double>(new double[] { 10, 20, 30, 40 }, 4);
        var model = new KNearestNeighborsRegressor(k: 2);
        model.Fit(X, y);

        // Score on 1 sample with an extreme target the model can't predict
        var Xtest = new Tensor<double>(new double[] { 0.5, 0.5 }, 1, 2);
        var ytest = new Tensor<double>(new double[] { 9999 }, 1);
        var score = model.Score(Xtest, ytest);

        // With 1 sample, ssTot=0 always. Model predicts ~25 not 9999.
        score.Should().BeLessThanOrEqualTo(0.0,
            "wrong prediction on single sample should not give R²=1.0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 6: GradientBoostedTreeRegressor.Score has same bug
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void GradientBoostedTreeRegressor_Score_ConstantTarget_WrongPredictions()
    {
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
            2, 1,
        }, 5, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3, 4, 5 }, 5);
        var model = new GradientBoostedTreeRegressor(nEstimators: 5, maxDepth: 2, seed: 42);
        model.Fit(X, y);

        var Xtest = new Tensor<double>(new double[] { 10, 10, 20, 20 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 9999, 9999 }, 2);
        var score = model.Score(Xtest, yConst);

        score.Should().BeLessThanOrEqualTo(0.0,
            "wrong predictions on constant target should give R² <= 0");
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 7: RandomForestRegressor.Score has same bug
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void RandomForestRegressor_Score_ConstantTarget_WrongPredictions()
    {
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
            2, 1,
        }, 5, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3, 4, 5 }, 5);
        var model = new RandomForestRegressor(nEstimators: 5, maxDepth: 3, seed: 42);
        model.Fit(X, y);

        var Xtest = new Tensor<double>(new double[] { 10, 10, 20, 20 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 9999, 9999 }, 2);
        var score = model.Score(Xtest, yConst);

        score.Should().BeLessThanOrEqualTo(0.0);
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 8: ElasticNet.Score has same bug
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void ElasticNet_Score_ConstantTarget_WrongPredictions()
    {
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
            2, 1,
        }, 5, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3, 4, 5 }, 5);
        var model = new ElasticNet(l1Ratio: 0.5, alpha: 0.01);
        model.Fit(X, y);

        var Xtest = new Tensor<double>(new double[] { 10, 10, 20, 20 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 9999, 9999 }, 2);
        var score = model.Score(Xtest, yConst);

        score.Should().BeLessThanOrEqualTo(0.0);
    }

    // ═══════════════════════════════════════════════════════════
    // BUG 9: SGDRegressor.Score has same bug
    // ═══════════════════════════════════════════════════════════

    [Fact]
    public void SGDRegressor_Score_ConstantTarget_WrongPredictions()
    {
        var X = new Tensor<double>(new double[]
        {
            1, 0,
            0, 1,
            1, 1,
            0, 0,
            2, 1,
        }, 5, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3, 4, 5 }, 5);
        var model = new SGDRegressor(eta0: 0.01, maxEpochs: 100, seed: 42);
        model.Fit(X, y);

        var Xtest = new Tensor<double>(new double[] { 10, 10, 20, 20 }, 2, 2);
        var yConst = new Tensor<double>(new double[] { 9999, 9999 }, 2);
        var score = model.Score(Xtest, yConst);

        score.Should().BeLessThanOrEqualTo(0.0);
    }
}
