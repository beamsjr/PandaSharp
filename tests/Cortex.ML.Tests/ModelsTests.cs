using FluentAssertions;
using Cortex.ML.Models;
using Cortex.ML.ModelSelection;
using Cortex.ML.Tensors;

namespace Cortex.ML.Tests;

public class ModelsTests
{
    // Shared synthetic data: y = 2*x1 + 3*x2 + 1
    private static Tensor<double> X => new(new double[] { 1, 0, 2, 0, 3, 0, 0, 1, 0, 2, 0, 3 }, 6, 2);
    private static Tensor<double> Y => new(new double[] { 3, 5, 7, 4, 7, 10 }, 6);

    // Classification: 2 linearly separable classes
    private static Tensor<double> Xc => new(
        new double[] { 0, 0, 0, 1, 1, 0, 1, 1, 2, 2, 2, 3, 3, 2, 3, 3 }, 8, 2);
    private static Tensor<double> Yc => new(new double[] { 0, 0, 0, 0, 1, 1, 1, 1 }, 8);

    // ----------------------------------------------------------------
    // 1. LinearRegression
    // ----------------------------------------------------------------
    [Fact]
    public void LinearRegression_FitSimpleData_WeightsAndInterceptAreCorrect()
    {
        var model = new LinearRegression();
        model.Fit(X, Y);

        model.IsFitted.Should().BeTrue();
        model.Weights!.Length.Should().Be(2);

        // y = 2*x1 + 3*x2 + 1
        model.Weights[0].Should().BeApproximately(2.0, 0.01);
        model.Weights[1].Should().BeApproximately(3.0, 0.01);
        model.Intercept.Should().BeApproximately(1.0, 0.01);
    }

    [Fact]
    public void LinearRegression_Score_RSquaredCloseToOne()
    {
        var model = new LinearRegression();
        model.Fit(X, Y);

        double r2 = model.Score(X, Y);
        r2.Should().BeGreaterThan(0.99);
    }

    // ----------------------------------------------------------------
    // 2. LogisticRegression
    // ----------------------------------------------------------------
    [Fact]
    public void LogisticRegression_FitSeparableData_AccuracyAbove80Percent()
    {
        var model = new LogisticRegression(learningRate: 0.5, maxIterations: 2000);
        model.Fit(Xc, Yc);

        model.IsFitted.Should().BeTrue();
        model.NumClasses.Should().Be(2);

        double accuracy = model.Score(Xc, Yc);
        accuracy.Should().BeGreaterThan(0.8);
    }

    // ----------------------------------------------------------------
    // 3. DecisionTreeClassifier
    // ----------------------------------------------------------------
    [Fact]
    public void DecisionTreeClassifier_FitXorLikeData_PredictionsMatch()
    {
        // XOR-like pattern: class depends on quadrant
        var xXor = new Tensor<double>(
            new double[] { 0, 0, 0, 1, 1, 0, 1, 1 }, 4, 2);
        var yXor = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);

        var tree = new DecisionTreeClassifier();
        tree.Fit(xXor, yXor);

        tree.IsFitted.Should().BeTrue();

        var preds = tree.Predict(xXor);
        for (int i = 0; i < 4; i++)
        {
            preds[i].Should().BeApproximately(yXor.Span[i], 0.01,
                $"prediction for sample {i} should match label");
        }
    }

    // ----------------------------------------------------------------
    // 4. DecisionTreeRegressor
    // ----------------------------------------------------------------
    [Fact]
    public void DecisionTreeRegressor_FitLinearData_RSquaredAboveHalf()
    {
        var model = new DecisionTreeRegressor();
        model.Fit(X, Y);

        model.IsFitted.Should().BeTrue();

        double r2 = model.Score(X, Y);
        r2.Should().BeGreaterThan(0.5);
    }

    // ----------------------------------------------------------------
    // 5. RandomForestClassifier
    // ----------------------------------------------------------------
    [Fact]
    public void RandomForestClassifier_FitTwoClassData_AccuracyAbove70Percent()
    {
        var model = new RandomForestClassifier(nEstimators: 20, seed: 42);
        model.Fit(Xc, Yc);

        model.IsFitted.Should().BeTrue();

        double accuracy = model.Score(Xc, Yc);
        accuracy.Should().BeGreaterThan(0.7);
    }

    // ----------------------------------------------------------------
    // 6. KNearestNeighborsClassifier
    // ----------------------------------------------------------------
    [Fact]
    public void KNearestNeighborsClassifier_FitClusteredData_PredictionsCorrect()
    {
        var model = new KNearestNeighborsClassifier(k: 3);
        model.Fit(Xc, Yc);

        model.IsFitted.Should().BeTrue();

        var preds = model.Predict(Xc);
        int correct = 0;
        for (int i = 0; i < 8; i++)
        {
            if (Math.Abs(preds.Span[i] - Yc.Span[i]) < 0.01) correct++;
        }

        ((double)correct / 8).Should().BeGreaterThan(0.7);
    }

    // ----------------------------------------------------------------
    // 7. KMeans
    // ----------------------------------------------------------------
    [Fact]
    public void KMeans_FitTwoClusterData_FindsTwoDistinctCentroids()
    {
        // Two well-separated clusters
        var xKm = new Tensor<double>(
            new double[] { 0, 0, 0.1, 0.1, -0.1, 0.1, 0, -0.1,
                           10, 10, 10.1, 10.1, 9.9, 10.1, 10, 9.9 }, 8, 2);
        var yDummy = new Tensor<double>(new double[8], 8);

        var model = new KMeans(nClusters: 2, seed: 42);
        model.Fit(xKm, yDummy);

        model.IsFitted.Should().BeTrue();
        model.ClusterCenters.Should().NotBeNull();
        model.ClusterCenters!.Shape.Should().Equal(new[] { 2, 2 });

        // The two centroids should be far apart (one near 0,0 and one near 10,10)
        double c0x = model.ClusterCenters[0, 0];
        double c0y = model.ClusterCenters[0, 1];
        double c1x = model.ClusterCenters[1, 0];
        double c1y = model.ClusterCenters[1, 1];

        double dist = Math.Sqrt((c0x - c1x) * (c0x - c1x) + (c0y - c1y) * (c0y - c1y));
        dist.Should().BeGreaterThan(5.0, "centroids should be well-separated");
    }

    // ----------------------------------------------------------------
    // 8. PCA
    // ----------------------------------------------------------------
    [Fact]
    public void PCA_FitPlanarData_FirstTwoComponentsExplainMostVariance()
    {
        // 3D data that lives in a 2D plane: z = x + y
        var xPca = new Tensor<double>(
            new double[]
            {
                1, 0, 1,
                0, 1, 1,
                1, 1, 2,
                2, 0, 2,
                0, 2, 2,
                2, 1, 3,
                1, 2, 3,
                3, 0, 3,
                0, 3, 3,
                2, 2, 4,
            }, 10, 3);

        var pca = new PCA(nComponents: 2);
        pca.Fit(xPca);

        pca.IsFitted.Should().BeTrue();
        pca.ExplainedVarianceRatio.Should().NotBeNull();
        pca.ExplainedVarianceRatio!.Length.Should().Be(2);

        double totalExplained = pca.ExplainedVarianceRatio[0] + pca.ExplainedVarianceRatio[1];
        totalExplained.Should().BeGreaterThan(0.90,
            "first 2 components should explain >90% of variance for planar data");
    }

    // ----------------------------------------------------------------
    // 9. ElasticNet
    // ----------------------------------------------------------------
    [Fact]
    public void ElasticNet_FitLinearData_WeightsAreShrunk()
    {
        // Fit OLS first for comparison
        var ols = new LinearRegression();
        ols.Fit(X, Y);

        // Fit ElasticNet with strong regularization
        var enet = new ElasticNet(alpha: 1.0, l1Ratio: 0.5);
        enet.Fit(X, Y);

        enet.IsFitted.Should().BeTrue();
        enet.Weights.Should().NotBeNull();

        // ElasticNet weights should be smaller in magnitude than OLS weights
        double olsNorm = Math.Abs(ols.Weights![0]) + Math.Abs(ols.Weights[1]);
        double enetNorm = Math.Abs(enet.Weights![0]) + Math.Abs(enet.Weights[1]);

        enetNorm.Should().BeLessThan(olsNorm,
            "regularized weights should be shrunk compared to OLS");
    }

    // ----------------------------------------------------------------
    // 10. CrossValidation
    // ----------------------------------------------------------------
    [Fact]
    public void CrossValScore_LinearRegression_ReturnsFiveFoldScores()
    {
        // More data for 5-fold CV
        var xCv = new Tensor<double>(
            new double[]
            {
                1, 0, 2, 0, 3, 0, 4, 0, 5, 0,
                0, 1, 0, 2, 0, 3, 0, 4, 0, 5,
                1, 1, 2, 2, 3, 3, 4, 4, 5, 5,
            }, 15, 2);
        var yCv = new Tensor<double>(
            new double[]
            {
                3, 5, 7, 9, 11,
                4, 7, 10, 13, 16,
                6, 11, 16, 21, 26,
            }, 15);

        var model = new LinearRegression();
        double[] scores = CrossValidation.CrossValScore(model, xCv, yCv, nFolds: 5, seed: 42);

        scores.Should().HaveCount(5);
        // Each fold score should be a finite number
        foreach (double s in scores)
        {
            double.IsFinite(s).Should().BeTrue();
        }
    }

    // ----------------------------------------------------------------
    // 11. Null guard tests
    // ----------------------------------------------------------------
    [Fact]
    public void LinearRegression_FitNullX_ThrowsArgumentNullException()
    {
        var model = new LinearRegression();
        var act = () => model.Fit(null!, Y);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void LinearRegression_FitNullY_ThrowsArgumentNullException()
    {
        var model = new LinearRegression();
        var act = () => model.Fit(X, null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void LinearRegression_PredictNull_ThrowsArgumentNullException()
    {
        var model = new LinearRegression();
        model.Fit(X, Y);
        var act = () => model.Predict(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void LogisticRegression_FitNullX_ThrowsArgumentNullException()
    {
        var model = new LogisticRegression();
        var act = () => model.Fit(null!, Yc);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void DecisionTreeClassifier_PredictNull_ThrowsArgumentNullException()
    {
        var tree = new DecisionTreeClassifier();
        tree.Fit(Xc, Yc);
        var act = () => tree.Predict(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    // ----------------------------------------------------------------
    // 12. Dimension mismatch tests
    // ----------------------------------------------------------------
    [Fact]
    public void LinearRegression_FitDimensionMismatch_ThrowsArgumentException()
    {
        var model = new LinearRegression();
        var wrongY = new Tensor<double>(new double[] { 1, 2, 3 }, 3); // 3 instead of 6
        var act = () => model.Fit(X, wrongY);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void LogisticRegression_FitDimensionMismatch_ThrowsArgumentException()
    {
        var model = new LogisticRegression();
        var wrongY = new Tensor<double>(new double[] { 0, 1 }, 2); // 2 instead of 8
        var act = () => model.Fit(Xc, wrongY);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DecisionTreeClassifier_FitDimensionMismatch_ThrowsArgumentException()
    {
        var tree = new DecisionTreeClassifier();
        var wrongY = new Tensor<double>(new double[] { 0, 1, 0 }, 3);
        var act = () => tree.Fit(Xc, wrongY);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void DecisionTreeRegressor_FitDimensionMismatch_ThrowsArgumentException()
    {
        var model = new DecisionTreeRegressor();
        var wrongY = new Tensor<double>(new double[] { 1, 2 }, 2);
        var act = () => model.Fit(X, wrongY);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ElasticNet_FitDimensionMismatch_ThrowsArgumentException()
    {
        var model = new ElasticNet();
        var wrongY = new Tensor<double>(new double[] { 1 }, 1);
        var act = () => model.Fit(X, wrongY);
        act.Should().Throw<ArgumentException>();
    }
}
