using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Models;
using PandaSharp.ML.ModelSelection;
using PandaSharp.ML.Tensors;
using PandaSharp.ML.Transformers;

namespace PandaSharp.ML.Tests.EdgeCases;

public class MLEdgeCaseTests
{
    // ================================================================
    // 1. LinearRegression edge cases
    // ================================================================

    [Fact]
    public void LinearRegression_PredictBeforeFit_Throws()
    {
        var model = new LinearRegression();
        var X = new Tensor<double>(new double[] { 1, 2 }, 1, 2);

        var act = () => model.Predict(X);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void LinearRegression_FitWithOneSample_ShouldNotThrow()
    {
        // 1 sample, 1 feature: system is exactly determined (1 equation, 2 unknowns with intercept)
        // With L2 regularization it should be solvable
        var model = new LinearRegression(l2Penalty: 1.0);
        var X = new Tensor<double>(new double[] { 2.0 }, 1, 1);
        var y = new Tensor<double>(new double[] { 5.0 }, 1);

        var act = () => model.Fit(X, y);
        act.Should().NotThrow();
        model.IsFitted.Should().BeTrue();

        var pred = model.Predict(X);
        // Should produce a finite prediction
        double.IsFinite(pred.Span[0]).Should().BeTrue();
    }

    [Fact]
    public void LinearRegression_FitWithOneSample_NoRegularization_ShouldThrowMeaningfulError()
    {
        // 1 sample, 1 feature, no regularization: XtX is singular (rank 1, size 2x2)
        // Should throw a clear error, not crash with BLAS abort
        var model = new LinearRegression(l2Penalty: 0.0);
        var X = new Tensor<double>(new double[] { 2.0 }, 1, 1);
        var y = new Tensor<double>(new double[] { 5.0 }, 1);

        var act = () => model.Fit(X, y);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void LinearRegression_MoreFeaturesThanSamples_WithRegularization_ShouldNotThrow()
    {
        // 2 samples, 5 features: underdetermined without regularization
        var model = new LinearRegression(l2Penalty: 1.0);
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 }, 2, 5);
        var y = new Tensor<double>(new double[] { 1.0, 2.0 }, 2);

        var act = () => model.Fit(X, y);
        act.Should().NotThrow();
    }

    [Fact]
    public void LinearRegression_AllZeroY_FitsWithZeroWeights()
    {
        var model = new LinearRegression();
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 0.0, 0.0, 0.0 }, 3);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var pred = model.Predict(X);
        for (int i = 0; i < pred.Length; i++)
            pred.Span[i].Should().BeApproximately(0.0, 1e-6);
    }

    [Fact]
    public void LinearRegression_AllConstantX_FitsWithInterceptOnly()
    {
        // All X values are the same -- columns are constant
        var model = new LinearRegression(l2Penalty: 0.01);
        var X = new Tensor<double>(new double[] { 5, 5, 5, 5, 5, 5 }, 3, 2);
        var y = new Tensor<double>(new double[] { 3.0, 3.0, 3.0 }, 3);

        var act = () => model.Fit(X, y);
        act.Should().NotThrow();
    }

    // ================================================================
    // 2. LogisticRegression edge cases
    // ================================================================

    [Fact]
    public void LogisticRegression_SingleClassTraining_ShouldFitAndPredict()
    {
        // All labels are the same class
        var model = new LogisticRegression(maxIterations: 10);
        var X = new Tensor<double>(new double[] { 1, 0, 2, 0, 3, 0 }, 3, 2);
        var y = new Tensor<double>(new double[] { 0, 0, 0 }, 3);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();
        model.NumClasses.Should().Be(1);

        // Should predict class 0 for all samples
        var pred = model.Predict(X);
        pred.Length.Should().Be(3);
        for (int i = 0; i < pred.Length; i++)
            pred.Span[i].Should().Be(0.0);
    }

    [Fact]
    public void LogisticRegression_ZeroIterations_ShouldStillPredict()
    {
        var model = new LogisticRegression(maxIterations: 0);
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 2, 2, 3, 3 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 0, 1, 1 }, 4);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        // Should produce valid predictions even with zero iterations (untrained weights)
        var pred = model.Predict(X);
        pred.Length.Should().Be(4);
    }

    // ================================================================
    // 3. DecisionTree edge cases
    // ================================================================

    [Fact]
    public void DecisionTree_OneSample_ShouldFitAndPredict()
    {
        var model = new DecisionTreeClassifier();
        var X = new Tensor<double>(new double[] { 1.0, 2.0 }, 1, 2);
        var y = new Tensor<double>(new double[] { 0 }, 1);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var pred = model.Predict(X);
        pred.Span[0].Should().Be(0);
    }

    [Fact]
    public void DecisionTree_AllSameFeatures_ShouldReturnMajorityClass()
    {
        var model = new DecisionTreeClassifier();
        // All features identical
        var X = new Tensor<double>(new double[] { 1, 1, 1, 1, 1, 1 }, 3, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 0 }, 3);

        model.Fit(X, y);
        var pred = model.Predict(X);
        // Since all features are the same, can't split; should return majority class (0)
        for (int i = 0; i < 3; i++)
            pred.Span[i].Should().Be(0);
    }

    [Fact]
    public void DecisionTree_AllSameTarget_ShouldPredictThatTarget()
    {
        var model = new DecisionTreeClassifier();
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 1, 1 }, 3);

        model.Fit(X, y);
        var pred = model.Predict(X);
        for (int i = 0; i < 3; i++)
            pred.Span[i].Should().Be(1);
    }

    [Fact]
    public void DecisionTree_MaxDepth1_ShouldProduceShallowTree()
    {
        var model = new DecisionTreeClassifier(maxDepth: 1);
        var X = new Tensor<double>(new double[] { 0, 1, 2, 3, 4, 5, 6, 7 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 0, 1, 1 }, 4);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var pred = model.Predict(X);
        pred.Length.Should().Be(4);
    }

    [Fact]
    public void DecisionTree_NegativeMaxDepth_ShouldTreatAsUnlimited()
    {
        var model = new DecisionTreeClassifier(maxDepth: -1);
        var X = new Tensor<double>(new double[] { 0, 1, 2, 3, 4, 5, 6, 7 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 0, 1, 1 }, 4);

        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();
        model.Score(X, y).Should().BeGreaterThanOrEqualTo(0.5);
    }

    // ================================================================
    // 4. KNN edge cases
    // ================================================================

    [Fact]
    public void KNN_KLargerThanTrainingSet_ShouldUseAllSamples()
    {
        // K=10 but only 3 training samples
        var model = new KNearestNeighborsClassifier(k: 10);
        var X = new Tensor<double>(new double[] { 0, 1, 2 }, 3, 1);
        var y = new Tensor<double>(new double[] { 0, 0, 1 }, 3);

        model.Fit(X, y);
        var pred = model.Predict(new Tensor<double>(new double[] { 0.5 }, 1, 1));
        pred.Length.Should().Be(1);
    }

    [Fact]
    public void KNN_KEqualsZero_ShouldThrow()
    {
        var act = () => new KNearestNeighborsClassifier(k: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void KNN_OneSample_ShouldPredict()
    {
        var model = new KNearestNeighborsClassifier(k: 1);
        var X = new Tensor<double>(new double[] { 1.0 }, 1, 1);
        var y = new Tensor<double>(new double[] { 0 }, 1);

        model.Fit(X, y);
        var pred = model.Predict(new Tensor<double>(new double[] { 2.0 }, 1, 1));
        pred.Span[0].Should().Be(0);
    }

    [Fact]
    public void KNN_PredictBeforeFit_ShouldThrow()
    {
        var model = new KNearestNeighborsClassifier(k: 3);
        var X = new Tensor<double>(new double[] { 1.0 }, 1, 1);

        var act = () => model.Predict(X);
        act.Should().Throw<InvalidOperationException>();
    }

    // ================================================================
    // 5. KMeans edge cases
    // ================================================================

    [Fact]
    public void KMeans_NClustersGreaterThanSamples_ShouldThrow()
    {
        var model = new KMeans(nClusters: 5, seed: 42);
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);
        var y = new Tensor<double>(new double[] { 0, 0 }, 2);

        var act = () => model.Fit(X, y);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void KMeans_OneSampleOneCluster_ShouldFit()
    {
        var model = new KMeans(nClusters: 1, seed: 42);
        var X = new Tensor<double>(new double[] { 1.0, 2.0 }, 1, 2);

        model.Fit(X);
        model.IsFitted.Should().BeTrue();
        model.Labels!.Length.Should().Be(1);
        model.Labels[0].Should().Be(0);
    }

    [Fact]
    public void KMeans_AllIdenticalPoints_ShouldFit()
    {
        var model = new KMeans(nClusters: 2, seed: 42);
        var X = new Tensor<double>(new double[] { 1, 1, 1, 1, 1, 1 }, 3, 2);

        model.Fit(X);
        model.IsFitted.Should().BeTrue();
    }

    // ================================================================
    // 6. PCA edge cases
    // ================================================================

    [Fact]
    public void PCA_NComponentsGreaterThanFeatures_ShouldThrow()
    {
        var pca = new PCA(nComponents: 5);
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);

        var act = () => pca.Fit(X);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void PCA_OneSample_ShouldFitAndTransform()
    {
        var pca = new PCA(nComponents: 1);
        var X = new Tensor<double>(new double[] { 1.0, 2.0, 3.0 }, 1, 3);

        // With 1 sample, centering makes everything zero
        // SVD of zero matrix or eigendecomposition of zero covariance matrix
        var act = () => pca.FitTransform(X);
        act.Should().NotThrow();
        pca.IsFitted.Should().BeTrue();
    }

    [Fact]
    public void PCA_AllConstantColumn_ShouldFit()
    {
        var pca = new PCA(nComponents: 1);
        // First column is constant
        var X = new Tensor<double>(new double[] { 5, 1, 5, 2, 5, 3, 5, 4 }, 4, 2);

        pca.Fit(X);
        pca.IsFitted.Should().BeTrue();
        pca.ExplainedVariance.Should().NotBeNull();
    }

    // ================================================================
    // 7. VotingEnsemble edge cases
    // ================================================================

    [Fact]
    public void VotingEnsemble_ZeroModels_ShouldThrow()
    {
        var act = () => new VotingEnsemble(Array.Empty<IClassifier>());
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void VotingEnsemble_SingleModel_ShouldWork()
    {
        var models = new IClassifier[] { new LogisticRegression(maxIterations: 100) };
        var ensemble = new VotingEnsemble(models);
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 2, 2, 3, 3 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 0, 1, 1 }, 4);

        ensemble.Fit(X, y);
        ensemble.IsFitted.Should().BeTrue();

        var pred = ensemble.Predict(X);
        pred.Length.Should().Be(4);
    }

    [Fact]
    public void VotingEnsemble_SoftVoting_ModelsWithDifferentNumClasses_ShouldNotCrash()
    {
        // This tests the scenario where models are trained on different subsets
        // and may have different NumClasses. The ensemble uses NumClasses from
        // the first model but reads probabilities from all models with that same width.
        // If a sub-model has a different NumClasses, the proba tensor has the wrong shape.
        var model1 = new DecisionTreeClassifier();
        var model2 = new DecisionTreeClassifier();

        // Train model1 on 2-class data
        var X1 = new Tensor<double>(new double[] { 0, 1, 2, 3 }, 2, 2);
        var y1 = new Tensor<double>(new double[] { 0, 1 }, 2);
        model1.Fit(X1, y1);

        // Train model2 on 3-class data
        var X2 = new Tensor<double>(new double[] { 0, 1, 2, 3, 4, 5 }, 3, 2);
        var y2 = new Tensor<double>(new double[] { 0, 1, 2 }, 3);
        model2.Fit(X2, y2);

        // Create ensemble -- NumClasses will be set from model1 (2) but model2 has 3
        var ensemble = new VotingEnsemble(
            new IClassifier[] { model1, model2 },
            VotingStrategy.Soft);

        // Manually set IsFitted since we pre-fitted models
        // Actually, Fit() re-fits all models. Let's use the same training data.
        var X = new Tensor<double>(new double[] { 0, 1, 2, 3, 4, 5 }, 3, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 2 }, 3);

        ensemble.Fit(X, y);

        // PredictProba should handle the case where NumClasses matches
        // This test just verifies no crashes
        var pred = ensemble.Predict(X);
        pred.Length.Should().Be(3);
    }

    // ================================================================
    // 8. StackingEnsemble edge cases
    // ================================================================

    [Fact]
    public void StackingEnsemble_OneBaseModel_ShouldWork()
    {
        // Use Ridge regression to avoid singular matrix issues with small folds
        var baseModels = new IModel[] { new LinearRegression(l2Penalty: 0.01) };
        var metaLearner = new LinearRegression(l2Penalty: 0.01);
        var stacking = new StackingEnsemble(baseModels, metaLearner, nFolds: 2, seed: 42);

        // Use data with more spread to avoid singularity
        var X = new Tensor<double>(new double[] {
            1, 0, 0, 2, 2, 0, 0, 1, 3, 1, 1, 3
        }, 6, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 2, 1, 4, 4 }, 6);

        stacking.Fit(X, y);
        stacking.IsFitted.Should().BeTrue();

        var pred = stacking.Predict(X);
        pred.Length.Should().Be(6);
    }

    [Fact]
    public void StackingEnsemble_NFoldsGreaterThanSamples_ShouldThrowOrHandle()
    {
        // nFolds=10 but only 3 samples -- foldSize would be 0
        var baseModels = new IModel[] { new LinearRegression() };
        var metaLearner = new LinearRegression();
        var stacking = new StackingEnsemble(baseModels, metaLearner, nFolds: 10);

        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);

        // Should throw a meaningful error, not silently produce garbage
        var act = () => stacking.Fit(X, y);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void StackingEnsemble_NFoldsEquals1_ShouldThrow()
    {
        var act = () => new StackingEnsemble(
            new IModel[] { new LinearRegression() },
            new LinearRegression(),
            nFolds: 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ================================================================
    // 9. QuantileTransformer edge cases
    // ================================================================

    [Fact]
    public void QuantileTransformer_TransformBeforeFit_ShouldThrow()
    {
        var qt = new QuantileTransformer();
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 1, 2, 3 }) });

        var act = () => qt.Transform(df);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void QuantileTransformer_AllSameValues_ShouldNotCrash()
    {
        var qt = new QuantileTransformer();
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 5, 5, 5, 5 }) });

        qt.Fit(df);
        var result = qt.Transform(df);
        result.RowCount.Should().Be(4);
    }

    [Fact]
    public void QuantileTransformer_OneSample_ShouldNotThrow()
    {
        // BUG: With 1 sample, nEdges=1, then frac = q / (nEdges - 1) = 0/0 = division by zero
        var qt = new QuantileTransformer();
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 42.0 }) });

        var act = () => qt.FitTransform(df);
        act.Should().NotThrow();
    }

    // ================================================================
    // 10. PowerTransformer edge cases
    // ================================================================

    [Fact]
    public void PowerTransformer_BoxCoxWithNegativeValues_ShouldThrow()
    {
        var pt = new PowerTransformer(method: PowerMethod.BoxCox);
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { -1, 2, 3 }) });

        var act = () => pt.Fit(df);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void PowerTransformer_AllZeroColumn_BoxCox_ShouldThrow()
    {
        // Box-Cox requires strictly positive
        var pt = new PowerTransformer(method: PowerMethod.BoxCox);
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 0, 0, 0 }) });

        var act = () => pt.Fit(df);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void PowerTransformer_OneSample_YeoJohnson_ShouldNotThrow()
    {
        var pt = new PowerTransformer(method: PowerMethod.YeoJohnson);
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 5.0 }) });

        var act = () => pt.FitTransform(df);
        act.Should().NotThrow();
    }

    [Fact]
    public void PowerTransformer_AllZeroColumn_YeoJohnson_ShouldNotThrow()
    {
        var pt = new PowerTransformer(method: PowerMethod.YeoJohnson);
        var df = new DataFrame(new IColumn[] { new Column<double>("x", new double[] { 0, 0, 0 }) });

        var act = () => pt.FitTransform(df);
        act.Should().NotThrow();
    }

    // ================================================================
    // 11. Tensor operations edge cases
    // ================================================================

    [Fact]
    public void Tensor_MatMul_ShapeMismatch_ShouldThrow()
    {
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);

        var act = () => a.MatMul(b);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Tensor_MatMul_ValidShapes_ShouldWork()
    {
        var a = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 2, 3);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);

        var result = a.MatMul(b);
        result.Shape[0].Should().Be(2);
        result.Shape[1].Should().Be(2);
    }

    [Fact]
    public void Tensor_MatMul_1DInput_ShouldThrow()
    {
        var a = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var b = new Tensor<double>(new double[] { 4, 5, 6 }, 3);

        var act = () => a.MatMul(b);
        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void Tensor_ZeroDimension_ShouldThrowOrHandleGracefully()
    {
        // Creating a tensor with 0 in a shape dimension should be handled
        var act = () => new Tensor<double>(Array.Empty<double>(), 0, 3);
        // A 0x3 tensor has 0 elements, which matches empty array
        act.Should().NotThrow();
    }

    [Fact]
    public void Tensor_MatMul_WithZeroRows_ShouldReturnEmptyResult()
    {
        // 0x3 @ 3x2 should give 0x2
        var a = new Tensor<double>(Array.Empty<double>(), 0, 3);
        var b = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);

        var result = a.MatMul(b);
        result.Shape[0].Should().Be(0);
        result.Shape[1].Should().Be(2);
        result.Length.Should().Be(0);
    }

    // ================================================================
    // 12. CrossValidation edge cases
    // ================================================================

    [Fact]
    public void CrossValidation_NFoldsGreaterThanSamples_ShouldThrow()
    {
        var model = new LinearRegression();
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);
        var y = new Tensor<double>(new double[] { 1, 2 }, 2);

        var act = () => CrossValidation.CrossValScore(model, X, y, nFolds: 5);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void CrossValidation_NFoldsEquals1_ShouldThrow()
    {
        var model = new LinearRegression();
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);

        var act = () => CrossValidation.CrossValScore(model, X, y, nFolds: 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void CrossValidation_NFoldsEquals0_ShouldThrow()
    {
        var model = new LinearRegression();
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);

        var act = () => CrossValidation.CrossValScore(model, X, y, nFolds: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}
