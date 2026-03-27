using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.ML.Models;
using Cortex.ML.ModelSelection;
using Cortex.ML.Tensors;
using Cortex.ML.Transformers;

namespace Cortex.ML.Tests.EdgeCases;

public class MLEdgeCaseRound2Tests
{
    // ================================================================
    // Helper: simple dataset generators
    // ================================================================

    private static (Tensor<double> X, Tensor<double> y) MakeClassificationData(int nSamples = 20, int nFeatures = 2, int nClasses = 2)
    {
        var rng = new Random(42);
        var xData = new double[nSamples * nFeatures];
        var yData = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            int cls = i % nClasses;
            yData[i] = cls;
            for (int j = 0; j < nFeatures; j++)
                xData[i * nFeatures + j] = cls * 3.0 + rng.NextDouble();
        }
        return (new Tensor<double>(xData, nSamples, nFeatures),
                new Tensor<double>(yData, nSamples));
    }

    private static (Tensor<double> X, Tensor<double> y) MakeRegressionData(int nSamples = 20, int nFeatures = 2)
    {
        var rng = new Random(42);
        var xData = new double[nSamples * nFeatures];
        var yData = new double[nSamples];
        for (int i = 0; i < nSamples; i++)
        {
            double sum = 0;
            for (int j = 0; j < nFeatures; j++)
            {
                xData[i * nFeatures + j] = rng.NextDouble() * 10;
                sum += xData[i * nFeatures + j];
            }
            yData[i] = sum + rng.NextDouble();
        }
        return (new Tensor<double>(xData, nSamples, nFeatures),
                new Tensor<double>(yData, nSamples));
    }

    // ================================================================
    // Area 1: Model Serialization round-trip
    // ================================================================

    [Fact]
    public void ModelSerializer_RoundTrip_LinearRegression_WeightsPreserved()
    {
        // LinearRegression has a constructor with an optional parameter (double l2Penalty = 0.0),
        // which is NOT a parameterless constructor at the reflection level.
        // Activator.CreateInstance will fail.
        var (X, y) = MakeRegressionData();
        var model = new LinearRegression(l2Penalty: 0.0);
        model.Fit(X, y);

        var path = Path.Combine(Path.GetTempPath(), $"linreg_test_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            // Bug: Load fails because LinearRegression has no true parameterless constructor
            var loaded = ModelSerializer.Load(path);
            loaded.Should().NotBeNull();
            loaded.Name.Should().Be("LinearRegression");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_LinearRegression_FittedStateRestored()
    {
        // Even if we fix the constructor issue, the fitted state (Weights tensor,
        // IsFitted flag with private set) cannot be restored via property reflection.
        // Weights has type Tensor<double> (namespace Cortex.ML.Tensors) which is
        // explicitly skipped during serialization (line 33 of ModelSerializer).
        // IsFitted has a private setter so it cannot be written back.
        var (X, y) = MakeRegressionData();
        var model = new LinearRegression(l2Penalty: 0.0);
        model.Fit(X, y);
        var originalPredictions = model.Predict(X).ToArray();

        var path = Path.Combine(Path.GetTempPath(), $"linreg_fitted_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            // The loaded model should ideally be fitted and produce same predictions.
            // Bug: IsFitted will be false (private setter not writable), and
            // Weights will be null (Tensor type skipped during serialization).
            loaded.IsFitted.Should().BeTrue("loaded model should preserve fitted state");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_DeserializeInvalidJson_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"invalid_model_{Guid.NewGuid()}.json");
        try
        {
            File.WriteAllText(path, "THIS IS NOT JSON {{{");
            var act = () => ModelSerializer.Load(path);
            act.Should().Throw<Exception>("corrupt JSON should throw during parsing");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_DeserializeMissingTypeName_Throws()
    {
        var path = Path.Combine(Path.GetTempPath(), $"no_type_{Guid.NewGuid()}.json");
        try
        {
            File.WriteAllText(path, """{"version":1,"properties":{}}""");
            var act = () => ModelSerializer.Load(path);
            act.Should().Throw<Exception>("missing typeName should throw");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    // ================================================================
    // Area 2: GridSearchCV edge cases
    // ================================================================

    [Fact]
    public void GridSearchCV_EmptyParameterGrid_StillRuns()
    {
        var (X, y) = MakeRegressionData(nSamples: 30);
        var model = new LinearRegression();
        var grid = new Dictionary<string, object[]>();
        var gs = new GridSearchCV(model, grid, nFolds: 3, seed: 42);

        // Empty grid should produce exactly 1 combination (the default params)
        var fitted = gs.Fit(X, y);
        fitted.BestParams.Should().BeEmpty();
        fitted.AllResults.RowCount.Should().Be(1);
    }

    [Fact]
    public void GridSearchCV_InvalidParameterName_Throws()
    {
        var (X, y) = MakeRegressionData(nSamples: 30);
        var model = new LinearRegression();
        var grid = new Dictionary<string, object[]>
        {
            ["NonExistentParam"] = new object[] { 1.0, 2.0 }
        };
        var gs = new GridSearchCV(model, grid, nFolds: 3, seed: 42);

        var act = () => gs.Fit(X, y);
        act.Should().Throw<ArgumentException>()
            .WithMessage("*NonExistentParam*");
    }

    [Fact]
    public void GridSearchCV_SingleParameterValue_NoSearchNeeded()
    {
        var (X, y) = MakeRegressionData(nSamples: 30);
        var model = new LinearRegression();
        var grid = new Dictionary<string, object[]>
        {
            ["L2Penalty"] = new object[] { 0.5 }
        };
        var gs = new GridSearchCV(model, grid, nFolds: 3, seed: 42);
        var fitted = gs.Fit(X, y);

        fitted.AllResults.RowCount.Should().Be(1);
        fitted.BestParams["L2Penalty"].Should().Be(0.5);
    }

    [Fact]
    public void GridSearchCV_AllResults_BeforeFit_Throws()
    {
        var model = new LinearRegression();
        var grid = new Dictionary<string, object[]>
        {
            ["L2Penalty"] = new object[] { 0.1 }
        };
        var gs = new GridSearchCV(model, grid);

        var act = () => { var _ = gs.AllResults; };
        act.Should().Throw<InvalidOperationException>();
    }

    // ================================================================
    // Area 3: LearningCurve edge cases
    // ================================================================

    [Fact]
    public void LearningCurve_TrainSizeLargerThanDataset_ClampedToAvailable()
    {
        var (X, y) = MakeRegressionData(nSamples: 20);
        var model = new LinearRegression();

        // Request 1000 training samples from a dataset of 20
        // With 3 folds, each fold has ~13 training samples
        // trainSize=1000 should be clamped to available (~13)
        var result = LearningCurve.Compute(model, X, y,
            trainSizes: new double[] { 1000 }, nFolds: 3, seed: 42);

        result.RowCount.Should().Be(1);
        // The effective train_size should be clamped to available training samples
        var trainSizes = (Column<int>)result["train_size"];
        trainSizes.Values[0].Should().BeLessThanOrEqualTo(20);
    }

    [Fact]
    public void LearningCurve_SingleTrainSize_Works()
    {
        var (X, y) = MakeRegressionData(nSamples: 20);
        var model = new LinearRegression();

        var result = LearningCurve.Compute(model, X, y,
            trainSizes: new double[] { 0.5 }, nFolds: 3, seed: 42);

        result.RowCount.Should().Be(1);
    }

    [Fact]
    public void LearningCurve_NFoldsGreaterThanSamples_ShouldFailGracefully()
    {
        // LearningCurve.Compute does NOT check nFolds vs nSamples (unlike CrossValidation).
        // With 5 samples and 10 folds, foldSize = 5/10 = 0, leading to broken fold splitting.
        var (X, y) = MakeRegressionData(nSamples: 5);
        var model = new LinearRegression();

        // This should throw a clear error rather than silently producing garbage
        var act = () => LearningCurve.Compute(model, X, y,
            trainSizes: new double[] { 0.8 }, nFolds: 10, seed: 42);

        act.Should().Throw<Exception>("nFolds > nSamples should be rejected");
    }

    // ================================================================
    // Area 4: RandomForest/GBT edge cases
    // ================================================================

    [Fact]
    public void RandomForestClassifier_NEstimators0_ThrowsAtConstruction()
    {
        // Bug fix: nEstimators=0 previously created an empty _trees array,
        // causing division by zero (NaN probabilities) in PredictProba.
        // Now validates nEstimators >= 1 in the constructor.
        var act = () => new RandomForestClassifier(nEstimators: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void RandomForestRegressor_NEstimators0_ThrowsAtConstruction()
    {
        // Bug fix: Same as classifier -- division by zero in Predict.
        // Now validates nEstimators >= 1 in the constructor.
        var act = () => new RandomForestRegressor(nEstimators: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void RandomForestClassifier_NEstimators1_Works()
    {
        var (X, y) = MakeClassificationData();
        var rf = new RandomForestClassifier(nEstimators: 1, seed: 42);
        rf.Fit(X, y);

        var preds = rf.Predict(X);
        preds.Length.Should().Be(X.Shape[0]);
    }

    [Fact]
    public void GradientBoostedTreeClassifier_NEstimators0_ShouldNotCrash()
    {
        // With nEstimators=0, no trees are fitted but the initial prediction is set.
        // The model reports IsFitted=true (empty list is non-null).
        // Predict should either throw or return initial predictions.
        var (X, y) = MakeClassificationData();
        var gbt = new GradientBoostedTreeClassifier(nEstimators: 0);

        var act = () =>
        {
            gbt.Fit(X, y);
            gbt.Predict(X);
        };

        // Should either throw at construction/fit or handle gracefully (not crash)
        act.Should().NotThrow("GBT with 0 estimators should handle gracefully since IsFitted=true");
    }

    [Fact]
    public void GradientBoostedTreeRegressor_NEstimators0_ShouldNotCrash()
    {
        var (X, y) = MakeRegressionData();
        var gbt = new GradientBoostedTreeRegressor(nEstimators: 0);

        var act = () =>
        {
            gbt.Fit(X, y);
            var preds = gbt.Predict(X);
            // All predictions should equal the initial value (mean of y)
            preds.Length.Should().Be(X.Shape[0]);
        };

        act.Should().NotThrow();
    }

    [Fact]
    public void GradientBoostedTreeRegressor_LearningRate0_AllPredictionsEqualInitial()
    {
        var (X, y) = MakeRegressionData();
        var yArr = y.ToArray();
        double expectedMean = yArr.Average();

        var gbt = new GradientBoostedTreeRegressor(nEstimators: 5, learningRate: 0.0);
        gbt.Fit(X, y);

        var preds = gbt.Predict(X).ToArray();
        // With learningRate=0, all tree contributions are multiplied by 0,
        // so predictions should all equal the initial value (mean)
        foreach (var p in preds)
        {
            p.Should().BeApproximately(expectedMean, 1e-10,
                "with learningRate=0, predictions should all equal the initial mean");
        }
    }

    // ================================================================
    // Area 5: DBSCAN/AgglomerativeClustering edge cases
    // ================================================================

    [Fact]
    public void DBSCAN_Eps0_ConstructorThrows()
    {
        // DBSCAN constructor already validates eps > 0
        var act = () => new DBSCAN(eps: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void DBSCAN_MinSamples1_EveryPointIsCluster()
    {
        // With minSamples=1, every point is a core point (its own neighborhood includes itself)
        var data = new double[] { 0, 0, 10, 10, 20, 20 };
        var X = new Tensor<double>(data, 3, 2);
        var dbscan = new DBSCAN(eps: 1.0, minSamples: 1);
        dbscan.Fit(X);

        // Every point should be in some cluster (no noise)
        dbscan.Labels.Should().NotContain(-1, "with minSamples=1, no point should be noise");
        dbscan.NClusters.Should().Be(3, "3 distant points with small eps should form 3 separate clusters");
    }

    [Fact]
    public void AgglomerativeClustering_NClusters1_EverythingInOneCluster()
    {
        var data = new double[] { 0, 0, 1, 1, 10, 10, 11, 11 };
        var X = new Tensor<double>(data, 4, 2);
        var agg = new AgglomerativeClustering(nClusters: 1);
        agg.Fit(X);

        // All points should be in the same cluster (label 0)
        agg.Labels.Should().NotBeNull();
        agg.Labels!.Distinct().Count().Should().Be(1);
    }

    [Fact]
    public void AgglomerativeClustering_NClustersEqualsSamples_EveryPointAlone()
    {
        var data = new double[] { 0, 0, 1, 1, 10, 10, 11, 11 };
        var X = new Tensor<double>(data, 4, 2);
        var agg = new AgglomerativeClustering(nClusters: 4);
        agg.Fit(X);

        // Each point should be in its own cluster (4 distinct labels)
        agg.Labels.Should().NotBeNull();
        agg.Labels!.Distinct().Count().Should().Be(4);
    }

    // ================================================================
    // Area 6: Transformer pipeline edge cases
    // ================================================================

    [Fact]
    public void StandardScaler_ConstantColumn_StdIsZero_ProducesAllZeros()
    {
        // When std=0, StandardScaler replaces std with 1.0 (line 55).
        // So (x - mean) / 1.0 = 0 for all values since they all equal mean.
        var df = new DataFrame(
            new Column<double>("const_col", new double[] { 5.0, 5.0, 5.0, 5.0, 5.0 }));

        var scaler = new StandardScaler("const_col");
        var result = scaler.FitTransform(df);

        var col = (Column<double>)result["const_col"];
        foreach (var v in col.Values)
        {
            v.Should().BeApproximately(0.0, 1e-10, "constant column should be scaled to all zeros");
        }
    }

    [Fact]
    public void MinMaxScaler_ConstantColumn_MaxEqualsMin_ProducesAllZeros()
    {
        // When max==min, MinMaxScaler sets max = min + 1 (line 40).
        // So (val - min) / (min + 1 - min) = 0 / 1 = 0 for all values.
        var df = new DataFrame(
            new Column<double>("const_col", new double[] { 7.0, 7.0, 7.0 }));

        var scaler = new MinMaxScaler("const_col");
        var result = scaler.FitTransform(df);

        var col = (Column<double>)result["const_col"];
        foreach (var v in col.Values)
        {
            v.Should().BeApproximately(0.0, 1e-10, "constant column with MinMaxScaler should produce zeros");
        }
    }

    [Fact]
    public void OneHotEncoder_UnseenCategory_ErrorMode_Throws()
    {
        var trainDf = new DataFrame(
            new StringColumn("color", new string?[] { "red", "blue", "green" }));

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Error, "color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("color", new string?[] { "red", "yellow" }));

        var act = () => encoder.Transform(testDf);
        act.Should().Throw<InvalidOperationException>().WithMessage("*yellow*");
    }

    [Fact]
    public void OneHotEncoder_UnseenCategory_IgnoreMode_AllZeroRow()
    {
        var trainDf = new DataFrame(
            new StringColumn("color", new string?[] { "red", "blue" }));

        var encoder = new OneHotEncoder(UnknownCategoryHandling.Ignore, "color");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("color", new string?[] { "red", "purple" }));

        var result = encoder.Transform(testDf);

        // "purple" should produce all-zero in the one-hot columns
        var redCol = (Column<int>)result["color_red"];
        var blueCol = (Column<int>)result["color_blue"];

        redCol.Values[0].Should().Be(1); // "red" -> color_red=1
        redCol.Values[1].Should().Be(0); // "purple" -> color_red=0
        blueCol.Values[1].Should().Be(0); // "purple" -> color_blue=0
    }

    [Fact]
    public void LabelEncoder_UnseenCategory_ErrorMode_Throws()
    {
        var trainDf = new DataFrame(
            new StringColumn("fruit", new string?[] { "apple", "banana" }));

        var encoder = new LabelEncoder(UnknownCategoryHandling.Error, "fruit");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("fruit", new string?[] { "apple", "cherry" }));

        var act = () => encoder.Transform(testDf);
        act.Should().Throw<InvalidOperationException>().WithMessage("*cherry*");
    }

    [Fact]
    public void LabelEncoder_UnseenCategory_IgnoreMode_ProducesNull()
    {
        var trainDf = new DataFrame(
            new StringColumn("fruit", new string?[] { "apple", "banana" }));

        var encoder = new LabelEncoder(UnknownCategoryHandling.Ignore, "fruit");
        encoder.Fit(trainDf);

        var testDf = new DataFrame(
            new StringColumn("fruit", new string?[] { "apple", "cherry" }));

        var result = encoder.Transform(testDf);
        var col = result["fruit"];
        col.IsNull(1).Should().BeTrue("unseen category with Ignore mode should produce null");
    }

    // ================================================================
    // Area 7: KDTree edge cases (tested via KNearestNeighborsClassifier)
    // ================================================================

    [Fact]
    public void KNN_DuplicatePoints_AllSameCoordinates()
    {
        // All training points have the same coordinates but different labels
        var xData = new double[] { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
        var yData = new double[] { 0, 1, 0 };
        var X = new Tensor<double>(xData, 3, 2);
        var y = new Tensor<double>(yData, 3);

        var knn = new KNearestNeighborsClassifier(k: 3);
        knn.Fit(X, y);

        var query = new Tensor<double>(new double[] { 1.0, 1.0 }, 1, 2);
        var pred = knn.Predict(query);

        // All 3 neighbors have same distance; majority class is 0
        pred.ToArray()[0].Should().Be(0);
    }

    [Fact]
    public void KNN_SingleTrainingPoint()
    {
        var X = new Tensor<double>(new double[] { 5.0, 5.0 }, 1, 2);
        var y = new Tensor<double>(new double[] { 1.0 }, 1);

        var knn = new KNearestNeighborsClassifier(k: 1);
        knn.Fit(X, y);

        var query = new Tensor<double>(new double[] { 0.0, 0.0 }, 1, 2);
        var pred = knn.Predict(query);
        pred.ToArray()[0].Should().Be(1.0, "only training point has label 1");
    }

    [Fact]
    public void KNN_HighDimensionality_FallsBackToBruteForce()
    {
        // With d=100 (>20), KNN should skip KDTree and use brute force
        int d = 100;
        int nTrain = 10;
        var rng = new Random(42);
        var xData = new double[nTrain * d];
        var yData = new double[nTrain];
        for (int i = 0; i < nTrain; i++)
        {
            yData[i] = i % 2;
            for (int j = 0; j < d; j++)
                xData[i * d + j] = rng.NextDouble() + yData[i] * 5;
        }

        var X = new Tensor<double>(xData, nTrain, d);
        var y = new Tensor<double>(yData, nTrain);

        var knn = new KNearestNeighborsClassifier(k: 3);
        knn.Fit(X, y);

        var queryData = new double[d];
        for (int j = 0; j < d; j++) queryData[j] = 0.5;
        var query = new Tensor<double>(queryData, 1, d);

        var act = () => knn.Predict(query);
        act.Should().NotThrow("high-D KNN should fall back to brute force without error");
    }
}
