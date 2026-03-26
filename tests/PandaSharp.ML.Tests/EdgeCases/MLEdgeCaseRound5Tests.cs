using FluentAssertions;
using PandaSharp.Column;
using PandaSharp.ML.Metrics;
using PandaSharp.ML.Models;
using PandaSharp.ML.Tensors;
using Xunit;

namespace PandaSharp.ML.Tests.EdgeCases;

public class MLEdgeCaseRound5Tests
{
    // ═══════════════════════════════════════════════════════════════
    // BUG 1: ModelSerializer.Load does NOT restore IsFitted state
    // for models with private setters on IsFitted.
    // The serializer saves IsFitted=true in the envelope, but on
    // deserialization it only sets public writable properties.
    // IsFitted has { get; private set; } on LinearRegression,
    // LogisticRegression, SGDClassifier, SGDRegressor, ElasticNet,
    // and KMeans. After Load(), IsFitted remains false and Predict
    // throws InvalidOperationException.
    //
    // Fix: After restoring properties, check the envelope's
    // "isFitted" flag and use reflection to set the backing field
    // or property even if it has a private setter.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ModelSerializer_RoundTrip_LinearRegression_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new LinearRegression();
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"lr_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            // BUG: IsFitted is false after deserialization because it has private set
            loaded.IsFitted.Should().BeTrue(
                "deserialized model should report IsFitted=true when the saved model was fitted");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_LogisticRegression_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new LogisticRegression(maxIterations: 100);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"logreg_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            loaded.IsFitted.Should().BeTrue(
                "deserialized LogisticRegression should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_SGDClassifier_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new SGDClassifier(maxEpochs: 50, loss: ClassificationLoss.Log);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"sgdc_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            loaded.IsFitted.Should().BeTrue(
                "deserialized SGDClassifier should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_SGDRegressor_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new SGDRegressor(maxEpochs: 50);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"sgdr_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            loaded.IsFitted.Should().BeTrue(
                "deserialized SGDRegressor should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_ElasticNet_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new ElasticNet(alpha: 0.1);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"enet_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            loaded.IsFitted.Should().BeTrue(
                "deserialized ElasticNet should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 2: KMeans cannot be deserialized at all because its
    // constructor requires a non-optional `nClusters` parameter.
    // Activator.CreateInstance fails, and the fallback looks for
    // a constructor where ALL parameters have default values,
    // but nClusters has no default. Load() throws
    // InvalidOperationException.
    //
    // Fix: In the fallback, also try constructors where we can
    // supply parameter values from the serialized properties.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ModelSerializer_RoundTrip_KMeans_CanDeserialize()
    {
        var X = new Tensor<double>(new double[]
        {
            0, 0,  0, 1,  1, 0,  1, 1,
            10, 10,  10, 11,  11, 10,  11, 11
        }, 8, 2);
        var y = new Tensor<double>(new double[8], 8); // dummy
        var model = new KMeans(nClusters: 2, seed: 42, nInit: 1);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"kmeans_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);

            // BUG: This throws because KMeans has no parameterless constructor
            // and nClusters has no default value
            var act = () => ModelSerializer.Load(path);
            act.Should().NotThrow(
                "ModelSerializer.Load should handle constructors with required parameters " +
                "by supplying values from the serialized properties");
        }
        finally
        {
            File.Delete(path);
        }
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 3: SGDClassifier with maxEpochs=0 reports IsFitted=true
    // but leaves _classes as null (set before the epoch loop, but
    // actually it IS set). Wait - let me re-check. Actually classes
    // ARE set before the epoch loop. The FitBinary loop just doesn't
    // run, weights stay zero. PredictProba divides by sum which is
    // ~1.0 for binary. So it should work. Let me verify.
    //
    // Actually the real issue: SGDClassifier.Fit with maxEpochs=0
    // still sets _classes and _weights correctly (zero weights).
    // However, the weights array has pAug elements all zero.
    // PredictProba computes z=0 for all samples, sigmoid(0)=0.5.
    // For binary: P(class0)=0.5, P(class1)=0.5, argmax picks class
    // index 0 (class with lower label value). This is valid.
    // No bug here - just an edge case that works.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void SGDClassifier_MaxEpochs0_FitsWithoutError()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new SGDClassifier(maxEpochs: 0, loss: ClassificationLoss.Log);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        // With zero epochs, all weights are zero, sigmoid(0)=0.5
        // Should not throw
        var preds = model.Predict(X);
        preds.Length.Should().Be(4);
    }

    [Fact]
    public void SGDClassifier_LearningRate0_NoLearning_StillPredicts()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new SGDClassifier(
            eta0: 0.0,
            maxEpochs: 10,
            schedule: LearningRateSchedule.Constant,
            loss: ClassificationLoss.Log);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        // With eta=0, weights never change from zero
        var preds = model.Predict(X);
        preds.Length.Should().Be(4);
        // All predictions should be the same (class 0 since both probabilities are 0.5
        // and argmax picks index 0)
        var span = preds.Span;
        for (int i = 1; i < 4; i++)
            span[i].Should().Be(span[0], "with zero learning rate all predictions should be identical");
    }

    [Fact]
    public void SGDRegressor_AllZeroFeatures_Predicts()
    {
        var X = new Tensor<double>(new double[6], 3, 2); // all zeros
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new SGDRegressor(maxEpochs: 100);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var preds = model.Predict(X);
        preds.Length.Should().Be(3);
        // With all zero features, only the intercept matters
        // All predictions should be the same
        var span = preds.Span;
        for (int i = 1; i < 3; i++)
            Math.Abs(span[i] - span[0]).Should().BeLessThan(1e-6,
                "with all-zero features, predictions depend only on intercept");
    }

    [Fact]
    public void SGDClassifier_BatchSizeGreaterThanNSamples()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        // batchSize=100 > nSamples=4
        var model = new SGDClassifier(batchSize: 100, maxEpochs: 50, loss: ClassificationLoss.Log);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var preds = model.Predict(X);
        preds.Length.Should().Be(4);
    }

    [Fact]
    public void SGDClassifier_BatchSize1_PureStochastic()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new SGDClassifier(batchSize: 1, maxEpochs: 200, loss: ClassificationLoss.Log);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var preds = model.Predict(X);
        preds.Length.Should().Be(4);
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 4 (area 3): ElasticNet with alpha=0 should behave like
    // OLS LinearRegression. This works correctly - no bug.
    // But: Lasso with very large alpha should zero out weights.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ElasticNet_Alpha0_BehavesLikeOLS()
    {
        // Use enough samples (10) to ensure well-conditioned normal equations on all platforms
        var X = new Tensor<double>(new double[] {
            1, 2,  2, 1,  3, 3,  4, 2,  5, 1,
            1, 5,  2, 4,  3, 2,  4, 3,  5, 5
        }, 10, 2);
        var y = new Tensor<double>(new double[] { 5, 4, 9, 8, 7, 11, 10, 8, 10, 15 }, 10);

        var ols = new LinearRegression();
        ols.Fit(X, y);

        var enet = new ElasticNet(alpha: 0.0, l1Ratio: 0.5);
        enet.Fit(X, y);

        var olsPreds = ols.Predict(X);
        var enetPreds = enet.Predict(X);

        for (int i = 0; i < 10; i++)
            Math.Abs(olsPreds.Span[i] - enetPreds.Span[i]).Should().BeLessThan(0.5,
                "ElasticNet with alpha=0 should produce predictions close to OLS");
    }

    [Fact]
    public void Lasso_VeryLargeAlpha_ZerosOutWeights()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);
        var model = new Lasso(alpha: 1e6);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        // With enormous alpha, all weights should be driven to zero
        var w = model.Weights!.Span;
        for (int i = 0; i < w.Length; i++)
            Math.Abs(w[i]).Should().BeLessThan(1e-6,
                "very large L1 penalty should shrink all weights to zero");
    }

    [Fact]
    public void ElasticNet_L1Ratio0_PureRidge()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 5, 11, 17 }, 3);

        var enet = new ElasticNet(alpha: 0.1, l1Ratio: 0.0); // pure Ridge
        enet.Fit(X, y);
        enet.IsFitted.Should().BeTrue();

        // Ridge (L2) should not zero out weights, just shrink them
        var w = enet.Weights!.Span;
        bool anyNonZero = false;
        for (int i = 0; i < w.Length; i++)
            if (Math.Abs(w[i]) > 1e-6) anyNonZero = true;
        anyNonZero.Should().BeTrue("Ridge regression should not zero out weights entirely");
    }

    [Fact]
    public void ElasticNet_L1Ratio1_PureLasso()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4, 5, 6 }, 3, 2);
        var y = new Tensor<double>(new double[] { 1, 2, 3 }, 3);

        var enet = new ElasticNet(alpha: 1.0, l1Ratio: 1.0); // pure Lasso
        enet.Fit(X, y);
        enet.IsFitted.Should().BeTrue();

        // Should produce sparse solution (some weights near zero)
        var preds = enet.Predict(X);
        preds.Length.Should().Be(3);
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 5 (area 4): TSNE with perplexity > nSamples/3.
    // TSNE only checks perplexity >= n, not perplexity > n/3.
    // With 4 samples and perplexity=30, it correctly throws.
    // With 4 samples and perplexity=3 (> 4/3 but < 4), it runs
    // but the perplexity binary search may not converge well.
    // Not a code bug per se, just an edge case.
    //
    // TSNE with n=2: throws "Perplexity (30) must be less than
    // n_samples (2)" - correct behavior.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void TSNE_2Samples_DefaultPerplexity_Throws()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);
        // Default perplexity is 30, which is >= 2 samples
        var tsne = new TSNE(seed: 42);
        var act = () => tsne.Fit(X);
        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void TSNE_2Samples_Perplexity1_Runs()
    {
        var X = new Tensor<double>(new double[] { 1, 2, 3, 4 }, 2, 2);
        var tsne = new TSNE(perplexity: 1.0, maxIterations: 10, seed: 42);
        var embedding = tsne.FitTransform(X);
        embedding.Shape[0].Should().Be(2);
        embedding.Shape[1].Should().Be(2);
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 6 (area 4): UMAP with nNeighbors > nSamples.
    // UMAP.Fit correctly clamps k = Math.Min(NNeighbors, n - 1),
    // so this should work. Let's verify.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void UMAP_NNeighborsGreaterThanNSamples_Works()
    {
        var X = new Tensor<double>(new double[]
        {
            0, 0,  1, 1,  2, 2,  3, 3,  4, 4
        }, 5, 2);
        // nNeighbors=100 > nSamples=5, should clamp internally
        var umap = new UMAP(nNeighbors: 100, seed: 42);
        var embedding = umap.FitTransform(X);
        embedding.Shape[0].Should().Be(5);
        embedding.Shape[1].Should().Be(2);
    }

    [Fact]
    public void UMAP_NComponentsGreaterThanNFeatures_Works()
    {
        var X = new Tensor<double>(new double[]
        {
            0, 0,  1, 1,  2, 2,  3, 3,  4, 4
        }, 5, 2);
        // nComponents=5 > nFeatures=2
        var umap = new UMAP(nComponents: 5, nNeighbors: 3, seed: 42);
        var embedding = umap.FitTransform(X);
        embedding.Shape[0].Should().Be(5);
        embedding.Shape[1].Should().Be(5);
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 7 (area 5): KMeans.Predict on very distant data.
    // Should still assign to nearest centroid without error.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void KMeans_Predict_VeryDistantData_AssignsCluster()
    {
        var X = new Tensor<double>(new double[]
        {
            0, 0,  0, 1,  1, 0,  1, 1,
            10, 10,  10, 11,  11, 10,  11, 11
        }, 8, 2);
        var y = new Tensor<double>(new double[8], 8);
        var model = new KMeans(nClusters: 2, seed: 42, nInit: 1);
        model.Fit(X, y);

        // Points very far from all centroids
        var XFar = new Tensor<double>(new double[] { 1e6, 1e6, -1e6, -1e6 }, 2, 2);
        var preds = model.Predict(XFar);
        preds.Length.Should().Be(2);
        // Both should get valid cluster labels (0 or 1)
        var span = preds.Span;
        span[0].Should().BeOneOf(0.0, 1.0);
        span[1].Should().BeOneOf(0.0, 1.0);
    }

    [Fact]
    public void KMeans_NInit1_Works()
    {
        var X = new Tensor<double>(new double[]
        {
            0, 0,  0, 1,  1, 0,
            10, 10,  10, 11,  11, 10
        }, 6, 2);
        var y = new Tensor<double>(new double[6], 6);
        var model = new KMeans(nClusters: 2, seed: 42, nInit: 1);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();
    }

    [Fact]
    public void KMeans_NInit0_Throws()
    {
        var act = () => new KMeans(nClusters: 2, nInit: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    // ═══════════════════════════════════════════════════════════════
    // DBSCAN is transductive — no Predict method exposed.
    // Verify it only has Fit, not Predict/IModel.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void DBSCAN_HasNoPredict_IsNotIModel()
    {
        // DBSCAN does not implement IModel, so no Predict method
        var dbscan = new DBSCAN(eps: 1.0, minSamples: 2);
        (dbscan is IModel).Should().BeFalse("DBSCAN is transductive and should not implement IModel");
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 8 (area 6): MultiClassResult.ToDataFrame with many
    // classes - labels should match. This works correctly.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void MultiClass_10PlusClasses_ToDataFrame_CorrectLabels()
    {
        // Create 12-class classification problem
        int n = 12;
        var trueLabels = new int[n];
        var predLabels = new int[n];
        for (int i = 0; i < n; i++)
        {
            trueLabels[i] = i; // one sample per class
            predLabels[i] = i; // perfect predictions
        }

        var yTrue = new Column<int>("true", trueLabels);
        var yPred = new Column<int>("pred", predLabels);
        var result = MetricsCalculator.MultiClass(yTrue, yPred);

        result.Classes.Length.Should().Be(12);
        result.Accuracy.Should().Be(1.0);

        var df = result.ToDataFrame();
        df.ColumnCount.Should().Be(12, "one column per class");

        // Verify column names match class labels
        for (int i = 0; i < 12; i++)
            df.ColumnNames[i].Should().Be(i.ToString());
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 9 (area 6): Binary classification where all predictions
    // are one class — TP=0, precision=0/0 handled as 0, F1=0.
    // The code handles this correctly (precision+recall > 0 check).
    // But let's verify accuracy is correct.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void BinaryClassification_AllPredictionsOneClass_MetricsCorrect()
    {
        // True: [T, T, T, F, F] — 3 positive, 2 negative
        // Pred: [F, F, F, F, F] — all negative
        var yTrue = new Column<bool>("true", new[] { true, true, true, false, false });
        var yPred = new Column<bool>("pred", new[] { false, false, false, false, false });
        var result = MetricsCalculator.Classification(yTrue, yPred);

        result.TruePositive.Should().Be(0);
        result.FalseNegative.Should().Be(3);
        result.TrueNegative.Should().Be(2);
        result.FalsePositive.Should().Be(0);

        // Precision = 0/(0+0) = 0
        result.Precision.Should().Be(0.0);
        // Recall = 0/(0+3) = 0
        result.Recall.Should().Be(0.0);
        // F1 = 0
        result.F1.Should().Be(0.0);
        // Accuracy = 2/5 = 0.4
        result.Accuracy.Should().BeApproximately(0.4, 1e-9);
    }

    [Fact]
    public void BinaryClassification_AllPositivePredictions_MetricsCorrect()
    {
        // True: [T, F, F, F, F] — 1 positive, 4 negative
        // Pred: [T, T, T, T, T] — all positive
        var yTrue = new Column<bool>("true", new[] { true, false, false, false, false });
        var yPred = new Column<bool>("pred", new[] { true, true, true, true, true });
        var result = MetricsCalculator.Classification(yTrue, yPred);

        result.TruePositive.Should().Be(1);
        result.FalsePositive.Should().Be(4);
        result.TrueNegative.Should().Be(0);
        result.FalseNegative.Should().Be(0);

        // Precision = 1/5 = 0.2
        result.Precision.Should().BeApproximately(0.2, 1e-9);
        // Recall = 1/1 = 1.0
        result.Recall.Should().BeApproximately(1.0, 1e-9);
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 10 (area 7): Pipeline deserialization is NOT implemented.
    // PipelineSerialization.Deserialize throws NotImplementedException.
    // This is a documented limitation, but let's verify.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void PipelineSerialization_Deserialize_ThrowsNotImplemented()
    {
        var act = () => PandaSharp.ML.Pipeline.PipelineSerialization.Deserialize(Array.Empty<byte>());
        act.Should().Throw<NotImplementedException>();
    }

    // ═══════════════════════════════════════════════════════════════
    // BUG 11: DecisionTreeClassifier deserialization creates an
    // instance with IsFitted => _root is not null, but _root is
    // a private field that is never serialized by ModelSerializer.
    // So IsFitted will always be false after deserialization.
    // ═══════════════════════════════════════════════════════════════

    [Fact]
    public void ModelSerializer_RoundTrip_DecisionTreeClassifier_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new DecisionTreeClassifier();
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"dtc_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            // BUG: DecisionTreeClassifier.IsFitted is computed from _root is not null
            // and _root is never serialized, so loaded.IsFitted will be false
            loaded.IsFitted.Should().BeTrue(
                "deserialized DecisionTreeClassifier should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void ModelSerializer_RoundTrip_RandomForestClassifier_RestoresIsFitted()
    {
        var X = new Tensor<double>(new double[] { 0, 0, 1, 1, 0, 1, 1, 0 }, 4, 2);
        var y = new Tensor<double>(new double[] { 0, 1, 1, 0 }, 4);
        var model = new RandomForestClassifier(nEstimators: 3, seed: 42);
        model.Fit(X, y);
        model.IsFitted.Should().BeTrue();

        var path = Path.Combine(Path.GetTempPath(), $"rfc_round5_{Guid.NewGuid()}.json");
        try
        {
            ModelSerializer.Save(model, path);
            var loaded = ModelSerializer.Load(path);

            // BUG: RandomForestClassifier.IsFitted is computed from _trees is not null
            // and _trees is never serialized
            loaded.IsFitted.Should().BeTrue(
                "deserialized RandomForestClassifier should report IsFitted=true");
        }
        finally
        {
            File.Delete(path);
        }
    }
}
