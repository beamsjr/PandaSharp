using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.ML.Metrics;

namespace PandaSharp.ML.Tests;

public class FeatureImportanceTests
{
    [Fact]
    public void PermutationImportance_IdentifiesImportantFeature()
    {
        // Create data where F1 is informative and F2 is noise
        var df = new DataFrame(
            new Column<double>("F1", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
            new Column<double>("F2", [5, 3, 8, 1, 9, 2, 7, 4, 6, 10]),
            new Column<double>("Target", [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]) // Target = 2 * F1
        );

        // Simple scorer: R² of predicting Target as 2*F1
        double Scorer(DataFrame d)
        {
            var f1 = d.GetColumn<double>("F1");
            var target = d.GetColumn<double>("Target");
            double ssTot = 0, ssRes = 0;
            double mean = 0;
            for (int i = 0; i < d.RowCount; i++) mean += target[i]!.Value;
            mean /= d.RowCount;
            for (int i = 0; i < d.RowCount; i++)
            {
                double pred = f1[i]!.Value * 2;
                ssRes += (target[i]!.Value - pred) * (target[i]!.Value - pred);
                ssTot += (target[i]!.Value - mean) * (target[i]!.Value - mean);
            }
            return ssTot > 0 ? 1 - ssRes / ssTot : 0;
        }

        var result = FeatureImportance.PermutationImportance(
            df, Scorer, ["F1", "F2"], nRepeats: 10, seed: 42);

        result.RowCount.Should().Be(2);
        result.ColumnNames.Should().Equal(["Feature", "Importance", "StdDev"]);

        // F1 should be more important than F2
        result.GetStringColumn("Feature")[0].Should().Be("F1");
        result.GetColumn<double>("Importance")[0]!.Value.Should().BeGreaterThan(
            result.GetColumn<double>("Importance")[1]!.Value);
    }

    [Fact]
    public void PermutationImportance_AllFeaturesNoisy()
    {
        var df = new DataFrame(
            new Column<double>("A", [1, 2, 3, 4, 5]),
            new Column<double>("B", [5, 4, 3, 2, 1]),
            new Column<double>("Y", [10, 10, 10, 10, 10]) // constant target
        );

        // Constant prediction → shuffling any feature has no effect
        double Scorer(DataFrame d) => 1.0; // always perfect

        var result = FeatureImportance.PermutationImportance(
            df, Scorer, ["A", "B"], nRepeats: 3, seed: 42);

        // Both features should have ~0 importance
        result.GetColumn<double>("Importance")[0].Should().BeApproximately(0, 0.01);
        result.GetColumn<double>("Importance")[1].Should().BeApproximately(0, 0.01);
    }

    [Fact]
    public void PermutationImportance_SortedByImportance()
    {
        var df = new DataFrame(
            new Column<double>("X1", [1, 2, 3, 4, 5]),
            new Column<double>("X2", [10, 20, 30, 40, 50]),
            new Column<double>("X3", [100, 200, 300, 400, 500]),
            new Column<double>("Y", [1, 2, 3, 4, 5])
        );

        double Scorer(DataFrame d)
        {
            // Score based on correlation with X1
            var x1 = d.GetColumn<double>("X1");
            var y = d.GetColumn<double>("Y");
            double corr = 0;
            for (int i = 0; i < d.RowCount; i++)
                corr += x1[i]!.Value * y[i]!.Value;
            return corr;
        }

        var result = FeatureImportance.PermutationImportance(
            df, Scorer, ["X1", "X2", "X3"], nRepeats: 5, seed: 42);

        // Results should be sorted by importance descending
        var imp = result.GetColumn<double>("Importance");
        for (int i = 1; i < result.RowCount; i++)
            imp[i]!.Value.Should().BeLessThanOrEqualTo(imp[i - 1]!.Value);
    }
}
