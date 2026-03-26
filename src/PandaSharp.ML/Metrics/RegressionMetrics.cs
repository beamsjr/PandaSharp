using PandaSharp.ML.Tensors;

namespace PandaSharp.ML.Metrics;

/// <summary>
/// Regression metrics computed directly on Tensor&lt;double&gt; predictions.
/// Complements <see cref="MetricsCalculator"/> which operates on PandaSharp Column types.
/// </summary>
public static class RegressionMetrics
{
    /// <summary>Mean Squared Error: average of (y_true - y_pred)².</summary>
    public static double MSE(Tensor<double> yTrue, Tensor<double> yPred)
    {
        ValidateSameLength(yTrue, yPred);
        int n = yTrue.Length;
        double sum = 0;
        var trueSpan = yTrue.Span;
        var predSpan = yPred.Span;
        for (int i = 0; i < n; i++)
        {
            double err = trueSpan[i] - predSpan[i];
            sum += err * err;
        }
        return sum / n;
    }

    /// <summary>Root Mean Squared Error: sqrt(MSE).</summary>
    public static double RMSE(Tensor<double> yTrue, Tensor<double> yPred)
        => Math.Sqrt(MSE(yTrue, yPred));

    /// <summary>Mean Absolute Error: average of |y_true - y_pred|.</summary>
    public static double MAE(Tensor<double> yTrue, Tensor<double> yPred)
    {
        ValidateSameLength(yTrue, yPred);
        int n = yTrue.Length;
        double sum = 0;
        var trueSpan = yTrue.Span;
        var predSpan = yPred.Span;
        for (int i = 0; i < n; i++)
            sum += Math.Abs(trueSpan[i] - predSpan[i]);
        return sum / n;
    }

    /// <summary>
    /// R² (coefficient of determination): 1 - SS_res / SS_tot.
    /// Returns 1.0 for a perfect model, 0.0 for a model that predicts the mean, and negative for worse.
    /// </summary>
    public static double R2(Tensor<double> yTrue, Tensor<double> yPred)
    {
        ValidateSameLength(yTrue, yPred);
        int n = yTrue.Length;
        var trueSpan = yTrue.Span;
        var predSpan = yPred.Span;

        double meanTrue = 0;
        for (int i = 0; i < n; i++)
            meanTrue += trueSpan[i];
        meanTrue /= n;

        double ssRes = 0, ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double err = trueSpan[i] - predSpan[i];
            ssRes += err * err;
            double diff = trueSpan[i] - meanTrue;
            ssTot += diff * diff;
        }

        return ssTot > 0 ? 1.0 - ssRes / ssTot : 0.0;
    }

    /// <summary>
    /// Mean Absolute Percentage Error: average of |y_true - y_pred| / |y_true|.
    /// Samples where y_true == 0 are skipped to avoid division by zero.
    /// </summary>
    public static double MAPE(Tensor<double> yTrue, Tensor<double> yPred)
    {
        ValidateSameLength(yTrue, yPred);
        int n = yTrue.Length;
        var trueSpan = yTrue.Span;
        var predSpan = yPred.Span;

        double sum = 0;
        int count = 0;
        for (int i = 0; i < n; i++)
        {
            if (trueSpan[i] == 0) continue;
            sum += Math.Abs((trueSpan[i] - predSpan[i]) / trueSpan[i]);
            count++;
        }

        return count > 0 ? sum / count : 0.0;
    }

    private static void ValidateSameLength(Tensor<double> a, Tensor<double> b)
    {
        if (a.Length != b.Length)
            throw new ArgumentException($"Length mismatch: yTrue has {a.Length} elements, yPred has {b.Length}.");
    }
}
