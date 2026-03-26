using PandaSharp.Column;

namespace PandaSharp.ML.Metrics;

public record ClassificationResult(
    double Accuracy, double Precision, double Recall, double F1,
    int[,] ConfusionMatrix, int TruePositive, int FalsePositive, int TrueNegative, int FalseNegative)
{
    public override string ToString() =>
        $"Accuracy: {Accuracy:P2}  Precision: {Precision:P2}  Recall: {Recall:P2}  F1: {F1:P2}\n" +
        $"TP: {TruePositive}  FP: {FalsePositive}  TN: {TrueNegative}  FN: {FalseNegative}";

    /// <summary>
    /// Convert the confusion matrix to a labeled DataFrame.
    /// Rows = actual class, columns = predicted class. Labels are "0" and "1" for binary.
    /// </summary>
    public DataFrame ToDataFrame()
    {
        int size = ConfusionMatrix.GetLength(0);
        var columns = new List<IColumn>(size);
        for (int col = 0; col < size; col++)
        {
            var values = new int[size];
            for (int row = 0; row < size; row++)
                values[row] = ConfusionMatrix[row, col];
            columns.Add(new Column<int>(col.ToString(), values));
        }
        return new DataFrame(columns);
    }
}

public record RegressionResult(double MSE, double RMSE, double MAE, double R2, double MAPE)
{
    public override string ToString() =>
        $"MSE: {MSE:F4}  RMSE: {RMSE:F4}  MAE: {MAE:F4}  R²: {R2:F4}  MAPE: {MAPE:P2}";
}

public static class MetricsCalculator
{
    /// <summary>Binary classification metrics.</summary>
    public static ClassificationResult Classification(Column<bool> yTrue, Column<bool> yPred)
    {
        int tp = 0, fp = 0, tn = 0, fn = 0;
        for (int i = 0; i < yTrue.Length; i++)
        {
            bool actual = yTrue[i] ?? false;
            bool predicted = yPred[i] ?? false;
            if (actual && predicted) tp++;
            else if (!actual && predicted) fp++;
            else if (!actual && !predicted) tn++;
            else fn++;
        }

        double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        double accuracy = (double)(tp + tn) / yTrue.Length;

        var cm = new int[2, 2] { { tn, fp }, { fn, tp } };

        return new ClassificationResult(accuracy, precision, recall, f1, cm, tp, fp, tn, fn);
    }

    /// <summary>Classification from int columns (0/1).</summary>
    public static ClassificationResult Classification(Column<int> yTrue, Column<int> yPred)
    {
        int n = yTrue.Length;
        int tp = 0, fp = 0, tn = 0, fn = 0;
        var trueVals = yTrue.Values;
        var predVals = yPred.Values;
        for (int i = 0; i < n; i++)
        {
            bool actual = trueVals[i] == 1;
            bool predicted = predVals[i] == 1;
            if (actual && predicted) tp++;
            else if (!actual && predicted) fp++;
            else if (!actual && !predicted) tn++;
            else fn++;
        }

        double precision = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
        double recall = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
        double f1 = precision + recall > 0 ? 2 * precision * recall / (precision + recall) : 0;
        double accuracy = (double)(tp + tn) / n;
        var cm = new int[2, 2] { { tn, fp }, { fn, tp } };
        return new ClassificationResult(accuracy, precision, recall, f1, cm, tp, fp, tn, fn);
    }

    /// <summary>Regression metrics.</summary>
    public static RegressionResult Regression(Column<double> yTrue, Column<double> yPred)
    {
        int n = yTrue.Length;
        double sumSqErr = 0, sumAbsErr = 0, sumAbsPctErr = 0;
        double meanTrue = 0;

        for (int i = 0; i < n; i++)
            meanTrue += yTrue[i] ?? 0;
        meanTrue /= n;

        double ssTot = 0;
        for (int i = 0; i < n; i++)
        {
            double actual = yTrue[i] ?? 0;
            double predicted = yPred[i] ?? 0;
            double err = actual - predicted;
            sumSqErr += err * err;
            sumAbsErr += Math.Abs(err);
            if (actual != 0) sumAbsPctErr += Math.Abs(err / actual);
            ssTot += (actual - meanTrue) * (actual - meanTrue);
        }

        double mse = sumSqErr / n;
        double r2 = ssTot > 0 ? 1 - sumSqErr / ssTot : (sumSqErr == 0 ? 1.0 : 0.0);

        return new RegressionResult(
            MSE: mse,
            RMSE: Math.Sqrt(mse),
            MAE: sumAbsErr / n,
            R2: r2,
            MAPE: sumAbsPctErr / n
        );
    }

    /// <summary>
    /// Multi-class classification metrics from integer label columns.
    /// Returns per-class precision/recall/f1 and macro/weighted averages.
    /// Throws <see cref="ArgumentException"/> if any null values are found in the input columns.
    /// </summary>
    /// <exception cref="ArgumentException">Thrown when yTrue or yPred contain null values.</exception>
    public static MultiClassResult MultiClass(Column<int> yTrue, Column<int> yPred)
    {
        int n = yTrue.Length;

        // Validate: null values in classification labels are likely a data error
        for (int i = 0; i < n; i++)
        {
            if (!yTrue[i].HasValue)
                throw new ArgumentException(
                    $"yTrue contains a null value at index {i}. " +
                    "Multi-class metrics require non-null integer labels. " +
                    "Use DropNa() or FillNa() to handle missing values before computing metrics.");
            if (!yPred[i].HasValue)
                throw new ArgumentException(
                    $"yPred contains a null value at index {i}. " +
                    "Multi-class metrics require non-null integer labels. " +
                    "Use DropNa() or FillNa() to handle missing values before computing metrics.");
        }

        var classes = new HashSet<int>();
        for (int i = 0; i < n; i++)
        {
            classes.Add(yTrue[i]!.Value);
            classes.Add(yPred[i]!.Value);
        }
        var sortedClasses = classes.OrderBy(c => c).ToArray();
        int nClasses = sortedClasses.Length;
        var classIndex = new Dictionary<int, int>();
        for (int i = 0; i < nClasses; i++) classIndex[sortedClasses[i]] = i;

        // Build confusion matrix
        var cm = new int[nClasses, nClasses];
        int correct = 0;
        for (int i = 0; i < n; i++)
        {
            int actual = yTrue[i]!.Value;
            int predicted = yPred[i]!.Value;
            cm[classIndex[actual], classIndex[predicted]]++;
            if (actual == predicted) correct++;
        }

        // Per-class metrics
        var precision = new double[nClasses];
        var recall = new double[nClasses];
        var f1 = new double[nClasses];
        var support = new int[nClasses];

        for (int c = 0; c < nClasses; c++)
        {
            int tp = cm[c, c];
            int fp = 0, fn = 0;
            for (int i = 0; i < nClasses; i++)
            {
                if (i != c) { fp += cm[i, c]; fn += cm[c, i]; }
            }
            support[c] = tp + fn;
            precision[c] = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
            recall[c] = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
            f1[c] = precision[c] + recall[c] > 0 ? 2 * precision[c] * recall[c] / (precision[c] + recall[c]) : 0;
        }

        double accuracy = (double)correct / n;
        double macroPrecision = precision.Average();
        double macroRecall = recall.Average();
        double macroF1 = f1.Average();

        int totalSupport = support.Sum();
        double weightedF1 = totalSupport > 0
            ? Enumerable.Range(0, nClasses).Sum(c => f1[c] * support[c]) / totalSupport
            : 0;

        return new MultiClassResult(
            accuracy, macroPrecision, macroRecall, macroF1, weightedF1,
            sortedClasses, precision, recall, f1, support, cm);
    }
}

public record MultiClassResult(
    double Accuracy, double MacroPrecision, double MacroRecall, double MacroF1, double WeightedF1,
    int[] Classes, double[] Precision, double[] Recall, double[] F1, int[] Support, int[,] ConfusionMatrix)
{
    public override string ToString()
    {
        var sb = new System.Text.StringBuilder();
        sb.AppendLine($"Accuracy: {Accuracy:P2}  Macro F1: {MacroF1:F4}  Weighted F1: {WeightedF1:F4}");
        sb.AppendLine($"{"Class",8} {"Precision",10} {"Recall",8} {"F1",8} {"Support",8}");
        for (int i = 0; i < Classes.Length; i++)
            sb.AppendLine($"{Classes[i],8} {Precision[i],10:F4} {Recall[i],8:F4} {F1[i],8:F4} {Support[i],8}");
        return sb.ToString();
    }

    /// <summary>
    /// Convert the confusion matrix to a labeled DataFrame.
    /// Rows = actual class, columns = predicted class. Labels are the class integers as strings.
    /// </summary>
    public DataFrame ToDataFrame()
    {
        int size = ConfusionMatrix.GetLength(0);
        var columns = new List<IColumn>(size);
        for (int col = 0; col < size; col++)
        {
            var values = new int[size];
            for (int row = 0; row < size; row++)
                values[row] = ConfusionMatrix[row, col];
            columns.Add(new Column<int>(Classes[col].ToString(), values));
        }
        return new DataFrame(columns);
    }
}
