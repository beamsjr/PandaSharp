using Cortex;
using Cortex.Column;

namespace Cortex.ML.Metrics;

/// <summary>
/// Converts a confusion matrix into a DataFrame formatted for Viz heatmap display.
/// The output DataFrame has columns: actual_class, predicted_class_0, predicted_class_1, ...
/// </summary>
public static class ConfusionMatrixDisplay
{
    /// <summary>
    /// Convert a confusion matrix to a DataFrame suitable for heatmap visualization.
    /// Rows represent actual classes; columns represent predicted classes.
    /// </summary>
    /// <param name="matrix">Confusion matrix of shape [n_classes, n_classes].</param>
    /// <param name="classNames">Optional class names. If null, uses "0", "1", "2", etc.</param>
    /// <returns>DataFrame with actual_class column and one column per predicted class.</returns>
    public static DataFrame ToDataFrame(int[,] matrix, string[]? classNames = null)
    {
        int nClasses = matrix.GetLength(0);
        if (matrix.GetLength(1) != nClasses)
            throw new ArgumentException("Confusion matrix must be square (n_classes x n_classes).");

        classNames ??= Enumerable.Range(0, nClasses).Select(i => i.ToString()).ToArray();

        if (classNames.Length != nClasses)
            throw new ArgumentException(
                $"classNames length ({classNames.Length}) must match matrix dimensions ({nClasses}).");

        var columns = new List<IColumn>();

        // First column: actual class labels
        columns.Add(new StringColumn("actual_class", classNames));

        // One column per predicted class
        for (int pred = 0; pred < nClasses; pred++)
        {
            var values = new int[nClasses];
            for (int actual = 0; actual < nClasses; actual++)
                values[actual] = matrix[actual, pred];
            columns.Add(new Column<int>($"predicted_{classNames[pred]}", values));
        }

        return new DataFrame(columns);
    }
}
