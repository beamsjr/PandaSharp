using System.Runtime.InteropServices;
using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.TimeSeries.Models;

/// <summary>Strategy for time series cross-validation.</summary>
public enum BacktestStrategy
{
    /// <summary>Expanding window: training set grows with each fold.</summary>
    Expanding,
    /// <summary>Sliding window: fixed-size training window that slides forward.</summary>
    Sliding
}

/// <summary>
/// Time series backtesting via expanding or sliding window cross-validation.
/// Evaluates a forecaster across multiple folds and returns a DataFrame with per-fold metrics.
/// </summary>
public static class Backtesting
{
    /// <summary>
    /// Run backtesting on a forecaster over the given DataFrame.
    /// </summary>
    /// <param name="forecasterFactory">Factory that creates a fresh forecaster instance for each fold.</param>
    /// <param name="df">DataFrame containing the time series.</param>
    /// <param name="dateColumn">Name of the DateTime column.</param>
    /// <param name="valueColumn">Name of the numeric value column.</param>
    /// <param name="initialTrainSize">Number of observations in the first training window.</param>
    /// <param name="horizon">Forecast horizon for each fold.</param>
    /// <param name="step">Number of observations to advance between folds.</param>
    /// <param name="strategy">Expanding or Sliding window strategy.</param>
    /// <returns>DataFrame with columns: Fold, TrainSize, MAE, RMSE, MAPE.</returns>
    public static DataFrame Evaluate(
        Func<IForecaster> forecasterFactory,
        DataFrame df,
        string dateColumn,
        string valueColumn,
        int initialTrainSize,
        int horizon = 1,
        int step = 1,
        BacktestStrategy strategy = BacktestStrategy.Expanding)
    {
        ArgumentNullException.ThrowIfNull(forecasterFactory);
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(dateColumn);
        ArgumentNullException.ThrowIfNull(valueColumn);
        if (initialTrainSize < 1) throw new ArgumentOutOfRangeException(nameof(initialTrainSize));
        if (horizon < 1) throw new ArgumentOutOfRangeException(nameof(horizon));
        if (step < 1) throw new ArgumentOutOfRangeException(nameof(step));

        int n = df.RowCount;
        var valueCol = df[valueColumn];

        var foldList = new List<int>();
        var trainSizeList = new List<int>();
        var maeList = new List<double>();
        var rmseList = new List<double>();
        var mapeList = new List<double>();

        int fold = 0;
        for (int trainEnd = initialTrainSize; trainEnd + horizon <= n; trainEnd += step)
        {
            int trainStart = strategy == BacktestStrategy.Sliding
                ? Math.Max(0, trainEnd - initialTrainSize)
                : 0;
            int trainSize = trainEnd - trainStart;

            // Slice training data
            var trainDf = SliceRows(df, trainStart, trainSize);

            try
            {
                var forecaster = forecasterFactory();
                forecaster.Fit(trainDf, dateColumn, valueColumn);
                var result = forecaster.Forecast(horizon);

                // Compute metrics against actuals
                double mae = 0, sse = 0, mape = 0;
                int validCount = 0;
                for (int h = 0; h < horizon; h++)
                {
                    int actualIdx = trainEnd + h;
                    if (actualIdx >= n) break;
                    double actual = TypeHelpers.GetDouble(valueCol, actualIdx);
                    double predicted = result.Values[h];
                    double err = actual - predicted;
                    mae += Math.Abs(err);
                    sse += err * err;
                    if (Math.Abs(actual) > 1e-15)
                        mape += Math.Abs(err / actual);
                    validCount++;
                }

                if (validCount > 0)
                {
                    foldList.Add(fold);
                    trainSizeList.Add(trainSize);
                    maeList.Add(mae / validCount);
                    rmseList.Add(Math.Sqrt(sse / validCount));
                    mapeList.Add(mape / validCount * 100);
                }
            }
            catch (Exception)
            {
                // Skip folds that fail (e.g., insufficient data)
            }

            fold++;
        }

        return DataFrame.FromDictionary(new Dictionary<string, Array>
        {
            ["Fold"] = foldList.ToArray(),
            ["TrainSize"] = trainSizeList.ToArray(),
            ["MAE"] = maeList.ToArray(),
            ["RMSE"] = rmseList.ToArray(),
            ["MAPE"] = mapeList.ToArray()
        });
    }

    private static DataFrame SliceRows(DataFrame df, int start, int count)
    {
        var cols = new List<IColumn>();
        foreach (var name in df.ColumnNames)
            cols.Add(df[name].Slice(start, count));
        return new DataFrame(cols);
    }
}
