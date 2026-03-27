using Cortex;

namespace Cortex.TimeSeries.Models;

/// <summary>
/// Result of a time series forecast, including point predictions and optional confidence intervals.
/// </summary>
/// <param name="Dates">Forecasted date points.</param>
/// <param name="Values">Point forecast values.</param>
/// <param name="LowerBound">Lower confidence bound (empty if not computed).</param>
/// <param name="UpperBound">Upper confidence bound (empty if not computed).</param>
public record ForecastResult(
    DateTime[] Dates,
    double[] Values,
    double[] LowerBound,
    double[] UpperBound);

/// <summary>
/// Common interface for all time series forecasters.
/// Call <see cref="Fit"/> to train the model, then <see cref="Forecast"/> or
/// <see cref="ForecastWithInterval"/> to produce predictions.
/// </summary>
public interface IForecaster
{
    /// <summary>
    /// Fit the forecaster to historical data.
    /// </summary>
    /// <param name="df">DataFrame containing the time series.</param>
    /// <param name="dateColumn">Name of the column holding DateTime values.</param>
    /// <param name="valueColumn">Name of the column holding numeric observation values.</param>
    /// <returns>The fitted forecaster instance for fluent chaining.</returns>
    IForecaster Fit(DataFrame df, string dateColumn, string valueColumn);

    /// <summary>
    /// Produce a point forecast for the given number of future steps.
    /// </summary>
    /// <param name="horizon">Number of future time steps to forecast.</param>
    /// <returns>A <see cref="ForecastResult"/> with point predictions (bounds will be empty).</returns>
    ForecastResult Forecast(int horizon);

    /// <summary>
    /// Produce a forecast with confidence intervals.
    /// </summary>
    /// <param name="horizon">Number of future time steps to forecast.</param>
    /// <param name="alpha">Significance level (e.g. 0.05 for a 95% interval).</param>
    /// <returns>A <see cref="ForecastResult"/> with point predictions and confidence bounds.</returns>
    ForecastResult ForecastWithInterval(int horizon, double alpha = 0.05);
}
