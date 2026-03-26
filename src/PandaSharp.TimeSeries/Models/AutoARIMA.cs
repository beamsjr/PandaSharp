using PandaSharp;
using PandaSharp.Column;

namespace PandaSharp.TimeSeries.Models;

/// <summary>
/// Information criterion for model selection.
/// </summary>
public enum InformationCriterion
{
    /// <summary>Akaike Information Criterion.</summary>
    AIC,
    /// <summary>Bayesian Information Criterion.</summary>
    BIC
}

/// <summary>
/// Automatic ARIMA order selection. Performs a stepwise search over (p,d,q)
/// ranges and selects the model with the best AIC or BIC.
/// </summary>
public class AutoARIMA : IForecaster
{
    private readonly int _maxP;
    private readonly int _maxD;
    private readonly int _maxQ;
    private readonly InformationCriterion _criterion;

    private ARIMA? _bestModel;

    /// <summary>
    /// Create an AutoARIMA selector.
    /// </summary>
    /// <param name="maxP">Maximum AR order to consider.</param>
    /// <param name="maxD">Maximum differencing order to consider.</param>
    /// <param name="maxQ">Maximum MA order to consider.</param>
    /// <param name="criterion">Information criterion for selection (AIC or BIC).</param>
    public AutoARIMA(int maxP = 3, int maxD = 2, int maxQ = 3, InformationCriterion criterion = InformationCriterion.AIC)
    {
        if (maxP < 0) throw new ArgumentOutOfRangeException(nameof(maxP));
        if (maxD < 0) throw new ArgumentOutOfRangeException(nameof(maxD));
        if (maxQ < 0) throw new ArgumentOutOfRangeException(nameof(maxQ));
        _maxP = maxP;
        _maxD = maxD;
        _maxQ = maxQ;
        _criterion = criterion;
    }

    /// <summary>The best-fit ARIMA model after calling <see cref="Fit"/>.</summary>
    public ARIMA? BestModel => _bestModel;

    /// <summary>The (p,d,q) order of the best model.</summary>
    public (int P, int D, int Q) BestOrder { get; private set; }

    /// <summary>The best information criterion score achieved.</summary>
    public double BestScore { get; private set; } = double.MaxValue;

    /// <inheritdoc />
    public IForecaster Fit(DataFrame df, string dateColumn, string valueColumn)
    {
        ArgumentNullException.ThrowIfNull(df);
        ArgumentNullException.ThrowIfNull(dateColumn);
        ArgumentNullException.ThrowIfNull(valueColumn);

        double bestScore = double.MaxValue;
        ARIMA? bestModel = null;
        (int, int, int) bestOrder = (0, 0, 0);

        // Determine d by checking stationarity (simple heuristic: try d=0,1,2 and pick lowest IC)
        // Full stepwise search
        for (int d = 0; d <= _maxD; d++)
        {
            for (int p = 0; p <= _maxP; p++)
            {
                for (int q = 0; q <= _maxQ; q++)
                {
                    // ARIMA(0,d,0) is a valid model (random walk with drift)

                    try
                    {
                        var model = new ARIMA(p, d, q);
                        model.Fit(df, dateColumn, valueColumn);

                        double score = _criterion == InformationCriterion.AIC ? model.AIC : model.BIC;

                        if (score < bestScore)
                        {
                            bestScore = score;
                            bestModel = model;
                            bestOrder = (p, d, q);
                        }
                    }
                    catch (Exception)
                    {
                        // Skip invalid configurations (e.g., not enough data)
                    }
                }
            }
        }

        if (bestModel is null)
            throw new InvalidOperationException("No valid ARIMA model found. Check your data.");

        _bestModel = bestModel;
        BestOrder = bestOrder;
        BestScore = bestScore;

        return this;
    }

    /// <inheritdoc />
    public ForecastResult Forecast(int horizon)
    {
        if (_bestModel is null) throw new InvalidOperationException("Call Fit() first.");
        return _bestModel.Forecast(horizon);
    }

    /// <inheritdoc />
    public ForecastResult ForecastWithInterval(int horizon, double alpha = 0.05)
    {
        if (_bestModel is null) throw new InvalidOperationException("Call Fit() first.");
        return _bestModel.ForecastWithInterval(horizon, alpha);
    }
}
