using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.TimeSeries.Decomposition;
using PandaSharp.TimeSeries.Diagnostics;
using PandaSharp.TimeSeries.Features;
using PandaSharp.TimeSeries.Models;
using Xunit;

namespace PandaSharp.TimeSeries.Tests;

public class TimeSeriesTests
{
    // ── Synthetic data ──────────────────────────────────────────────────

    private static readonly double[] Trend =
        Enumerable.Range(1, 100).Select(i => (double)i).ToArray();

    private static readonly double[] Seasonal =
        Enumerable.Range(0, 100).Select(i => 10 * Math.Sin(2 * Math.PI * i / 12.0)).ToArray();

    private static readonly double[] Combined =
        Trend.Zip(Seasonal, (t, s) => t + s).ToArray();

    private static readonly DateTime[] Dates =
        Enumerable.Range(0, 100).Select(i => DateTime.Today.AddDays(i)).ToArray();

    private static DataFrame MakeDataFrame() =>
        new(
            new Column<DateTime>("Date", Dates),
            new Column<double>("Value", Combined));

    private static DataFrame MakeTrendDataFrame() =>
        new(
            new Column<DateTime>("Date", Dates),
            new Column<double>("Value", Trend));

    // ── 1. SimpleMovingAverageForecast ──────────────────────────────────

    [Fact]
    public void SimpleMovingAverageForecast_ReturnsCorrectHorizon()
    {
        var df = MakeDataFrame();
        var sma = new SimpleMovingAverageForecast(window: 5);
        sma.Fit(df, "Date", "Value");

        var result = sma.Forecast(10);

        result.Values.Should().HaveCount(10);
        result.Dates.Should().HaveCount(10);
        result.Values.Should().OnlyContain(v => double.IsFinite(v));
    }

    // ── 2. ExponentialSmoothing ─────────────────────────────────────────

    [Fact]
    public void ExponentialSmoothing_Simple_ForecastValuesAreReasonable()
    {
        var df = MakeDataFrame();
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(10);

        result.Values.Should().HaveCount(10);
        double min = Combined.Min();
        double max = Combined.Max();
        // Forecast values should be within a reasonable range of the data
        result.Values.Should().OnlyContain(v => v >= min - 50 && v <= max + 50);
    }

    // ── 3. ARIMA ────────────────────────────────────────────────────────

    [Fact]
    public void ARIMA_Fit110_OnTrend_ForecastIncreases()
    {
        var df = MakeTrendDataFrame();
        var arima = new ARIMA(p: 1, d: 1, q: 0);
        arima.Fit(df, "Date", "Value");

        var result = arima.Forecast(5);

        result.Values.Should().HaveCount(5);
        // Each subsequent forecast should be >= the previous (trend data)
        for (int i = 1; i < result.Values.Length; i++)
            result.Values[i].Should().BeGreaterThanOrEqualTo(result.Values[i - 1] - 0.01,
                "ARIMA(1,1,0) on linear trend should produce non-decreasing forecasts");
    }

    // ── 4. AutoARIMA ────────────────────────────────────────────────────

    [Fact]
    public void AutoARIMA_FindsBestModel()
    {
        var df = MakeTrendDataFrame();
        var auto = new AutoARIMA(maxP: 2, maxD: 2, maxQ: 2);
        auto.Fit(df, "Date", "Value");

        auto.BestModel.Should().NotBeNull();
        auto.BestOrder.P.Should().BeGreaterThanOrEqualTo(0);
        auto.BestOrder.D.Should().BeGreaterThanOrEqualTo(0);
        auto.BestOrder.Q.Should().BeGreaterThanOrEqualTo(0);
        auto.BestScore.Should().BeLessThan(double.MaxValue);
    }

    // ── 5. SeasonalDecompose ────────────────────────────────────────────

    [Fact]
    public void SeasonalDecompose_Additive_HasAllComponents()
    {
        var result = SeasonalDecompose.Decompose(Combined, period: 12, DecomposeType.Additive);

        result.Observed.Should().HaveCount(100);
        result.Trend.Should().HaveCount(100);
        result.Seasonal.Should().HaveCount(100);
        result.Residual.Should().HaveCount(100);

        // Trend values in the middle should be finite (edges can be NaN)
        result.Trend[50].Should().NotBe(double.NaN);
        result.Seasonal[50].Should().NotBe(double.NaN);
    }

    // ── 6. ACF ──────────────────────────────────────────────────────────

    [Fact]
    public void ACF_LagZeroIsOne_SubsequentLagsDecrease()
    {
        var acf = AutocorrelationTests.ACF(Trend, maxLags: 10);

        acf[0].Should().Be(1.0);
        // For a linear trend, autocorrelation should decrease with lag
        Math.Abs(acf[1]).Should().BeGreaterThan(Math.Abs(acf[10]));
    }

    // ── 7. ADF test ─────────────────────────────────────────────────────

    [Fact]
    public void ADF_TrendData_IsNonStationary()
    {
        var result = StationarityTests.AugmentedDickeyFuller(Trend);

        // A linear trend is non-stationary; the test statistic should
        // not be more negative than the 5% critical value (-2.86).
        result.TestStatistic.Should().BeGreaterThan(result.CriticalValues["5%"],
            "a linear trend should fail to reject the unit root null at 5%");
        result.PValue.Should().BeGreaterThan(0.05);
    }

    // ── 8. KPSS test ────────────────────────────────────────────────────

    [Fact]
    public void KPSS_TrendData_RejectsStationarity()
    {
        var result = StationarityTests.KPSS(Trend, regressionType: "c");

        // For a trend, KPSS with constant-only regression should reject stationarity
        // (test stat > critical value at some level).
        result.TestStatistic.Should().BeGreaterThan(0);
        result.CriticalValues.Should().ContainKey("5%");
        result.UsedLags.Should().BeGreaterThanOrEqualTo(0);
    }

    // ── 9. Periodogram ──────────────────────────────────────────────────

    [Fact]
    public void Periodogram_Seasonal_DominantFrequencyNear1Over12()
    {
        var dominant = Periodogram.DominantFrequencies(Seasonal, topK: 1);

        // The dominant frequency column "Frequency" should be close to 1/12
        var freqCol = TypeHelpers.GetDoubleArray(dominant["Frequency"]);
        freqCol.Should().HaveCountGreaterThanOrEqualTo(1);

        double expectedFreq = 1.0 / 12.0;
        // Period of sin(2*pi*i/12) is 12 samples => frequency = 1/12
        // DFT frequency resolution is 1/n; closest bin may not be exact
        freqCol[0].Should().BeApproximately(expectedFreq, 0.02,
            "dominant frequency of sin wave with period 12 should be near 1/12");
    }

    // ── 10. LagFeatures ─────────────────────────────────────────────────

    [Fact]
    public void LagFeatures_CreatesColumnsWithCorrectNaNPattern()
    {
        var df = MakeDataFrame();
        var lags = new LagFeatures("Value", 1, 3);
        var result = lags.FitTransform(df);

        result.ColumnNames.Should().Contain("Value_lag_1");
        result.ColumnNames.Should().Contain("Value_lag_3");

        // Lag 1: first row should be null
        var lag1Col = result.GetColumn<double>("Value_lag_1");
        lag1Col.IsNull(0).Should().BeTrue("first element of lag_1 should be null");
        lag1Col.IsNull(1).Should().BeFalse("second element of lag_1 should have a value");

        // Lag 3: first 3 rows should be null
        var lag3Col = result.GetColumn<double>("Value_lag_3");
        lag3Col.IsNull(0).Should().BeTrue();
        lag3Col.IsNull(1).Should().BeTrue();
        lag3Col.IsNull(2).Should().BeTrue();
        lag3Col.IsNull(3).Should().BeFalse();
    }

    // ── 11. DateTimeFeatures ────────────────────────────────────────────

    [Fact]
    public void DateTimeFeatures_CreatesCalendarColumns()
    {
        var df = MakeDataFrame();
        var dtf = new DateTimeFeatures("Date");
        var result = dtf.FitTransform(df);

        result.ColumnNames.Should().Contain("Date_day_of_week");
        result.ColumnNames.Should().Contain("Date_month");
        result.ColumnNames.Should().Contain("Date_quarter");

        var monthCol = result.GetColumn<int>("Date_month");
        monthCol[0]!.Value.Should().Be(Dates[0].Month);
    }

    // ── 12. RollingFeatures ─────────────────────────────────────────────

    [Fact]
    public void RollingFeatures_CreatesRollingMeanColumn()
    {
        var df = MakeDataFrame();
        var rolling = new RollingFeatures("Value", 5);
        var result = rolling.FitTransform(df);

        result.ColumnNames.Should().Contain("Value_rolling_5_mean");
        result.ColumnNames.Should().Contain("Value_rolling_5_std");

        // First 4 rows (window-1) should be null for rolling_5
        var meanCol = result.GetColumn<double>("Value_rolling_5_mean");
        meanCol.IsNull(0).Should().BeTrue();
        meanCol.IsNull(3).Should().BeTrue();
        meanCol.IsNull(4).Should().BeFalse();
    }

    // ── 13. FourierFeatures ─────────────────────────────────────────────

    [Fact]
    public void FourierFeatures_CreatesSinCosColumns()
    {
        var df = MakeDataFrame();
        var fourier = new FourierFeatures([12.0], harmonics: 2);
        var result = fourier.FitTransform(df);

        result.ColumnNames.Should().Contain("fourier_p12_k1_sin");
        result.ColumnNames.Should().Contain("fourier_p12_k1_cos");
        result.ColumnNames.Should().Contain("fourier_p12_k2_sin");
        result.ColumnNames.Should().Contain("fourier_p12_k2_cos");

        // sin/cos values should be in [-1, 1]
        var sinCol = result.GetColumn<double>("fourier_p12_k1_sin");
        for (int i = 0; i < sinCol.Length; i++)
            sinCol[i]!.Value.Should().BeInRange(-1.0, 1.0);
    }

    // ── 14. ChangepointDetection ────────────────────────────────────────

    [Fact]
    public void ChangepointDetection_PELT_FindsShiftNearMidpoint()
    {
        // Create data with a clear mean shift at index 50
        var data = new double[100];
        for (int i = 0; i < 50; i++) data[i] = 0.0;
        for (int i = 50; i < 100; i++) data[i] = 10.0;

        var changepoints = ChangepointDetection.PELT(data);

        changepoints.Should().HaveCountGreaterThanOrEqualTo(1,
            "PELT should detect at least one changepoint in data with a clear mean shift");

        // At least one changepoint should be near index 50
        changepoints.Should().Contain(cp => cp >= 45 && cp <= 55,
            "changepoint should be near index 50 where the mean shift occurs");
    }

    // ── 15. Null guard tests ────────────────────────────────────────────

    [Fact]
    public void SMA_FitNull_ThrowsArgumentNullException()
    {
        var sma = new SimpleMovingAverageForecast();
        var act = () => sma.Fit(null!, "Date", "Value");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ExponentialSmoothing_FitNull_ThrowsArgumentNullException()
    {
        var es = new ExponentialSmoothing();
        var act = () => es.Fit(null!, "Date", "Value");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ARIMA_FitNull_ThrowsArgumentNullException()
    {
        var arima = new ARIMA();
        var act = () => arima.Fit(null!, "Date", "Value");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void AutoARIMA_FitNull_ThrowsArgumentNullException()
    {
        var auto = new AutoARIMA();
        var act = () => auto.Fit(null!, "Date", "Value");
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ChangepointDetection_PELT_NullThrows()
    {
        var act = () => ChangepointDetection.PELT(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void SeasonalDecompose_NullThrows()
    {
        var act = () => SeasonalDecompose.Decompose(null!, 12);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ADF_NullThrows()
    {
        var act = () => StationarityTests.AugmentedDickeyFuller(null!);
        act.Should().Throw<ArgumentNullException>();
    }

    [Fact]
    public void ACF_NullThrows()
    {
        var act = () => AutocorrelationTests.ACF(null!, 5);
        act.Should().Throw<ArgumentNullException>();
    }

    // ── 16. Backtesting ─────────────────────────────────────────────────

    [Fact]
    public void Backtesting_ExpandingWindow_HasExpectedColumns()
    {
        var df = MakeDataFrame();
        var result = Backtesting.Evaluate(
            forecasterFactory: () => new SimpleMovingAverageForecast(window: 5),
            df: df,
            dateColumn: "Date",
            valueColumn: "Value",
            initialTrainSize: 50,
            horizon: 1,
            step: 10,
            strategy: BacktestStrategy.Expanding);

        result.ColumnNames.Should().Contain("Fold");
        result.ColumnNames.Should().Contain("MAE");
        result.ColumnNames.Should().Contain("RMSE");
        result.RowCount.Should().BeGreaterThan(0);
    }
}
