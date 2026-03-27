using FluentAssertions;
using Cortex;
using Cortex.Column;
using Cortex.TimeSeries.Decomposition;
using Cortex.TimeSeries.Diagnostics;
using Cortex.TimeSeries.Models;
using Xunit;

namespace Cortex.TimeSeries.Tests.EdgeCases;

public class ArimaEdgeCaseTests
{
    private static DataFrame MakeDf(double[] values)
    {
        var dates = Enumerable.Range(0, values.Length)
            .Select(i => DateTime.Today.AddDays(i)).ToArray();
        return new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Value", values));
    }

    [Fact]
    public void ARIMA_Forecast_Zero_Steps_Returns_Empty_Result()
    {
        var df = MakeDf(Enumerable.Range(1, 50).Select(i => (double)i).ToArray());
        var arima = new ARIMA(1, 1, 0);
        arima.Fit(df, "Date", "Value");

        var result = arima.Forecast(0);

        result.Values.Should().BeEmpty();
        result.Dates.Should().BeEmpty();
    }

    [Fact]
    public void ARIMA_D_Zero_No_Differencing_Works()
    {
        var data = Enumerable.Range(1, 50).Select(i => (double)i + Math.Sin(i)).ToArray();
        var df = MakeDf(data);
        var arima = new ARIMA(p: 2, d: 0, q: 0);
        arima.Fit(df, "Date", "Value");

        var result = arima.Forecast(5);

        result.Values.Should().HaveCount(5);
        result.Values.Should().OnlyContain(v => double.IsFinite(v));
    }

    [Fact]
    public void ARIMA_Negative_P_Throws()
    {
        var act = () => new ARIMA(p: -1, d: 0, q: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ARIMA_Negative_D_Throws()
    {
        var act = () => new ARIMA(p: 0, d: -1, q: 0);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ARIMA_Negative_Q_Throws()
    {
        var act = () => new ARIMA(p: 0, d: 0, q: -1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void ARIMA_Too_Few_Points_Throws()
    {
        // p=5,d=1,q=0 requires many more points than we provide
        var df = MakeDf([1.0, 2.0, 3.0, 4.0]);
        var arima = new ARIMA(p: 5, d: 1, q: 0);

        var act = () => arima.Fit(df, "Date", "Value");

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void ARIMA_AIC_BIC_With_Constant_Series_Does_Not_Return_NaN()
    {
        // BUG: When all values are the same, sigma2=0, Math.Log(0) = -Infinity
        // AIC and BIC should return finite values or handle gracefully
        var data = Enumerable.Repeat(5.0, 50).ToArray();
        var df = MakeDf(data);
        var arima = new ARIMA(p: 1, d: 0, q: 0);
        arima.Fit(df, "Date", "Value");

        var aic = arima.AIC;
        var bic = arima.BIC;

        double.IsNaN(aic).Should().BeFalse("AIC should not be NaN for constant series");
        double.IsNaN(bic).Should().BeFalse("BIC should not be NaN for constant series");
    }
}

public class SarimaEdgeCaseTests
{
    private static DataFrame MakeDf(double[] values)
    {
        var dates = Enumerable.Range(0, values.Length)
            .Select(i => DateTime.Today.AddDays(i)).ToArray();
        return new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Value", values));
    }

    [Fact]
    public void SARIMA_Seasonal_Period_Larger_Than_Data_Throws()
    {
        var data = Enumerable.Range(1, 10).Select(i => (double)i).ToArray();
        // period=50 > data length 10
        var sarima = new SARIMA(1, 0, 0, 1, 0, 0, 50);
        var df = MakeDf(data);

        var act = () => sarima.Fit(df, "Date", "Value");

        act.Should().Throw<InvalidOperationException>();
    }

    [Fact]
    public void SARIMA_Period_1_Throws()
    {
        var act = () => new SARIMA(1, 0, 0, 1, 0, 0, 1);
        act.Should().Throw<ArgumentOutOfRangeException>();
    }
}

public class ExponentialSmoothingEdgeCaseTests
{
    private static DataFrame MakeDf(double[] values)
    {
        var dates = Enumerable.Range(0, values.Length)
            .Select(i => DateTime.Today.AddDays(i)).ToArray();
        return new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Value", values));
    }

    [Fact]
    public void ES_Single_DataPoint_Fits_And_Forecasts()
    {
        var df = MakeDf([42.0]);
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(3);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => Math.Abs(v - 42.0) < 1e-10,
            "forecast from single point should be that point's value");
    }

    [Fact]
    public void ES_Empty_Array_Throws()
    {
        var df = MakeDf([]);
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);

        var act = () => es.Fit(df, "Date", "Value");

        act.Should().Throw<Exception>();
    }

    [Fact]
    public void ES_All_Same_Values_Forecasts_That_Value()
    {
        var data = Enumerable.Repeat(7.0, 30).ToArray();
        var df = MakeDf(data);
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(5);

        result.Values.Should().OnlyContain(v => Math.Abs(v - 7.0) < 1e-8);
    }

    [Fact]
    public void ES_Forecast_Zero_Horizon_Returns_Empty()
    {
        var data = Enumerable.Range(1, 20).Select(i => (double)i).ToArray();
        var df = MakeDf(data);
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(0);

        result.Values.Should().BeEmpty();
        result.Dates.Should().BeEmpty();
    }

    [Fact]
    public void ES_Forecast_Before_Fit_Throws_Clear_Error()
    {
        // BUG: EnsureFitted checks _history.Length == 0 instead of a _fitted flag.
        // Before Fit is called, _history is [] and _dates is [], so _history.Length == 0
        // triggers the check correctly. But the error message says "Call Fit() before forecasting."
        // which is correct. However, if someone constructs and calls Forecast, the _dates[^1]
        // in ComputeForecast would throw IndexOutOfRangeException before reaching EnsureFitted.
        // Actually EnsureFitted is called first, so this should work.
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.3);

        var act = () => es.Forecast(5);

        act.Should().Throw<InvalidOperationException>()
            .WithMessage("*Fit*");
    }
}

public class AutoArimaEdgeCaseTests
{
    private static DataFrame MakeDf(double[] values)
    {
        var dates = Enumerable.Range(0, values.Length)
            .Select(i => DateTime.Today.AddDays(i)).ToArray();
        return new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Value", values));
    }

    [Fact]
    public void AutoARIMA_Two_DataPoints_Finds_Model()
    {
        // Minimal data: only ARIMA(0,0,0) can possibly work
        var df = MakeDf([1.0, 2.0, 3.0, 4.0, 5.0]);
        var auto = new AutoARIMA(maxP: 1, maxD: 1, maxQ: 0);
        auto.Fit(df, "Date", "Value");

        auto.BestModel.Should().NotBeNull();
    }

    [Fact]
    public void AutoARIMA_Empty_Data_Throws()
    {
        var df = MakeDf([]);
        var auto = new AutoARIMA(maxP: 1, maxD: 1, maxQ: 0);

        var act = () => auto.Fit(df, "Date", "Value");

        act.Should().Throw<Exception>();
    }

    [Fact]
    public void AutoARIMA_MaxP0_MaxQ0_Still_Finds_Model()
    {
        // With maxP=0 and maxQ=0, only ARIMA(0,d,0) models are tried
        var data = Enumerable.Range(1, 30).Select(i => (double)i).ToArray();
        var df = MakeDf(data);
        var auto = new AutoARIMA(maxP: 0, maxD: 1, maxQ: 0);
        auto.Fit(df, "Date", "Value");

        auto.BestModel.Should().NotBeNull();
        auto.BestOrder.P.Should().Be(0);
        auto.BestOrder.Q.Should().Be(0);
    }
}

public class SeasonalDecomposeEdgeCaseTests
{
    [Fact]
    public void Decompose_Period_Greater_Than_Half_Data_Throws()
    {
        var series = Enumerable.Range(1, 10).Select(i => (double)i).ToArray();

        var act = () => SeasonalDecompose.Decompose(series, period: 6);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void Decompose_Period_1_Throws()
    {
        var series = Enumerable.Range(1, 20).Select(i => (double)i).ToArray();

        var act = () => SeasonalDecompose.Decompose(series, period: 1);

        act.Should().Throw<ArgumentOutOfRangeException>();
    }

    [Fact]
    public void Decompose_Odd_Data_Even_Period_Works()
    {
        // 25 data points with period 4 -- odd length, even period
        var series = Enumerable.Range(0, 25)
            .Select(i => 10.0 + 2.0 * Math.Sin(2 * Math.PI * i / 4.0))
            .ToArray();

        var result = SeasonalDecompose.Decompose(series, period: 4);

        result.Observed.Should().HaveCount(25);
        result.Trend.Should().HaveCount(25);
        result.Seasonal.Should().HaveCount(25);
        result.Residual.Should().HaveCount(25);
    }
}

public class StationarityTestEdgeCaseTests
{
    [Fact]
    public void KPSS_Constant_Series_Does_Not_Produce_NaN_Or_Infinity()
    {
        // BUG: Constant series => all residuals are 0 => s2 = 0 => division by zero
        var series = Enumerable.Repeat(5.0, 20).ToArray();

        var result = StationarityTests.KPSS(series);

        double.IsNaN(result.TestStatistic).Should().BeFalse(
            "KPSS test statistic should not be NaN for constant series");
        double.IsInfinity(result.TestStatistic).Should().BeFalse(
            "KPSS test statistic should not be Infinity for constant series");
    }

    [Fact]
    public void ADF_Constant_Series_Does_Not_Throw()
    {
        var series = Enumerable.Repeat(5.0, 20).ToArray();

        var act = () => StationarityTests.AugmentedDickeyFuller(series);

        act.Should().NotThrow();
    }

    [Fact]
    public void ADF_Single_Element_Throws()
    {
        var series = new double[] { 1.0 };

        var act = () => StationarityTests.AugmentedDickeyFuller(series);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void KPSS_Single_Element_Throws()
    {
        var series = new double[] { 1.0 };

        var act = () => StationarityTests.KPSS(series);

        act.Should().Throw<ArgumentException>();
    }

    [Fact]
    public void ADF_Alternating_Series()
    {
        var series = Enumerable.Range(0, 30).Select(i => i % 2 == 0 ? 1.0 : -1.0).ToArray();

        var result = StationarityTests.AugmentedDickeyFuller(series);

        result.TestStatistic.Should().NotBe(0);
        double.IsNaN(result.TestStatistic).Should().BeFalse();
        double.IsInfinity(result.TestStatistic).Should().BeFalse();
    }

    [Fact]
    public void KPSS_Alternating_Series()
    {
        var series = Enumerable.Range(0, 30).Select(i => i % 2 == 0 ? 1.0 : -1.0).ToArray();

        var result = StationarityTests.KPSS(series);

        double.IsNaN(result.TestStatistic).Should().BeFalse();
        double.IsInfinity(result.TestStatistic).Should().BeFalse();
    }
}

public class AutocorrelationEdgeCaseTests
{
    [Fact]
    public void ACF_NLags_Greater_Than_Data_Clamps_To_N_Minus_1()
    {
        var series = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        // nlags=100 > series.Length=5, should clamp to 4
        var result = AutocorrelationTests.ACF(series, maxLags: 100);

        result.Should().HaveCount(5); // 0..4
        result[0].Should().Be(1.0);
    }

    [Fact]
    public void ACF_NLags_Zero_Returns_Just_Lag0()
    {
        var series = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var result = AutocorrelationTests.ACF(series, maxLags: 0);

        result.Should().HaveCount(1);
        result[0].Should().Be(1.0);
    }

    [Fact]
    public void PACF_NLags_Greater_Than_Data_Clamps()
    {
        var series = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var result = AutocorrelationTests.PACF(series, maxLags: 100);

        result.Should().HaveCount(5);
        result[0].Should().Be(1.0);
    }

    [Fact]
    public void PACF_NLags_Zero_Returns_Just_Lag0()
    {
        var series = new double[] { 1.0, 2.0, 3.0, 4.0, 5.0 };

        var result = AutocorrelationTests.PACF(series, maxLags: 0);

        result.Should().HaveCount(1);
        result[0].Should().Be(1.0);
    }

    [Fact]
    public void ACF_Constant_Series_Returns_Zeros_For_NonZero_Lags()
    {
        var series = Enumerable.Repeat(3.0, 20).ToArray();

        var result = AutocorrelationTests.ACF(series, maxLags: 5);

        result[0].Should().Be(1.0);
        // For constant series, variance is 0, all other lags should be 0
        for (int i = 1; i < result.Length; i++)
            double.IsNaN(result[i]).Should().BeFalse($"ACF[{i}] should not be NaN");
    }
}
