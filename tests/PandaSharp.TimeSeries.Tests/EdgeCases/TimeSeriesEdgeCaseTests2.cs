using FluentAssertions;
using PandaSharp;
using PandaSharp.Column;
using PandaSharp.TimeSeries.Decomposition;
using PandaSharp.TimeSeries.Features;
using PandaSharp.TimeSeries.Models;
using Xunit;

namespace PandaSharp.TimeSeries.Tests.EdgeCases;

public class ArimaTripleDifferencingTests
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
    public void ARIMA_D2_Forecast_Continues_Trend()
    {
        // Perfect linear series: values = 1,2,3,...,30
        // d=2 differencing should give zeros. Forecast should continue the linear trend.
        var data = Enumerable.Range(1, 30).Select(i => (double)i).ToArray();
        var df = MakeDf(data);
        var arima = new ARIMA(p: 0, d: 2, q: 0);
        arima.Fit(df, "Date", "Value");

        var result = arima.Forecast(3);

        result.Values.Should().HaveCount(3);
        // Should continue: 31, 32, 33
        result.Values[0].Should().BeApproximately(31.0, 0.5, "d=2 on linear data should continue the trend");
        result.Values[1].Should().BeApproximately(32.0, 0.5, "d=2 on linear data should continue the trend");
        result.Values[2].Should().BeApproximately(33.0, 0.5, "d=2 on linear data should continue the trend");
    }

    [Fact]
    public void ARIMA_D3_Forecast_Produces_Finite_Values()
    {
        // Quadratic series: values = i^2 for i=1..30
        // d=3 differencing of a quadratic should give zeros
        var data = Enumerable.Range(1, 30).Select(i => (double)(i * i)).ToArray();
        var df = MakeDf(data);
        var arima = new ARIMA(p: 0, d: 3, q: 0);
        arima.Fit(df, "Date", "Value");

        var result = arima.Forecast(3);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => double.IsFinite(v), "d=3 forecast should produce finite values");
        // Next values of i^2: 31^2=961, 32^2=1024, 33^2=1089
        result.Values[0].Should().BeApproximately(961.0, 5.0, "d=3 on quadratic should continue the trend");
    }
}

public class FFTPeriodogramEdgeTests
{
    [Fact]
    public void FFTPeriodogram_SingleElement_Returns_Something_Meaningful()
    {
        // A single-element series should not return empty
        var series = new double[] { 42.0 };

        var result = FFTPeriodogram.Compute(series);

        // After mean removal, the only value is 0, so power should be 0
        // But we should still get a valid result, not empty arrays
        result.Frequencies.Should().NotBeNull();
        result.Power.Should().NotBeNull();
        // At minimum, should have at least 1 frequency bin (DC component is excluded from halfN,
        // but halfN = fftSize/2 = 0 for fftSize=1, so this returns empty arrays)
        // This is arguably a bug: single element FFT should still return something
        result.Frequencies.Length.Should().BeGreaterThan(0,
            "FFT on 1-element array should return at least one frequency bin");
    }

    [Fact]
    public void FFTPeriodogram_AllZero_Returns_Zero_Power()
    {
        var series = new double[] { 0, 0, 0, 0, 0, 0, 0, 0 };

        var result = FFTPeriodogram.Compute(series);

        result.Power.Should().OnlyContain(p => p == 0,
            "FFT of all-zero series should have zero power at all frequencies");
    }

    [Fact]
    public void FFTPeriodogram_ConstantSeries_Returns_Zero_Power()
    {
        // After mean removal, constant series becomes all zeros
        var series = Enumerable.Repeat(5.0, 16).ToArray();

        var result = FFTPeriodogram.Compute(series);

        result.Power.Should().OnlyContain(p => Math.Abs(p) < 1e-10,
            "FFT of constant series should have near-zero power at all frequencies");
    }

    [Fact]
    public void Periodogram_DFT_ConstantSeries_Returns_Zero_Power()
    {
        var series = Enumerable.Repeat(5.0, 16).ToArray();

        var result = Periodogram.Compute(series);
        var powerCol = result.GetColumn<double>("Power");

        for (int i = 0; i < powerCol.Length; i++)
        {
            Math.Abs(powerCol.Values[i]).Should().BeLessThan(1e-10,
                $"Power at index {i} should be near-zero for constant series");
        }
    }
}

public class FourierFeaturesEdgeTests
{
    private static DataFrame MakeDf(int n)
    {
        var dates = Enumerable.Range(0, n)
            .Select(i => DateTime.Today.AddDays(i)).ToArray();
        var values = Enumerable.Range(0, n).Select(i => (double)i).ToArray();
        return new DataFrame(
            new Column<DateTime>("Date", dates),
            new Column<double>("Value", values));
    }

    [Fact]
    public void FourierFeatures_Period_Zero_Should_Throw()
    {
        // period=0 would cause division by zero in freq = 2*PI*k / period
        var act = () => new FourierFeatures(new double[] { 0.0 }, harmonics: 1);
        act.Should().Throw<ArgumentOutOfRangeException>("period=0 should be rejected");
    }

    [Fact]
    public void FourierFeatures_Negative_Period_Should_Throw()
    {
        // Negative period is nonsensical and should be rejected
        var act = () => new FourierFeatures(new double[] { -7.0 }, harmonics: 1);
        act.Should().Throw<ArgumentOutOfRangeException>("negative period should be rejected");
    }
}

public class LagFeaturesEdgeTests
{
    private static DataFrame MakeDf(int n)
    {
        var values = Enumerable.Range(1, n).Select(i => (double)i).ToArray();
        return new DataFrame(
            new Column<double>("Value", values));
    }

    [Fact]
    public void LagFeatures_Lag_Greater_Than_DataLength_Returns_All_Null()
    {
        var df = MakeDf(5);
        // lag=10 > data length 5
        var lags = new LagFeatures("Value", 10);
        var result = lags.FitTransform(df);

        var lagCol = result["Value_lag_10"];
        // All values should be null since lag exceeds data length
        for (int i = 0; i < df.RowCount; i++)
        {
            lagCol.IsNull(i).Should().BeTrue($"row {i} should be null when lag > data length");
        }
    }

    [Fact]
    public void LagFeatures_Lag_Zero_Should_Throw()
    {
        var act = () => new LagFeatures("Value", 0);
        act.Should().Throw<ArgumentOutOfRangeException>("lag=0 should be rejected");
    }
}

public class RollingFeaturesEdgeTests
{
    private static DataFrame MakeDf(int n)
    {
        var values = Enumerable.Range(1, n).Select(i => (double)i).ToArray();
        return new DataFrame(
            new Column<double>("Value", values));
    }

    [Fact]
    public void RollingFeatures_Window_Greater_Than_DataLength_Returns_All_Null()
    {
        var df = MakeDf(3);
        // window=10 > data length 3
        var rolling = new RollingFeatures("Value", 10);
        var result = rolling.FitTransform(df);

        var meanCol = result["Value_rolling_10_mean"];
        // All values should be null since window > data length
        for (int i = 0; i < df.RowCount; i++)
        {
            meanCol.IsNull(i).Should().BeTrue($"row {i} mean should be null when window > data length");
        }
    }
}

public class ExponentialSmoothingEdgeTests2
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
    public void ES_Alpha_Zero_Should_Throw_Or_Handle()
    {
        // alpha=0 means no smoothing - constructor should reject
        var act = () => new ExponentialSmoothing(ESType.Simple, alpha: 0.0);
        act.Should().Throw<ArgumentOutOfRangeException>("alpha=0 should be rejected");
    }

    [Fact]
    public void ES_Alpha_One_Should_Throw_Or_Handle()
    {
        // alpha=1 means full replacement - constructor should reject
        var act = () => new ExponentialSmoothing(ESType.Simple, alpha: 1.0);
        act.Should().Throw<ArgumentOutOfRangeException>("alpha=1 should be rejected");
    }

    [Fact]
    public void ES_Two_DataPoints_Should_Forecast_Finite_Values()
    {
        var df = MakeDf(new double[] { 10.0, 20.0 });
        var es = new ExponentialSmoothing(ESType.Simple, alpha: 0.5);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(3);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => double.IsFinite(v),
            "forecast from 2 data points should produce finite values");
    }

    [Fact]
    public void ES_Double_Two_DataPoints_Should_Forecast_Trend()
    {
        var df = MakeDf(new double[] { 10.0, 20.0 });
        var es = new ExponentialSmoothing(ESType.Double, alpha: 0.5, beta: 0.5);
        es.Fit(df, "Date", "Value");

        var result = es.Forecast(3);

        result.Values.Should().HaveCount(3);
        result.Values.Should().OnlyContain(v => double.IsFinite(v));
        // With an upward trend, forecasts should be > 20
        result.Values[0].Should().BeGreaterThan(15, "forecast should continue upward trend");
    }
}

public class AutoArimaConstantSeriesTests
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
    public void AutoARIMA_ConstantSeries_Should_Fit_And_Forecast()
    {
        // All same values - AutoARIMA should find ARIMA(0,0,0) as best model
        var data = Enumerable.Repeat(42.0, 30).ToArray();
        var df = MakeDf(data);
        var auto = new AutoARIMA(maxP: 2, maxD: 1, maxQ: 1);
        auto.Fit(df, "Date", "Value");

        auto.BestModel.Should().NotBeNull();

        var result = auto.Forecast(3);
        result.Values.Should().HaveCount(3);
        // For constant series, forecast should be approximately the constant value
        result.Values.Should().OnlyContain(v => double.IsFinite(v),
            "forecast on constant series should be finite");
    }
}

public class DateTimeFeaturesEdgeTests
{
    [Fact]
    public void DateTimeFeatures_Column_Not_DateTime_Should_Throw()
    {
        // Create a DataFrame with a non-DateTime column named "Date"
        var df = new DataFrame(
            new Column<double>("Date", new double[] { 1.0, 2.0, 3.0 }),
            new Column<double>("Value", new double[] { 10, 20, 30 }));

        var dtFeatures = new DateTimeFeatures("Date");

        // Transform should throw because "Date" is not a DateTime column
        var act = () => dtFeatures.FitTransform(df);
        act.Should().Throw<Exception>("DateTimeFeatures on non-DateTime column should throw");
    }
}
