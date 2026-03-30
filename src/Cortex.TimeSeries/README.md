# Cortex.TimeSeries

Time series forecasting and analysis for Cortex: ARIMA, SARIMA, ExponentialSmoothing, decomposition, and diagnostics.

Part of the [Cortex](https://github.com/beamsjr/Cortex) data science ecosystem for .NET.

> Requires .NET 10+ and the `Cortex` core package.

## Features

- **Forecasting models** — ARIMA, SARIMA, and Exponential Smoothing (Holt-Winters)
- **AutoARIMA** — automatic model order selection (p, d, q) via information criteria
- **Seasonal decomposition** — additive and multiplicative decomposition
- **Stationarity tests** — ADF, KPSS, and autocorrelation analysis
- **Diagnostics** — residual plots, Ljung-Box test, ACF/PACF
- **Rolling and expanding window** statistics

## Installation

```bash
dotnet add package Cortex.TimeSeries
```

## Quick Start

```csharp
using Cortex;
using Cortex.TimeSeries;

var df = DataFrame.ReadCsv("monthly_sales.csv", parseDate: "date");

var model = new ARIMA(p: 1, d: 1, q: 1);
model.Fit(df["revenue"]);

var forecast = model.Forecast(steps: 12);
forecast.Print();
```

## Automatic Model Selection

```csharp
var auto = new AutoARIMA(maxP: 5, maxD: 2, maxQ: 5);
auto.Fit(data);
var forecast = auto.Forecast(steps: 30);
```

## Seasonal Decomposition

```csharp
var decomposition = df["revenue"].Decompose(period: 12, model: "additive");
// Access: decomposition.Trend, decomposition.Seasonal, decomposition.Residual
```

## Performance

131x faster than Python on average — Holt-Winters 699x, AutoARIMA 258x, ARIMA 33x.

## Related Packages

| Package | Description |
|---------|-------------|
| **Cortex** | Core DataFrame (required) |
| **Cortex.Viz** | Chart forecasts with Plotly |
| **Cortex.Streaming** | Real-time streaming with windowed aggregations |

## Links

- [GitHub Repository](https://github.com/beamsjr/Cortex)
- [License: MIT](https://opensource.org/licenses/MIT)
