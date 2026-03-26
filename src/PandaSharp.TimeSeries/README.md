# PandaSharp.TimeSeries

Time series forecasting and analysis for PandaSharp: ARIMA, SARIMA, ExponentialSmoothing, decomposition, and diagnostics.

## Features

- **Forecasting models** — ARIMA, SARIMA, and Exponential Smoothing (Holt-Winters)
- **Seasonal decomposition** — additive and multiplicative decomposition
- **Stationarity tests** — ADF, KPSS, and autocorrelation analysis
- **Diagnostics** — residual plots, Ljung-Box test, ACF/PACF
- **Rolling and expanding window** statistics

## Installation

```bash
dotnet add package PandaSharp.TimeSeries
```

## Quick Start

```csharp
using PandaSharp;
using PandaSharp.TimeSeries;

var df = DataFrame.ReadCsv("monthly_sales.csv", parseDate: "date");

var model = new ARIMA(p: 1, d: 1, q: 1);
model.Fit(df["revenue"]);

var forecast = model.Forecast(steps: 12);
forecast.Print();
```

## Links

- [GitHub Repository](https://github.com/beamsjr/PandaSharp)
- [License: MIT](https://opensource.org/licenses/MIT)
