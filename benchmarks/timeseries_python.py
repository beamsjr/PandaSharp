import time, json, os, numpy as np
import statsmodels.api as sm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
try: from pmdarima import auto_arima
except: auto_arima = None

os.makedirs("ts_bench_output", exist_ok=True)
results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")

# Generate data: trend + seasonal + noise
np.random.seed(42)
N = 5000
trend = np.linspace(10, 50, N)
seasonal = 5 * np.sin(2 * np.pi * np.arange(N) / 12)
noise = np.random.randn(N) * 0.5
series = trend + seasonal + noise

print(f"=== Python TimeSeries Benchmark ({N} points) ===\n")

# Forecasting
print("── Forecasting ──")
t = time.perf_counter()
model = ARIMA(series, order=(2,1,1)).fit()
model.forecast(50)
lap("Forecast", "ARIMA(2,1,1) fit+forecast(50)", t)

t = time.perf_counter()
model = ExponentialSmoothing(series, seasonal_periods=12, trend='add', seasonal='add').fit()
model.forecast(50)
lap("Forecast", "Holt-Winters fit+forecast(50)", t)

if auto_arima:
    t = time.perf_counter()
    auto_arima(series[:1000], max_p=3, max_q=3, max_d=2, seasonal=False, stepwise=True, suppress_warnings=True)
    lap("Forecast", "AutoARIMA (1K points)", t)

# Decomposition
print("\n── Decomposition ──")
t = time.perf_counter()
result = seasonal_decompose(series, model='additive', period=12)
lap("Decompose", "Seasonal decompose (additive)", t)

# Diagnostics
print("\n── Diagnostics ──")
t = time.perf_counter()
adfuller(series)
lap("Diagnostic", "ADF test", t)

t = time.perf_counter()
kpss(series, regression='c')
lap("Diagnostic", "KPSS test", t)

t = time.perf_counter()
acf(series, nlags=50)
lap("Diagnostic", "ACF (50 lags)", t)

t = time.perf_counter()
pacf(series, nlags=50)
lap("Diagnostic", "PACF (50 lags)", t)

# Summary
print(f"\n{'═'*70}")
cats = {}
for r in results: cats[r["category"]] = cats.get(r["category"], 0) + r["ms"]
for c, m in sorted(cats.items(), key=lambda x: -x[1]): print(f"  {c:<30} {m:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
with open("ts_bench_output/python_ts_results.json", "w") as f: json.dump(results, f, indent=2)
