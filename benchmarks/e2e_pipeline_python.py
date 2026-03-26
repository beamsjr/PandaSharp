"""
End-to-End Pipeline Benchmark: Python Ecosystem
Measures TOTAL wall-clock including all library boundary crossings.

Workflows:
  1. ML Pipeline:        Load → clean → feature engineer → train → evaluate → export
  2. Time Series:        Load → resample → decompose → forecast → evaluate
  3. Geospatial + Agg:   Load → spatial join → group by region → aggregate stats
  4. NLP Pipeline:        Load → tokenize → vectorize → classify → evaluate
  5. Full Data Science:   Load → EDA → multi-model comparison → best model report

Each workflow uses the natural Python library stack:
  pandas + scikit-learn + statsmodels + geopandas + nltk/sklearn.text
"""

import time
import json
import warnings
import os
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

results = []

def timed(name, workflow, func):
    """Run func, record wall-clock ms."""
    import gc
    gc.collect()
    start = time.perf_counter()
    detail = func()
    elapsed_ms = (time.perf_counter() - start) * 1000
    results.append({
        "workflow": workflow,
        "operation": name,
        "ms": round(elapsed_ms, 2),
        "detail": str(detail) if detail else ""
    })
    print(f"  {name}: {elapsed_ms:.1f}ms" + (f"  ({detail})" if detail else ""))
    return elapsed_ms

# ════════════════════════════════════════════════════════════════
# DATA GENERATION — identical synthetic data for both benchmarks
# ════════════════════════════════════════════════════════════════

ROWS = 500_000
SEED = 42
np.random.seed(SEED)

print(f"\n{'='*60}")
print(f"  End-to-End Pipeline Benchmark — Python Ecosystem")
print(f"  Rows: {ROWS:,}")
print(f"{'='*60}\n")

# Pre-generate data
categories = ["Tech", "Finance", "Health", "Energy", "Consumer"]
cities = ["New York", "London", "Tokyo", "Sydney", "Berlin", "Toronto", "Mumbai", "São Paulo"]
texts = [
    "The quarterly revenue exceeded expectations with strong growth",
    "Market downturn caused significant losses across portfolios",
    "New product launch received positive customer feedback",
    "Regulatory changes impacted operations in multiple regions",
    "Strategic partnership announced to expand market presence",
    "Cost reduction initiative achieved target savings ahead of schedule",
    "Customer satisfaction scores improved significantly this quarter",
    "Supply chain disruptions affected delivery timelines globally",
    "Innovation investment increased to accelerate digital transformation",
    "Employee retention rates improved following new benefits package",
]

data_cat    = np.random.choice(categories, ROWS)
data_city   = np.random.choice(cities, ROWS)
data_text   = np.random.choice(texts, ROWS)
data_value  = np.random.normal(100, 25, ROWS)
data_price  = np.abs(np.random.normal(50, 15, ROWS))
data_volume = np.random.randint(100, 10000, ROWS).astype(float)
data_lat    = np.random.uniform(25, 48, ROWS)
data_lon    = np.random.uniform(-125, -65, ROWS)
data_dates  = np.array([f"2020-{(i % 12)+1:02d}-{(i % 28)+1:02d}" for i in range(ROWS)])
data_label  = (data_value > 100).astype(int)

# Inject ~5% nulls
null_mask = np.random.random(ROWS) < 0.05
data_value_with_nulls = data_value.copy()
data_value_with_nulls[null_mask] = np.nan

# ════════════════════════════════════════════════════════════════
print("─" * 60)
print("  WORKFLOW 1: ML Pipeline")
print("  pandas → numpy → scikit-learn → back to pandas")
print("─" * 60)

workflow1_total = 0

def w1_step1():
    global w1_df
    w1_df = pd.DataFrame({
        "category": data_cat,
        "value": data_value_with_nulls,
        "price": data_price,
        "volume": data_volume,
        "label": data_label,
    })
    return f"{len(w1_df):,} rows"
workflow1_total += timed("1a. Load into pandas", "ML Pipeline", w1_step1)

def w1_step2():
    global w1_clean
    df = w1_df.dropna()
    df = df[df["volume"] > 200].copy()
    df["total_value"] = df["price"] * df["volume"]
    df["log_value"] = np.log(df["value"])
    df["price_ratio"] = df["price"] / df["price"].mean()
    w1_clean = df
    return f"{len(w1_clean):,} rows after clean"
workflow1_total += timed("1b. Clean + feature engineer (pandas)", "ML Pipeline", w1_step2)

def w1_step3():
    """CONVERSION BOUNDARY: pandas → numpy for scikit-learn."""
    global w1_X, w1_y
    feature_cols = ["value", "price", "volume", "total_value", "log_value", "price_ratio"]
    w1_X = w1_clean[feature_cols].values  # .values forces copy to numpy
    w1_y = w1_clean["label"].values
    return f"X shape: {w1_X.shape}"
workflow1_total += timed("1c. ★ CONVERT: pandas → numpy", "ML Pipeline", w1_step3)

def w1_step4():
    from sklearn.preprocessing import StandardScaler
    global w1_X_scaled, w1_scaler
    w1_scaler = StandardScaler()
    w1_X_scaled = w1_scaler.fit_transform(w1_X)
    return None
workflow1_total += timed("1d. StandardScaler fit+transform", "ML Pipeline", w1_step4)

def w1_step5():
    from sklearn.model_selection import train_test_split
    global w1_X_train, w1_X_test, w1_y_train, w1_y_test
    w1_X_train, w1_X_test, w1_y_train, w1_y_test = train_test_split(
        w1_X_scaled, w1_y, test_size=0.2, random_state=SEED
    )
    return f"train={len(w1_X_train):,}, test={len(w1_X_test):,}"
workflow1_total += timed("1e. Train/test split", "ML Pipeline", w1_step5)

def w1_step6():
    from sklearn.ensemble import RandomForestClassifier
    global w1_model
    w1_model = RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1)
    w1_model.fit(w1_X_train, w1_y_train)
    return None
workflow1_total += timed("1f. Train RandomForest (100 trees)", "ML Pipeline", w1_step6)

def w1_step7():
    from sklearn.metrics import accuracy_score, f1_score
    global w1_preds
    w1_preds = w1_model.predict(w1_X_test)
    acc = accuracy_score(w1_y_test, w1_preds)
    f1 = f1_score(w1_y_test, w1_preds)
    return f"acc={acc:.4f}, f1={f1:.4f}"
workflow1_total += timed("1g. Predict + evaluate", "ML Pipeline", w1_step7)

def w1_step8():
    """CONVERSION BOUNDARY: numpy predictions → back to pandas DataFrame."""
    result_df = pd.DataFrame({
        "actual": w1_y_test,
        "predicted": w1_preds,
        "correct": (w1_y_test == w1_preds),
    })
    return f"result: {result_df.shape}"
workflow1_total += timed("1h. ★ CONVERT: numpy → pandas results", "ML Pipeline", w1_step8)

results.append({"workflow": "ML Pipeline", "operation": "TOTAL", "ms": round(workflow1_total, 2), "detail": ""})
print(f"  {'─'*40}")
print(f"  ML Pipeline TOTAL: {workflow1_total:.1f}ms\n")


# ════════════════════════════════════════════════════════════════
print("─" * 60)
print("  WORKFLOW 2: Time Series Forecasting")
print("  pandas → statsmodels (already pandas) → pandas")
print("─" * 60)

workflow2_total = 0

def w2_step1():
    global w2_df
    dates = pd.date_range(start="2015-01-01", end="2025-12-31", freq="D")
    n = len(dates)
    np.random.seed(SEED)
    trend = np.linspace(100, 300, n)
    seasonal = 20 * np.sin(2 * np.pi * np.arange(n) / 365.25)
    noise = np.random.normal(0, 5, n)
    w2_df = pd.DataFrame({"date": dates, "value": trend + seasonal + noise})
    return f"{n:,} daily observations"
workflow2_total += timed("2a. Generate time series (pandas)", "TimeSeries", w2_step1)

def w2_step2():
    global w2_monthly
    w2_df_indexed = w2_df.set_index("date")
    w2_monthly = w2_df_indexed.resample("MS").mean()
    return f"{len(w2_monthly)} months"
workflow2_total += timed("2b. Resample to monthly (pandas)", "TimeSeries", w2_step2)

def w2_step3():
    """NO CONVERSION — statsmodels works with pandas natively."""
    # For pandas-based workflow, no conversion needed here
    # but we still need to ensure freq is set
    global w2_series
    w2_series = w2_monthly["value"]
    w2_series.index.freq = "MS"
    return None
workflow2_total += timed("2c. Prepare series (no conversion — pandas native)", "TimeSeries", w2_step3)

def w2_step4():
    from statsmodels.tsa.seasonal import seasonal_decompose
    global w2_decomp
    w2_decomp = seasonal_decompose(w2_series, model="additive", period=12)
    return None
workflow2_total += timed("2d. Seasonal decomposition (statsmodels)", "TimeSeries", w2_step4)

def w2_step5():
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    global w2_hw_fit
    train = w2_series[:-12]
    model = ExponentialSmoothing(train, seasonal_periods=12, trend="add", seasonal="add")
    w2_hw_fit = model.fit(optimized=True)
    return None
workflow2_total += timed("2e. Holt-Winters fit (statsmodels)", "TimeSeries", w2_step5)

def w2_step6():
    global w2_forecast
    w2_forecast = w2_hw_fit.forecast(12)
    return f"forecast len={len(w2_forecast)}"
workflow2_total += timed("2f. Forecast 12 months", "TimeSeries", w2_step6)

def w2_step7():
    actual = w2_series[-12:].values
    predicted = w2_forecast.values
    mae = np.mean(np.abs(actual - predicted))
    rmse = np.sqrt(np.mean((actual - predicted) ** 2))
    return f"MAE={mae:.2f}, RMSE={rmse:.2f}"
workflow2_total += timed("2g. Evaluate (MAE, RMSE)", "TimeSeries", w2_step7)

results.append({"workflow": "TimeSeries", "operation": "TOTAL", "ms": round(workflow2_total, 2), "detail": ""})
print(f"  {'─'*40}")
print(f"  TimeSeries TOTAL: {workflow2_total:.1f}ms\n")


# ════════════════════════════════════════════════════════════════
print("─" * 60)
print("  WORKFLOW 3: Geospatial Analytics")
print("  pandas → geopandas/shapely → back to pandas")
print("─" * 60)

workflow3_total = 0

def w3_step1():
    global w3_df
    w3_df = pd.DataFrame({
        "city": data_city[:ROWS],
        "lat": data_lat[:ROWS],
        "lon": data_lon[:ROWS],
        "value": data_value[:ROWS],
        "category": data_cat[:ROWS],
    })
    return f"{len(w3_df):,} rows"
workflow3_total += timed("3a. Load locations (pandas)", "Geospatial", w3_step1)

def w3_step2():
    """CONVERSION BOUNDARY: pandas → geopandas."""
    import geopandas as gpd
    from shapely.geometry import Point
    global w3_gdf
    geometry = [Point(lon, lat) for lon, lat in zip(w3_df["lon"], w3_df["lat"])]
    w3_gdf = gpd.GeoDataFrame(w3_df, geometry=geometry, crs="EPSG:4326")
    return f"GeoDataFrame: {len(w3_gdf):,} rows"
workflow3_total += timed("3b. ★ CONVERT: pandas → GeoDataFrame", "Geospatial", w3_step2)

def w3_step3():
    import geopandas as gpd
    from shapely.geometry import box
    global w3_joined
    regions = []
    region_names = []
    idx = 0
    for lat_start in range(25, 48, 5):
        for lon_start in range(-125, -65, 10):
            regions.append(box(lon_start, lat_start, lon_start + 10, lat_start + 5))
            region_names.append(f"Region_{idx}")
            idx += 1
    regions_gdf = gpd.GeoDataFrame(
        {"region": region_names}, geometry=regions, crs="EPSG:4326"
    )
    w3_joined = gpd.sjoin(w3_gdf, regions_gdf, how="left", predicate="within")
    return f"{len(w3_joined):,} joined rows, {len(regions)} regions"
workflow3_total += timed("3c. Spatial join (geopandas)", "Geospatial", w3_step3)

def w3_step4():
    global w3_agg
    w3_agg = w3_joined.groupby("region").agg(
        count=("value", "count"),
        mean_value=("value", "mean"),
        sum_value=("value", "sum"),
        std_value=("value", "std"),
    ).reset_index()
    return f"{len(w3_agg)} regions aggregated"
workflow3_total += timed("3d. GroupBy region + aggregate", "Geospatial", w3_step4)

def w3_step5():
    lats = w3_gdf["lat"].values
    lons = w3_gdf["lon"].values
    ref_lat, ref_lon = 40.7128, -74.0060
    lat1 = np.radians(lats)
    lat2 = np.radians(ref_lat)
    dlat = np.radians(ref_lat - lats)
    dlon = np.radians(ref_lon - lons)
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return f"distances computed for {len(distances):,} points"
workflow3_total += timed("3e. Haversine distances (numpy)", "Geospatial", w3_step5)

def w3_step6():
    """CONVERSION BOUNDARY: geopandas results → back to pandas."""
    result_df = pd.DataFrame(w3_agg.drop(columns="geometry", errors="ignore"))
    return f"result: {result_df.shape}"
workflow3_total += timed("3f. ★ CONVERT: geopandas → pandas results", "Geospatial", w3_step6)

results.append({"workflow": "Geospatial", "operation": "TOTAL", "ms": round(workflow3_total, 2), "detail": ""})
print(f"  {'─'*40}")
print(f"  Geospatial TOTAL: {workflow3_total:.1f}ms\n")


# ════════════════════════════════════════════════════════════════
print("─" * 60)
print("  WORKFLOW 4: NLP / Text Classification")
print("  pandas → sklearn.text → sklearn.model → back to pandas")
print("─" * 60)

workflow4_total = 0
NLP_ROWS = 200_000

def w4_step1():
    global w4_df
    w4_df = pd.DataFrame({
        "text": data_text[:NLP_ROWS],
        "category": data_cat[:NLP_ROWS],
        "label": data_label[:NLP_ROWS],
    })
    return f"{len(w4_df):,} documents"
workflow4_total += timed("4a. Load text data (pandas)", "NLP", w4_step1)

def w4_step2():
    global w4_clean
    w4_clean = w4_df.copy()
    w4_clean["text_clean"] = w4_clean["text"].str.lower()
    return None
workflow4_total += timed("4b. Lowercase (pandas)", "NLP", w4_step2)

def w4_step3():
    """CONVERSION BOUNDARY: pandas → python lists for sklearn."""
    global w4_texts, w4_labels
    w4_texts = w4_clean["text_clean"].tolist()
    w4_labels = w4_clean["label"].values
    return None
workflow4_total += timed("4c. ★ CONVERT: pandas → Python lists", "NLP", w4_step3)

def w4_step4():
    from sklearn.feature_extraction.text import TfidfVectorizer
    global w4_tfidf, w4_X
    w4_tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    w4_X = w4_tfidf.fit_transform(w4_texts)
    return f"matrix: {w4_X.shape}"
workflow4_total += timed("4d. TF-IDF vectorize (sklearn)", "NLP", w4_step4)

def w4_step5():
    from sklearn.model_selection import train_test_split
    global w4_X_train, w4_X_test, w4_y_train, w4_y_test
    w4_X_train, w4_X_test, w4_y_train, w4_y_test = train_test_split(
        w4_X, w4_labels, test_size=0.2, random_state=SEED
    )
    return None
workflow4_total += timed("4e. Train/test split", "NLP", w4_step5)

def w4_step6():
    from sklearn.linear_model import LogisticRegression
    global w4_model
    w4_model = LogisticRegression(max_iter=200, random_state=SEED)
    w4_model.fit(w4_X_train, w4_y_train)
    return None
workflow4_total += timed("4f. Train LogisticRegression", "NLP", w4_step6)

def w4_step7():
    from sklearn.metrics import accuracy_score
    global w4_preds
    w4_preds = w4_model.predict(w4_X_test)
    acc = accuracy_score(w4_y_test, w4_preds)
    return f"acc={acc:.4f}"
workflow4_total += timed("4g. Predict + evaluate", "NLP", w4_step7)

def w4_step8():
    """CONVERSION BOUNDARY: numpy results → back to pandas."""
    result_df = pd.DataFrame({
        "actual": w4_y_test,
        "predicted": w4_preds,
    })
    return f"result: {result_df.shape}"
workflow4_total += timed("4h. ★ CONVERT: numpy → pandas results", "NLP", w4_step8)

results.append({"workflow": "NLP", "operation": "TOTAL", "ms": round(workflow4_total, 2), "detail": ""})
print(f"  {'─'*40}")
print(f"  NLP Pipeline TOTAL: {workflow4_total:.1f}ms\n")


# ════════════════════════════════════════════════════════════════
print("─" * 60)
print("  WORKFLOW 5: Multi-Model Comparison (Full Data Science)")
print("  pandas → numpy → sklearn (×5 models) → pandas report")
print("─" * 60)

workflow5_total = 0

def w5_step1():
    global w5_df
    df = pd.DataFrame({
        "value": data_value[:ROWS],
        "price": data_price[:ROWS],
        "volume": data_volume[:ROWS],
        "label": data_label[:ROWS],
    })
    df["total"] = df["price"] * df["volume"]
    df["value_ma10"] = df["value"].rolling(window=10).mean()
    df["price_diff"] = df["price"].diff()
    w5_df = df.dropna()
    return f"{len(w5_df):,} rows, {len(w5_df.columns)} features"
workflow5_total += timed("5a. Feature engineering (pandas)", "MultiModel", w5_step1)

def w5_step2():
    """CONVERSION BOUNDARY: pandas → numpy."""
    global w5_X, w5_y
    feature_cols = ["value", "price", "volume", "total", "value_ma10", "price_diff"]
    w5_X = w5_df[feature_cols].values
    w5_y = w5_df["label"].values
    return f"X shape: {w5_X.shape}"
workflow5_total += timed("5b. ★ CONVERT: pandas → numpy", "MultiModel", w5_step2)

def w5_step3():
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    global w5_X_train, w5_X_test, w5_y_train, w5_y_test
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(w5_X)
    w5_X_train, w5_X_test, w5_y_train, w5_y_test = train_test_split(
        X_scaled, w5_y, test_size=0.2, random_state=SEED
    )
    return None
workflow5_total += timed("5c. Scale + split", "MultiModel", w5_step3)

def w5_step4():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, f1_score

    global w5_model_results
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200, random_state=SEED),
        "DecisionTree": DecisionTreeClassifier(max_depth=10, random_state=SEED),
        "RandomForest": RandomForestClassifier(n_estimators=50, random_state=SEED, n_jobs=-1),
        "GBT": GradientBoostingClassifier(n_estimators=50, random_state=SEED),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }
    w5_model_results = []
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(w5_X_train, w5_y_train)
        preds = model.predict(w5_X_test)
        train_ms = (time.perf_counter() - t0) * 1000
        acc = accuracy_score(w5_y_test, preds)
        f1 = f1_score(w5_y_test, preds)
        w5_model_results.append({
            "model": name, "accuracy": acc, "f1": f1, "train_ms": train_ms
        })
        print(f"    {name}: acc={acc:.4f}, f1={f1:.4f} [{train_ms:.0f}ms]")
    return f"{len(models)} models trained"
workflow5_total += timed("5d. Train 5 models + evaluate", "MultiModel", w5_step4)

def w5_step5():
    """CONVERSION BOUNDARY: Python dicts → pandas comparison DataFrame."""
    comparison = pd.DataFrame(w5_model_results)
    best = comparison.sort_values("f1", ascending=False).head(1)
    return f"best model: {best['model'].values[0]}"
workflow5_total += timed("5e. ★ CONVERT: results → pandas report", "MultiModel", w5_step5)

results.append({"workflow": "MultiModel", "operation": "TOTAL", "ms": round(workflow5_total, 2), "detail": ""})
print(f"  {'─'*40}")
print(f"  MultiModel TOTAL: {workflow5_total:.1f}ms\n")


# ════════════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print("  SUMMARY — Python Ecosystem End-to-End")
print(f"{'='*60}")

totals = [r for r in results if r["operation"] == "TOTAL"]
conversions = [r for r in results if "CONVERT" in r["operation"]]

total_all = sum(t["ms"] for t in totals)
total_convert = sum(c["ms"] for c in conversions)

print(f"\n  {'Workflow':<20} {'Total':>10}")
print(f"  {'─'*32}")
for t in totals:
    print(f"  {t['workflow']:<20} {t['ms']:>8.1f}ms")
print(f"  {'─'*32}")
print(f"  {'ALL WORKFLOWS':<20} {total_all:>8.1f}ms")
print(f"\n  Conversion overhead: {total_convert:.1f}ms ({total_convert/total_all*100:.1f}% of total)")
print(f"  Conversions counted: {len(conversions)}")

# Save results
output = {
    "language": "python",
    "rows": ROWS,
    "results": results,
    "summary": {
        "total_ms": round(total_all, 2),
        "conversion_ms": round(total_convert, 2),
        "conversion_pct": round(total_convert / total_all * 100, 2),
        "num_conversions": len(conversions),
    }
}

with open("e2e_pipeline_python_results.json", "w") as f:
    json.dump(output, f, indent=2)

print(f"\n  Results saved to e2e_pipeline_python_results.json")
