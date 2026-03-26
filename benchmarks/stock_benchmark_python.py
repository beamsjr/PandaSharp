"""
PandaSharp vs pandas/scikit-learn Performance Comparison
========================================================
Same stock market analysis pipeline in Python for direct comparison.
Run: python3 benchmarks/stock_benchmark_python.py
"""

import os
import time
import glob
import json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

DATA_DIR = "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks"
OUTPUT_DIR = "stock_output_python"
os.makedirs(OUTPUT_DIR, exist_ok=True)

results = []
def lap(name, start, rows, detail=""):
    elapsed = int((time.time() - start) * 1000)
    results.append({"op": name, "ms": elapsed, "rows": rows, "detail": detail})
    return elapsed

print("=== Python (pandas + scikit-learn) Stock Benchmark ===\n")

# ── 1. LOAD ALL CSVs ──────────────────────────────────
t = time.time()
files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
frames = []
skipped = 0
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) < 100:
            skipped += 1
            continue
        ticker = os.path.basename(f).replace(".us.txt", "").upper()
        df["Ticker"] = ticker
        frames.append(df)
    except:
        skipped += 1

all_stocks = pd.concat(frames, ignore_index=True)
ms = lap("Load CSVs", t, len(all_stocks), f"{len(frames)} files, {skipped} skipped")
print(f"1. Load CSVs: {len(frames)} stocks, {len(all_stocks):,} rows [{ms}ms]")

# ── 2. SCHEMA VALIDATION ──────────────────────────────
t = time.time()
required = {"Date", "Open", "High", "Low", "Close", "Volume"}
valid = required.issubset(set(all_stocks.columns))
ms = lap("Schema Validation", t, len(all_stocks), "PASSED" if valid else "FAILED")
print(f"2. Schema validation: {'PASSED' if valid else 'FAILED'} [{ms}ms]")

# ── 3. PROFILE ────────────────────────────────────────
t = time.time()
profile = all_stocks.head(10_000).describe()
ms = lap("Profile (10K)", t, 10_000, f"{len(profile.columns)} columns")
print(f"3. Profile: done [{ms}ms]")

# ── 4. EVAL ───────────────────────────────────────────
t = time.time()
all_stocks["DailyReturn"] = (all_stocks["Close"] - all_stocks["Open"]) / all_stocks["Open"] * 100
ms = lap("Eval (DailyReturn)", t, len(all_stocks), "Added computed column")
print(f"4. Eval: done [{ms}ms]")

# ── 5. NLARGEST ───────────────────────────────────────
t = time.time()
big_movers = all_stocks.nlargest(10, "DailyReturn")[["Ticker", "Date", "Open", "Close", "DailyReturn", "Volume"]]
ms = lap("Nlargest (top 10)", t, len(all_stocks), "Top 10 by DailyReturn")
print(f"5. Nlargest: done [{ms}ms]")

# ── 6. GROUPBY ────────────────────────────────────────
t = time.time()
avg_returns = all_stocks.groupby("Ticker")["DailyReturn"].mean().sort_values(ascending=False)
ms = lap("GroupBy Mean", t, len(all_stocks), f"{len(frames)} groups")
print(f"6. GroupBy mean: done [{ms}ms]")

# ── 7. CHAINED OPS ────────────────────────────────────
t = time.time()
lazy_result = (all_stocks[all_stocks["Volume"] > 10_000_000]
    .sort_values("DailyReturn", ascending=False)
    [["Ticker", "Date", "DailyReturn", "Volume"]]
    .head(20))
ms = lap("Chained Filter+Sort+Select", t, len(all_stocks), f"{len(lazy_result)} results")
print(f"7. Chained ops: done [{ms}ms]")

# ── 8. ROLLING WINDOW ─────────────────────────────────
t = time.time()
aapl = all_stocks[all_stocks["Ticker"] == "AAPL"].copy()
aapl["SMA20"] = aapl["Close"].rolling(20).mean()
aapl["SMA50"] = aapl["Close"].rolling(50).mean()
ms = lap("Rolling Window", t, len(aapl), "SMA20 + SMA50")
print(f"8. Rolling window: done ({len(aapl)} rows) [{ms}ms]")

# ── 9. QCUT ──────────────────────────────────────────
t = time.time()
valid_returns = all_stocks["DailyReturn"].dropna()
quartiles = pd.qcut(valid_returns, 4, labels=["Crash", "Down", "Up", "Surge"])
bin_counts = quartiles.value_counts().to_dict()
ms = lap("QCut Binning", t, len(all_stocks), "4 quartiles")
print(f"9. QCut: done [{ms}ms]")

# ── 10. ML: LOGISTIC REGRESSION ──────────────────────
t = time.time()
ml_accuracy = "N/A"
ml_f1 = "N/A"

if len(aapl) > 500:
    closes = aapl["Close"].values
    highs = aapl["High"].values
    lows = aapl["Low"].values
    vols = aapl["Volume"].values
    opens = aapl["Open"].values
    N = len(closes)
    lookback = 50
    usable = N - lookback - 1

    feat_return1 = np.zeros(usable)
    feat_return5 = np.zeros(usable)
    feat_return20 = np.zeros(usable)
    feat_vol_ratio = np.zeros(usable)
    feat_range = np.zeros(usable)
    feat_sma_ratio = np.zeros(usable)
    feat_gap = np.zeros(usable)
    labels_arr = np.zeros(usable, dtype=int)

    for i in range(usable):
        idx = i + lookback
        feat_return1[i] = (closes[idx] - closes[idx-1]) / closes[idx-1] * 100 if closes[idx-1] > 0 else 0
        feat_return5[i] = (closes[idx] - closes[idx-5]) / closes[idx-5] * 100 if closes[idx-5] > 0 else 0
        feat_return20[i] = (closes[idx] - closes[idx-20]) / closes[idx-20] * 100 if closes[idx-20] > 0 else 0
        avg_vol = np.mean(vols[idx-20:idx])
        feat_vol_ratio[i] = vols[idx] / avg_vol if avg_vol > 0 else 1
        feat_range[i] = (highs[idx] - lows[idx]) / closes[idx] * 100 if closes[idx] > 0 else 0
        sma20 = np.mean(closes[idx-20:idx])
        feat_sma_ratio[i] = closes[idx] / sma20 if sma20 > 0 else 1
        feat_gap[i] = (opens[idx] - closes[idx-1]) / closes[idx-1] * 100 if closes[idx-1] > 0 else 0
        labels_arr[i] = 1 if closes[idx+1] > closes[idx] else 0

    X = np.column_stack([feat_return1, feat_return5, feat_return20, feat_vol_ratio, feat_range, feat_sma_ratio, feat_gap])
    y = labels_arr

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler_ml = StandardScaler()
    X_train_scaled = scaler_ml.fit_transform(X_train)
    X_test_scaled = scaler_ml.transform(X_test)

    model = LogisticRegression(max_iter=200, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    ml_accuracy = f"{accuracy_score(y_test, y_pred):.1%}"
    ml_f1 = f"{f1_score(y_test, y_pred):.1%}"

ms = lap("ML: Logistic Regression", t, len(aapl), f"Acc: {ml_accuracy}, F1: {ml_f1}")
print(f"10. ML: done — Accuracy: {ml_accuracy}, F1: {ml_f1} [{ms}ms]")

# ── 11. PARQUET ──────────────────────────────────────
t = time.time()
pq_path = os.path.join(OUTPUT_DIR, "all_stocks.parquet")
all_stocks.head(100_000).to_parquet(pq_path)
reloaded = pd.read_parquet(pq_path)
ms = lap("Parquet Write+Read", t, 100_000, f"{os.path.getsize(pq_path) // 1024:,} KB")
print(f"11. Parquet: done ({os.path.getsize(pq_path) // 1024:,} KB) [{ms}ms]")

# ── 12. FILTER ────────────────────────────────────────
t = time.time()
big_days = all_stocks[all_stocks["DailyReturn"].abs() > 5]
ms = lap("Filter (abs > 5%)", t, len(all_stocks), f"{len(big_days):,} matches")
print(f"12. Filter: done — {len(big_days):,} matches [{ms}ms]")

# ── 13. SAVE ─────────────────────────────────────────
t = time.time()
big_movers.to_csv(os.path.join(OUTPUT_DIR, "big_movers.csv"), index=False)
big_movers.to_json(os.path.join(OUTPUT_DIR, "big_movers.json"), orient="records")
ms = lap("Save CSV + JSON", t, len(big_movers), "CSV + JSON")
print(f"13. Save: done [{ms}ms]")

# ── SUMMARY ──────────────────────────────────────────
total_ms = sum(r["ms"] for r in results)
print(f"\n=== Python Total: {total_ms:,} ms ===\n")

# Save results as JSON for comparison
with open(os.path.join(OUTPUT_DIR, "python_perf.json"), "w") as f:
    json.dump(results, f, indent=2)

# Print comparison table
print(f"{'Operation':<30} {'Time (ms)':>10} {'Rows':>15}")
print("-" * 60)
for r in results:
    print(f"{r['op']:<30} {r['ms']:>10,} {r['rows']:>15,}")
print("-" * 60)
print(f"{'TOTAL':<30} {total_ms:>10,}")
