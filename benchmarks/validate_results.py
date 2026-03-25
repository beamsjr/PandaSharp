"""
Validate PandaSharp results match Python/pandas results.
Runs both implementations and compares outputs.
"""
import os, glob, json, numpy as np, pandas as pd

DATA_DIR = "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks"

# Load data (same as benchmark)
files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
frames = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) < 252: continue
        ticker = os.path.basename(f).replace(".us.txt", "").upper()
        df["Ticker"] = ticker
        frames.append(df)
    except: pass
all_stocks = pd.concat(frames, ignore_index=True)
tickers = all_stocks["Ticker"].unique()
n_tickers = len(tickers)
ticker_to_id = {t: i for i, t in enumerate(tickers)}
all_stocks["TickerId"] = all_stocks["Ticker"].map(ticker_to_id)

print(f"Loaded {len(frames)} stocks, {len(all_stocks):,} rows")
print("="*60)

results = {}

# 1. String operations
print("\n1. String Operations")
results["str_upper_first5"] = all_stocks["Ticker"].str.upper().head(5).tolist()
results["str_lower_first5"] = all_stocks["Ticker"].str.lower().head(5).tolist()
results["str_contains_A_sum"] = int(all_stocks["Ticker"].str.contains("A").sum())
results["str_len_mean"] = float(all_stocks["Ticker"].str.len().mean())
results["str_startswith_AA_sum"] = int(all_stocks["Ticker"].str.startswith("AA").sum())
results["str_replace_first5"] = all_stocks["Ticker"].str.replace("A", "X").head(5).tolist()
results["str_slice_first5"] = all_stocks["Ticker"].str.slice(0, 2).head(5).tolist()
for k, v in results.items():
    if k.startswith("str_"): print(f"  {k}: {v}")

# 2. Expression chains
print("\n2. Expression Chains")
all_stocks["DailyReturn"] = (all_stocks["Close"] - all_stocks["Open"]) / all_stocks["Open"] * 100
all_stocks["Spread"] = (all_stocks["High"] - all_stocks["Low"]) / all_stocks["Close"] * 100
mask = (all_stocks["DailyReturn"] > 2) & (all_stocks["Volume"] > 1e6) & (all_stocks["Close"] > 5)
filtered = all_stocks[mask]
results["daily_return_mean"] = float(all_stocks["DailyReturn"].mean())
results["daily_return_std"] = float(all_stocks["DailyReturn"].std())
results["spread_mean"] = float(all_stocks["Spread"].mean())
results["complex_filter_count"] = int(len(filtered))
for k in ["daily_return_mean", "daily_return_std", "spread_mean", "complex_filter_count"]:
    print(f"  {k}: {results[k]}")

# 3. GroupBy aggregation
print("\n3. Multi-Aggregation")
agg = all_stocks.groupby("Ticker").agg(
    MeanClose=("Close", "mean"), StdClose=("Close", "std"),
    MinClose=("Close", "min"), MaxClose=("Close", "max"),
    Count=("Close", "count"))
results["agg_mean_of_means"] = float(agg["MeanClose"].mean())
results["agg_total_count"] = int(agg["Count"].sum())
results["agg_min_of_mins"] = float(agg["MinClose"].min())
results["agg_max_of_maxs"] = float(agg["MaxClose"].max())
for k in ["agg_mean_of_means", "agg_total_count", "agg_min_of_mins", "agg_max_of_maxs"]:
    print(f"  {k}: {results[k]}")

# 4. Window functions
print("\n4. Window Functions")
sma20 = all_stocks.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
results["sma20_non_nan_count"] = int(sma20.notna().sum())
results["sma20_mean"] = float(sma20.mean())
expand_max = all_stocks.groupby("Ticker")["Close"].transform(lambda x: x.expanding().max())
results["expand_max_mean"] = float(expand_max.mean())
for k in ["sma20_non_nan_count", "sma20_mean", "expand_max_mean"]:
    print(f"  {k}: {results[k]}")

# 5. Drop duplicates
print("\n5. ETL")
deduped = all_stocks.drop_duplicates(subset=["Ticker", "Date"])
results["dedup_count"] = int(len(deduped))
clipped = all_stocks["Close"].clip(lower=0, upper=10000)
results["clip_mean"] = float(clipped.mean())
for k in ["dedup_count", "clip_mean"]:
    print(f"  {k}: {results[k]}")

# 6. Correlation
print("\n6. Correlation")
sample = all_stocks[["Open", "High", "Low", "Close", "Volume"]].head(1_000_000)
corr = sample.corr()
results["corr_open_close"] = float(corr.loc["Open", "Close"])
results["corr_open_volume"] = float(corr.loc["Open", "Volume"])
results["corr_close_volume"] = float(corr.loc["Close", "Volume"])
for k in ["corr_open_close", "corr_open_volume", "corr_close_volume"]:
    print(f"  {k}: {results[k]:.6f}")

# 7. Star join
print("\n7. Star Schema Joins")
sectors = ["Tech","Finance","Health","Energy","Consumer","Industrial","Utility","Materials"]
dim_sector = pd.DataFrame({"TickerId": range(n_tickers), "Sector": [sectors[i%len(sectors)] for i in range(n_tickers)]})
joined = all_stocks.merge(dim_sector, on="TickerId")
results["join_row_count"] = int(len(joined))
results["join_sector_counts"] = joined["Sector"].value_counts().sort_index().to_dict()
for k in ["join_row_count"]:
    print(f"  {k}: {results[k]}")
print(f"  join_sector_counts: { {k: int(v) for k,v in results['join_sector_counts'].items()} }")

# 8. Reshape
print("\n8. Reshape")
reshape_data = all_stocks[all_stocks["Ticker"].isin(tickers[:20])].head(5000)
pivoted = reshape_data.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="mean")
results["pivot_shape"] = list(pivoted.shape)
melted = pivoted.reset_index().melt(id_vars=["Date"])
results["melt_row_count"] = int(len(melted))
for k in ["pivot_shape", "melt_row_count"]:
    print(f"  {k}: {results[k]}")

# Save
with open("stock_output_python/validation_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\n{'='*60}")
print("Saved to stock_output_python/validation_results.json")
