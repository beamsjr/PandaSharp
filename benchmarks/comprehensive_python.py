"""
Comprehensive PandaSharp vs pandas Benchmark — All Categories
=============================================================
"""
import time, os, glob, json
import numpy as np
import pandas as pd

DATA_DIR = "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks"
os.makedirs("stock_output_python", exist_ok=True)

results = []
def lap(cat, name, start, detail=""):
    ms = int((time.time() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms, "detail": detail})
    print(f"  {name:<55} {ms:>6,} ms")
    return ms

print("=== Comprehensive Python Benchmark ===\n")

# Load data
t = time.time()
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
lap("Setup", "Load all CSVs", t, f"{len(frames)} stocks, {len(all_stocks):,} rows")

# ════════════════════════════════════════════════════════
# 1. STAR SCHEMA JOINS
# ════════════════════════════════════════════════════════
print("\n── Star Schema Joins ──")
# Create dimension tables with int keys
tickers = all_stocks["Ticker"].unique()
n_tickers = len(tickers)
ticker_to_id = {t: i for i, t in enumerate(tickers)}
all_stocks["TickerId"] = all_stocks["Ticker"].map(ticker_to_id)

sectors = ["Tech","Finance","Health","Energy","Consumer","Industrial","Utility","Materials"]
dim_sector = pd.DataFrame({"TickerId": range(n_tickers), "Sector": [sectors[i%len(sectors)] for i in range(n_tickers)], "SectorId": [i%len(sectors) for i in range(n_tickers)]})
dim_exchange = pd.DataFrame({"TickerId": range(n_tickers), "Exchange": ["NYSE" if i%2==0 else "NASDAQ" for i in range(n_tickers)], "ListYear": [2000 + i%20 for i in range(n_tickers)]})
dim_fundamentals = pd.DataFrame({"TickerId": range(n_tickers), "MarketCap": np.random.default_rng(42).random(n_tickers)*1e9, "PE": np.random.default_rng(42).random(n_tickers)*50})
dim_country = pd.DataFrame({"TickerId": range(n_tickers), "Country": ["US" if i%3!=2 else "Intl" for i in range(n_tickers)]})

t = time.time()
joined = all_stocks.merge(dim_sector, on="TickerId")
lap("StarJoin", "Join fact × sector (int key)", t, f"{len(joined):,} rows")

t = time.time()
joined = joined.merge(dim_exchange, on="TickerId")
lap("StarJoin", "Join × exchange (int key)", t, f"{len(joined):,} rows")

t = time.time()
joined = joined.merge(dim_fundamentals, on="TickerId")
lap("StarJoin", "Join × fundamentals (int key)", t, f"{len(joined):,} rows")

t = time.time()
joined = joined.merge(dim_country, on="TickerId")
lap("StarJoin", "Join × country (int key)", t, f"{len(joined):,} rows")

# ════════════════════════════════════════════════════════
# 2. STRING PROCESSING
# ════════════════════════════════════════════════════════
print("\n── String Processing ──")
ticker_col = all_stocks["Ticker"]

t = time.time(); _ = ticker_col.str.upper(); lap("String", "str.upper()", t)
t = time.time(); _ = ticker_col.str.lower(); lap("String", "str.lower()", t)
t = time.time(); _ = ticker_col.str.contains("A"); lap("String", "str.contains('A')", t)
t = time.time(); _ = ticker_col.str.len(); lap("String", "str.len()", t)
t = time.time(); _ = ticker_col.str.startswith("AA"); lap("String", "str.startswith('AA')", t)
t = time.time(); _ = ticker_col.str.replace("A", "X"); lap("String", "str.replace('A','X')", t)
t = time.time(); _ = ticker_col.str.slice(0, 2); lap("String", "str.slice(0,2)", t)

# ════════════════════════════════════════════════════════
# 3. RESHAPE OPERATIONS
# ════════════════════════════════════════════════════════
print("\n── Reshape Operations ──")
# Use a subset for reshape (too large otherwise)
reshape_data = all_stocks[all_stocks["Ticker"].isin(tickers[:20])].head(5000)

t = time.time()
pivoted = reshape_data.pivot_table(index="Date", columns="Ticker", values="Close", aggfunc="mean")
lap("Reshape", "Pivot table (20 tickers)", t, f"{pivoted.shape}")

t = time.time()
melted = pivoted.reset_index().melt(id_vars=["Date"], var_name="Ticker", value_name="Close")
lap("Reshape", "Melt", t, f"{len(melted):,} rows")

t = time.time()
dummies = pd.get_dummies(reshape_data["Ticker"])
lap("Reshape", "get_dummies (20 tickers)", t, f"{dummies.shape}")

t = time.time()
ct = pd.crosstab(reshape_data["Ticker"], reshape_data["Date"].str[:7])
lap("Reshape", "Crosstab (ticker × month)", t, f"{ct.shape}")

# ════════════════════════════════════════════════════════
# 4. WINDOW AT SCALE
# ════════════════════════════════════════════════════════
print("\n── Window Functions at Scale ──")

t = time.time()
all_stocks["SMA20"] = all_stocks.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
lap("Window", "GroupBy Rolling SMA20 (all stocks)", t, f"{len(all_stocks):,} rows")

t = time.time()
all_stocks["EWM20"] = all_stocks.groupby("Ticker")["Close"].transform(lambda x: x.ewm(span=20).mean())
lap("Window", "GroupBy EWM span=20 (all stocks)", t, f"{len(all_stocks):,} rows")

t = time.time()
all_stocks["ExpandMax"] = all_stocks.groupby("Ticker")["Close"].transform(lambda x: x.expanding().max())
lap("Window", "GroupBy Expanding Max (all stocks)", t, f"{len(all_stocks):,} rows")

# ════════════════════════════════════════════════════════
# 5. DATA CLEANING / ETL
# ════════════════════════════════════════════════════════
print("\n── Data Cleaning / ETL ──")

t = time.time()
has_null = all_stocks.isnull().any()
null_counts = all_stocks.isnull().sum()
lap("ETL", "Null detection (all columns)", t)

t = time.time()
filled = all_stocks["Close"].ffill()
lap("ETL", "Forward fill (Close)", t)

t = time.time()
deduped = all_stocks.drop_duplicates(subset=["Ticker", "Date"])
lap("ETL", "Drop duplicates (Ticker+Date)", t, f"{len(deduped):,} unique")

t = time.time()
casted = all_stocks["Volume"].astype(float)
lap("ETL", "Type cast Volume→float", t)

t = time.time()
clipped = all_stocks["Close"].clip(lower=0, upper=10000)
lap("ETL", "Clip values [0, 10000]", t)

# ════════════════════════════════════════════════════════
# 6. MULTI-AGGREGATION
# ════════════════════════════════════════════════════════
print("\n── Multi-Aggregation ──")

t = time.time()
multi_agg = all_stocks.groupby("Ticker").agg(
    MeanClose=("Close", "mean"), StdClose=("Close", "std"),
    MinClose=("Close", "min"), MaxClose=("Close", "max"),
    MeanVol=("Volume", "mean"), TotalVol=("Volume", "sum"),
    Count=("Close", "count"))
lap("MultiAgg", "7-function named agg", t, f"{len(multi_agg)} groups")

t = time.time()
sector_agg = joined.groupby(["Sector","Exchange"]).agg(
    AvgClose=("Close","mean"), AvgVol=("Volume","mean"), Count=("Close","count"))
lap("MultiAgg", "Multi-key GroupBy (Sector×Exchange)", t)

# ════════════════════════════════════════════════════════
# 7. EXPRESSION CHAINS
# ════════════════════════════════════════════════════════
print("\n── Expression Chains ──")

t = time.time()
all_stocks["DailyReturn"] = (all_stocks["Close"] - all_stocks["Open"]) / all_stocks["Open"] * 100
lap("Expr", "DailyReturn = (Close-Open)/Open*100", t)

t = time.time()
all_stocks["Spread"] = (all_stocks["High"] - all_stocks["Low"]) / all_stocks["Close"] * 100
lap("Expr", "Spread = (High-Low)/Close*100", t)

t = time.time()
mask = (all_stocks["DailyReturn"] > 2) & (all_stocks["Volume"] > 1e6) & (all_stocks["Close"] > 5)
filtered = all_stocks[mask]
lap("Expr", "Complex filter: ret>2 & vol>1M & close>5", t, f"{len(filtered):,} matches")

t = time.time()
all_stocks["Score"] = all_stocks["DailyReturn"] * 0.5 + np.log1p(all_stocks["Volume"]) * 0.3 + all_stocks["Close"] * 0.2
lap("Expr", "Weighted score computation", t)

# ════════════════════════════════════════════════════════
# 8. CORRELATION MATRIX
# ════════════════════════════════════════════════════════
print("\n── Correlation Matrix ──")

numeric_cols = ["Open", "High", "Low", "Close", "Volume"]
sample = all_stocks[numeric_cols].head(1_000_000)

t = time.time()
corr = sample.corr()
lap("Corr", "5×5 correlation (1M rows)", t)

# Wide correlation: one column per ticker
t = time.time()
wide = all_stocks.pivot_table(index="Date", columns="Ticker", values="Close").head(1000)
wide_corr = wide.corr()
lap("Corr", f"Wide corr ({wide.shape[1]} tickers × 1K days)", t, f"{wide_corr.shape}")

# ════════════════════════════════════════════════════════
# 9. SORT
# ════════════════════════════════════════════════════════
print("\n── Sort Operations ──")

t = time.time()
sorted_df = all_stocks.sort_values("Close")
lap("Sort", "Sort by Close (full dataset)", t, f"{len(sorted_df):,} rows")

t = time.time()
sorted_multi = all_stocks.sort_values(["Ticker", "Close"], ascending=[True, False])
lap("Sort", "Multi-key sort (Ticker asc, Close desc)", t)

t = time.time()
top10 = all_stocks.nlargest(10, "Close")
lap("Sort", "Nlargest(10, Close)", t)

t = time.time()
bot10 = all_stocks.nsmallest(10, "Close")
lap("Sort", "Nsmallest(10, Close)", t)

# ════════════════════════════════════════════════════════
# 10. COLUMN ARITHMETIC
# ════════════════════════════════════════════════════════
print("\n── Column Arithmetic ──")

t = time.time()
turnover = all_stocks["Close"] * all_stocks["Volume"]
lap("Arithmetic", "Close * Volume (element-wise)", t)

t = time.time()
spread_pts = all_stocks["High"] - all_stocks["Low"]
lap("Arithmetic", "High - Low (element-wise)", t)

t = time.time()
mid = (all_stocks["High"] + all_stocks["Low"]) / 2
lap("Arithmetic", "(High + Low) / 2 (midprice)", t)

t = time.time()
pct = all_stocks["Close"].pct_change()
lap("Arithmetic", "Close.pct_change()", t)

# ════════════════════════════════════════════════════════
# 11. CUMULATIVE OPS
# ════════════════════════════════════════════════════════
print("\n── Cumulative Operations ──")

t = time.time()
cs = all_stocks["Close"].cumsum()
lap("Cumulative", "Close.cumsum()", t)

t = time.time()
cm = all_stocks["Close"].cummax()
lap("Cumulative", "Close.cummax()", t)

t = time.time()
cmin = all_stocks["Close"].cummin()
lap("Cumulative", "Close.cummin()", t)

# ════════════════════════════════════════════════════════
# 12. RANK
# ════════════════════════════════════════════════════════
print("\n── Rank ──")

t = time.time()
ranked = all_stocks["Close"].rank()
lap("Rank", "Close.rank() (14.7M rows)", t)

t = time.time()
ranked_pct = all_stocks["Close"].rank(pct=True)
lap("Rank", "Close.rank(pct=True)", t)

# ════════════════════════════════════════════════════════
# 13. VALUE COUNTS
# ════════════════════════════════════════════════════════
print("\n── Value Counts ──")

t = time.time()
vc = all_stocks["Ticker"].value_counts()
lap("ValueCounts", "Ticker.value_counts()", t)

t = time.time()
nu = all_stocks["Ticker"].nunique()
lap("ValueCounts", "Ticker.nunique()", t)

# ════════════════════════════════════════════════════════
# 14. DESCRIBE
# ════════════════════════════════════════════════════════
print("\n── Describe ──")

t = time.time()
desc = all_stocks[["Open","High","Low","Close","Volume"]].describe()
lap("Describe", "Describe (5 numeric columns)", t)

# ════════════════════════════════════════════════════════
# 15. FILLNA / INTERPOLATE
# ════════════════════════════════════════════════════════
print("\n── FillNa / Interpolate ──")

# Create a column with some NaN
close_with_nan = all_stocks["Close"].copy()
close_with_nan.iloc[::10] = np.nan  # every 10th value

t = time.time()
filled_ffill = close_with_nan.ffill()
lap("FillNa", "Forward fill (10% NaN)", t)

t = time.time()
filled_val = close_with_nan.fillna(0)
lap("FillNa", "FillNa(0)", t)

t = time.time()
interp = close_with_nan.interpolate()
lap("FillNa", "Interpolate (linear)", t)

# ════════════════════════════════════════════════════════
# 16. GROUPBY TRANSFORM
# ════════════════════════════════════════════════════════
print("\n── GroupBy Transform ──")

t = time.time()
zscore = all_stocks.groupby("Ticker")["Close"].transform(lambda x: (x - x.mean()) / x.std())
lap("GBTransform", "GroupBy z-score normalize", t)

t = time.time()
grp_rank = all_stocks.groupby("Ticker")["Close"].rank()
lap("GBTransform", "GroupBy per-group rank", t)

# ════════════════════════════════════════════════════════
# 17. SHIFT / LAG
# ════════════════════════════════════════════════════════
print("\n── Shift / Lag ──")

t = time.time()
shifted = all_stocks.groupby("Ticker")["Close"].shift(1)
lap("Shift", "GroupBy shift(1) (lag)", t)

t = time.time()
diff = all_stocks.groupby("Ticker")["Close"].diff()
lap("Shift", "GroupBy diff() (daily change)", t)

# ════════════════════════════════════════════════════════
# 18. MERGE (STRING KEY)
# ════════════════════════════════════════════════════════
print("\n── Merge (String Key) ──")

# Build a small lookup table
ticker_info = pd.DataFrame({
    "Ticker": tickers,
    "Industry": [sectors[i % len(sectors)] for i in range(n_tickers)],
    "Founded": [1950 + i % 70 for i in range(n_tickers)]
})

t = time.time()
merged_str = all_stocks.merge(ticker_info, on="Ticker")
lap("MergeStr", "Merge on Ticker (string key, 14.7M rows)", t, f"{len(merged_str):,} rows")

# ════════════════════════════════════════════════════════
# 19. SAMPLE / QUANTILE
# ════════════════════════════════════════════════════════
print("\n── Sample / Quantile ──")

t = time.time()
sampled = all_stocks.sample(1_000_000, random_state=42)
lap("SampleQ", "Sample 1M rows", t)

t = time.time()
quants = all_stocks["Close"].quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
lap("SampleQ", "Close.quantile([0.01..0.99])", t)

# ════════════════════════════════════════════════════════
# SUMMARY
# ════════════════════════════════════════════════════════
print(f"\n{'═'*70}")
categories = {}
for r in results:
    cat = r["category"]
    categories[cat] = categories.get(cat, 0) + r["ms"]

for cat, ms in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat:<30} {ms:>8,} ms")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
print(f"{'═'*70}")

with open("stock_output_python/comprehensive_results.json", "w") as f:
    json.dump(results, f, indent=2)
