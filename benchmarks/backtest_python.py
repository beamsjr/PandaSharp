"""
Momentum Strategy Backtest — Python (pandas) version
=====================================================
Identical logic to the PandaSharp version for fair comparison.
"""

import time, os, glob, json
import numpy as np
import pandas as pd

DATA_DIR = "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks"

results = []
total_start = time.time()
def lap(name, start, detail=""):
    ms = int((time.time() - start) * 1000)
    results.append({"op": name, "ms": ms, "detail": detail})
    print(f"  {name:<50} {ms:>8,} ms  {detail}")
    return ms

print("=== Momentum Backtest — Python (pandas) ===\n")

# ── 1. Load all stocks ─────────────────────────────────
t = time.time()
files = glob.glob(os.path.join(DATA_DIR, "*.txt"))
frames = []
for f in files:
    try:
        df = pd.read_csv(f)
        if len(df) < 252: continue  # need 1 year minimum
        ticker = os.path.basename(f).replace(".us.txt", "").upper()
        df["Ticker"] = ticker
        frames.append(df)
    except: pass
all_stocks = pd.concat(frames, ignore_index=True)
lap("Load CSVs", t, f"{len(frames)} stocks, {len(all_stocks):,} rows")

# ── 2. Create sector mapping (simulated) and JOIN ──────
t = time.time()
unique_tickers = all_stocks["Ticker"].unique()
sectors = ["Tech", "Finance", "Health", "Energy", "Consumer"]
sector_map = pd.DataFrame({
    "Ticker": unique_tickers,
    "SectorId": np.arange(len(unique_tickers)),
    "Sector": [sectors[i % len(sectors)] for i in range(len(unique_tickers))],
    "Weight": np.random.default_rng(42).random(len(unique_tickers))
})
# Join on Ticker (string key)
joined = all_stocks.merge(sector_map, on="Ticker", how="inner")
lap("Join with sector data", t, f"{len(joined):,} rows after join")

# ── 3. Filter: only stocks with avg volume > threshold ─
t = time.time()
avg_volume = joined.groupby("Ticker")["Volume"].mean()
liquid_tickers = avg_volume[avg_volume > 100_000].index
liquid = joined[joined["Ticker"].isin(liquid_tickers)]
lap("Filter by avg volume", t, f"{len(liquid_tickers)} liquid stocks, {len(liquid):,} rows")

# ── 4. Compute rolling signals PER STOCK ───────────────
t = time.time()
liquid = liquid.sort_values(["Ticker", "Date"])
liquid["SMA20"] = liquid.groupby("Ticker")["Close"].transform(lambda x: x.rolling(20).mean())
liquid["SMA50"] = liquid.groupby("Ticker")["Close"].transform(lambda x: x.rolling(50).mean())
liquid["Momentum"] = liquid.groupby("Ticker")["Close"].transform(lambda x: x.pct_change(20) * 100)
liquid = liquid.dropna(subset=["SMA20", "SMA50", "Momentum"])
lap("Rolling signals (SMA20/50, Momentum)", t, f"{len(liquid):,} rows with signals")

# ── 5. Generate buy signals: SMA20 crosses above SMA50 ─
t = time.time()
liquid["Signal"] = (liquid["SMA20"] > liquid["SMA50"]).astype(int)
liquid["PrevSignal"] = liquid.groupby("Ticker")["Signal"].shift(1).fillna(0).astype(int)
crossovers = liquid[(liquid["Signal"] == 1) & (liquid["PrevSignal"] == 0)]
lap("Detect SMA crossover signals", t, f"{len(crossovers):,} crossover events")

# ── 6. For each signal date, rank by momentum, pick top 20
t = time.time()
top_picks_list = []
for date, group in crossovers.groupby("Date"):
    top = group.nlargest(20, "Momentum")[["Date", "Ticker", "Close", "Momentum", "Sector", "Volume"]]
    top_picks_list.append(top)
if top_picks_list:
    top_picks = pd.concat(top_picks_list, ignore_index=True)
else:
    top_picks = pd.DataFrame()
lap("Rank & pick top 20 per day", t, f"{len(top_picks):,} total picks")

# ── 7. Compute forward returns (next 5 days) ──────────
t = time.time()
liquid["Fwd5d"] = liquid.groupby("Ticker")["Close"].transform(lambda x: x.shift(-5) / x - 1) * 100
if len(top_picks) > 0:
    top_picks = top_picks.merge(
        liquid[["Ticker", "Date", "Fwd5d"]].drop_duplicates(["Ticker", "Date"]),
        on=["Ticker", "Date"], how="left"
    )
    avg_return = top_picks["Fwd5d"].mean()
else:
    avg_return = 0
lap("Forward return calculation", t, f"Avg 5-day return: {avg_return:.2f}%")

# ── 8. Head/Tail slicing (repeated access pattern) ────
t = time.time()
for ticker in liquid_tickers[:500]:
    subset = liquid[liquid["Ticker"] == ticker]
    _ = subset.head(10)
    _ = subset.tail(10)
lap("Head/Tail × 500 stocks", t, "Repeated slice access")

# ── 9. Drop duplicate signals ─────────────────────────
t = time.time()
deduped = crossovers.drop_duplicates(subset=["Ticker", "Date"])
lap("Drop duplicate signals", t, f"{len(deduped):,} unique signals")

# ── 10. Lambda-based custom scoring ───────────────────
t = time.time()
if len(top_picks) > 0:
    top_picks["Score"] = top_picks.apply(
        lambda row: row["Momentum"] * 0.6 + np.log1p(row["Volume"]) * 0.4
        if pd.notna(row["Momentum"]) else 0, axis=1)
    best = top_picks.nlargest(50, "Score")
lap("Lambda scoring + nlargest", t, f"Top 50 scored picks")

# ── 11. Sector aggregation ────────────────────────────
t = time.time()
if len(top_picks) > 0:
    sector_perf = top_picks.groupby("Sector").agg(
        AvgMomentum=("Momentum", "mean"),
        AvgReturn=("Fwd5d", "mean"),
        Count=("Ticker", "count")
    ).sort_values("AvgReturn", ascending=False)
lap("Sector aggregation", t, "GroupBy + multi-agg")

# ── 12. Multi-join: picks × sectors × returns ─────────
t = time.time()
if len(top_picks) > 0:
    enriched = top_picks.merge(sector_map[["Ticker", "Weight"]], on="Ticker", how="left")
    enriched = enriched.merge(
        liquid.groupby("Ticker")["Volume"].mean().reset_index().rename(columns={"Volume": "AvgVolume"}),
        on="Ticker", how="left"
    )
lap("Multi-join enrichment", t, f"{len(enriched) if len(top_picks) > 0 else 0:,} enriched rows")

# ── SUMMARY ────────────────────────────────────────────
total_ms = int((time.time() - total_start) * 1000)
print(f"\n{'─'*70}")
print(f"{'TOTAL':<50} {total_ms:>8,} ms")
print(f"{'─'*70}")

with open("stock_output_python/backtest_results.json", "w") as f:
    json.dump({"total_ms": total_ms, "ops": results}, f, indent=2)
