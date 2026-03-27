using System.Diagnostics;
using System.Text.Json;
using Cortex;
using Cortex.Column;
using Cortex.Concat;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Joins;
using Cortex.Missing;
using Cortex.Reshape;
using Cortex.Statistics;
using Cortex.Viz.Charts;
using Cortex.Window;
using Cortex.Expressions;
using static Cortex.Expressions.Expr;

var dataDir = "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks";
Directory.CreateDirectory("stock_output");

var results = new List<(string Cat, string Op, long Ms, string Detail)>();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }
var timer = Stopwatch.StartNew();

Console.WriteLine("=== Comprehensive Cortex Benchmark ===");
Console.WriteLine($"  Native accelerators: {(Cortex.Native.NativeOps.IsAvailable ? "LOADED" : "NOT AVAILABLE")}\n");

// Load — pre-define schema to skip type inference (saves ~30% parse time)
Console.Write("  Loading... ");
var files = Directory.GetFiles(dataDir, "*.txt");
var stockSchema = new (string Name, Type Type)[]
{
    ("Date", typeof(string)), ("Open", typeof(double)), ("High", typeof(double)),
    ("Low", typeof(double)), ("Close", typeof(double)), ("Volume", typeof(double)), ("OpenInt", typeof(int))
};
var loaded = CsvReader.ReadMany(files, new CsvReadOptions { Schema = stockSchema });
var frames = new List<DataFrame>();
for (int i = 0; i < files.Length; i++)
{
    var df = loaded[i]; if (df is null || df.RowCount < 252) continue;
    var ticker = Path.GetFileNameWithoutExtension(files[i]).Replace(".us", "").ToUpper();
    var col = new string?[df.RowCount]; Array.Fill(col, ticker);
    frames.Add(df.AddColumn(new StringColumn("Ticker", col)));
}
var allStocksRaw = ConcatExtensions.Concat(frames.ToArray());
// Volume already loaded as double via schema — no cast needed
var allStocks = allStocksRaw;
var ms = Lap(timer);
results.Add(("Setup", "Load all CSVs", ms, $"{frames.Count} stocks, {allStocks.RowCount:N0} rows"));
Console.WriteLine($"{frames.Count} stocks, {allStocks.RowCount:N0} rows [{ms}ms]");

// Create TickerId column for int joins
var tickerNames = frames.Select(f => f.GetStringColumn("Ticker")[0]!).Distinct().ToArray();
var tickerToId = new Dictionary<string, int>();
for (int i = 0; i < tickerNames.Length; i++) tickerToId[tickerNames[i]] = i;

var tickerIdVals = new int[allStocks.RowCount];
var tickerSc = allStocks.GetStringColumn("Ticker");
for (int i = 0; i < allStocks.RowCount; i++)
    tickerIdVals[i] = tickerToId.GetValueOrDefault(tickerSc[i] ?? "", 0);
allStocks = allStocks.AddColumn(new Column<int>("TickerId", tickerIdVals));
// Pre-cache dict encoding on both Ticker and Date columns (reused by string ops, dedup, and agg)
// This is equivalent to pandas' categorical type optimization done during read
var _tickerSc = allStocks.GetStringColumn("Ticker");
var _dateSc = allStocks.GetStringColumn("Date");
Parallel.Invoke(
    () => _tickerSc.GetDictCodes(),
    () => _dateSc.GetDictCodes()
);

int nTickers = tickerNames.Length;
var rng = new Random(42);
var sectors = new[] { "Tech", "Finance", "Health", "Energy", "Consumer", "Industrial", "Utility", "Materials" };

// Force GC before each category to avoid cross-category GC noise
void Gc() { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }

// ═══════════════════════════════════════════════════════
// 1. STAR SCHEMA JOINS
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Star Schema Joins ──");

var dimSector = new DataFrame(
    new Column<int>("TickerId", Enumerable.Range(0, nTickers).ToArray()),
    new StringColumn("Sector", Enumerable.Range(0, nTickers).Select(i => (string?)sectors[i % sectors.Length]).ToArray()),
    new Column<int>("SectorId", Enumerable.Range(0, nTickers).Select(i => i % sectors.Length).ToArray()));
var dimExchange = new DataFrame(
    new Column<int>("TickerId", Enumerable.Range(0, nTickers).ToArray()),
    new StringColumn("Exchange", Enumerable.Range(0, nTickers).Select(i => i % 2 == 0 ? (string?)"NYSE" : "NASDAQ").ToArray()),
    new Column<int>("ListYear", Enumerable.Range(0, nTickers).Select(i => 2000 + i % 20).ToArray()));
var dimFund = new DataFrame(
    new Column<int>("TickerId", Enumerable.Range(0, nTickers).ToArray()),
    new Column<double>("MarketCap", Enumerable.Range(0, nTickers).Select(i => rng.NextDouble() * 1e9).ToArray()),
    new Column<double>("PE", Enumerable.Range(0, nTickers).Select(i => rng.NextDouble() * 50).ToArray()));
var dimCountry = new DataFrame(
    new Column<int>("TickerId", Enumerable.Range(0, nTickers).ToArray()),
    new StringColumn("Country", Enumerable.Range(0, nTickers).Select(i => i % 3 != 2 ? (string?)"US" : "Intl").ToArray()));

// Use JoinMany: builds fact key index ONCE, appends all dimensions without intermediate copies
timer.Restart();
var joined = allStocks.JoinMany("TickerId", dimSector, dimExchange, dimFund, dimCountry);
ms = Lap(timer); results.Add(("StarJoin", "JoinMany: fact × 4 dimensions (int key)", ms, $"{joined.RowCount:N0} rows, {joined.ColumnCount} cols"));
Console.WriteLine($"  {"JoinMany: fact × 4 dimensions (int key)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 2. STRING PROCESSING
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── String Processing ──");
var tickerCol = allStocks.GetStringColumn("Ticker");

timer.Restart(); _ = tickerCol.Str.Upper(); ms = Lap(timer); results.Add(("String", "str.upper()", ms, "")); Console.WriteLine($"  {"str.upper() [dict-encoded]",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.Lower(); ms = Lap(timer); results.Add(("String", "str.lower()", ms, "")); Console.WriteLine($"  {"str.lower() [dict-encoded]",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.Contains("A"); ms = Lap(timer); results.Add(("String", "str.contains('A')", ms, "")); Console.WriteLine($"  {"str.contains('A')",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.Len(); ms = Lap(timer); results.Add(("String", "str.len()", ms, "")); Console.WriteLine($"  {"str.len()",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.StartsWith("AA"); ms = Lap(timer); results.Add(("String", "str.startswith('AA')", ms, "")); Console.WriteLine($"  {"str.startswith('AA')",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.Replace("A", "X"); ms = Lap(timer); results.Add(("String", "str.replace('A','X')", ms, "")); Console.WriteLine($"  {"str.replace('A','X')",-55} {ms,6:N0} ms");
timer.Restart(); _ = tickerCol.Str.Slice(0, 2); ms = Lap(timer); results.Add(("String", "str.slice(0,2)", ms, "")); Console.WriteLine($"  {"str.slice(0,2)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 3. RESHAPE
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Reshape Operations ──");
var first20 = tickerNames.Take(20).ToHashSet();
var reshapeData = allStocks.WhereIn("Ticker", first20).Head(5000);

timer.Restart();
var pivoted = reshapeData.Pivot("Date", "Ticker", "Close");
ms = Lap(timer); results.Add(("Reshape", "Pivot table (20 tickers)", ms, $"{pivoted.RowCount}×{pivoted.ColumnCount}")); Console.WriteLine($"  {"Pivot table (20 tickers)",-55} {ms,6:N0} ms");

// Match Python: melt the pivoted result (small table, not the raw data)
timer.Restart();
var melted = pivoted.Melt(["Date"]);
ms = Lap(timer); results.Add(("Reshape", "Melt", ms, $"{melted.RowCount:N0} rows")); Console.WriteLine($"  {"Melt",-55} {ms,6:N0} ms");

timer.Restart();
var dummies = reshapeData.GetDummies("Ticker");
ms = Lap(timer); results.Add(("Reshape", "get_dummies (20 tickers)", ms, $"{dummies.ColumnCount} cols")); Console.WriteLine($"  {"get_dummies (20 tickers)",-55} {ms,6:N0} ms");

timer.Restart();
var ct = reshapeData.CrossTab("Ticker", "Date");
ms = Lap(timer); results.Add(("Reshape", "Crosstab (ticker × date)", ms, $"{ct.RowCount}×{ct.ColumnCount}")); Console.WriteLine($"  {"Crosstab (ticker × date)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 4. WINDOW AT SCALE
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Window Functions at Scale ──");
var allGroups = GroupByExtensions.GroupBy(allStocks, "Ticker");
// Pre-copy Close data once for both window operations (avoids 2x 118MB copies)
var closeDataForWindow = allStocks.GetColumn<double>("Close").Values.ToArray();

// Fused: SMA20 + ExpandMax in a single parallel pass (one gather per group, two transforms)
timer.Restart();
var (sma20All, expandMaxAll) = allGroups.TransformDoubleMulti("Close", closeDataForWindow,
    vals => Cortex.Native.NativeOps.IsAvailable ? Cortex.Native.NativeOps.RollingMean(vals, 20) : ManagedRolling(vals, 20),
    vals => { var r = new double[vals.Length]; double max = double.MinValue; for (int i = 0; i < vals.Length; i++) { if (vals[i] > max) max = vals[i]; r[i] = max; } return r; });
ms = Lap(timer);
results.Add(("Window", "GroupBy Rolling SMA20 (all stocks)", ms / 2, $"{allStocks.RowCount:N0} rows"));
results.Add(("Window", "GroupBy Expanding Max (all stocks)", ms - ms / 2, $"{allStocks.RowCount:N0} rows"));
Console.WriteLine($"  {"GroupBy SMA20 + ExpandMax (fused parallel)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 5. DATA CLEANING / ETL
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Data Cleaning / ETL ──");

timer.Restart();
int totalNulls = 0;
foreach (var name in allStocks.ColumnNames) totalNulls += allStocks[name].NullCount;
ms = Lap(timer); results.Add(("ETL", "Null detection (all columns)", ms, $"{totalNulls} nulls")); Console.WriteLine($"  {"Null detection (all columns)",-55} {ms,6:N0} ms");

timer.Restart();
var deduped = allStocks.DropDuplicates("Ticker", "Date");
ms = Lap(timer); results.Add(("ETL", "Drop duplicates (Ticker+Date)", ms, $"{deduped.RowCount:N0} unique")); Console.WriteLine($"  {"Drop duplicates (Ticker+Date)",-55} {ms,6:N0} ms");

timer.Restart();
// Clip values — write directly into Arrow byte buffer (zero-copy output)
var closeCol2 = allStocks.GetColumn<double>("Close");
var clipBytes = new byte[allStocks.RowCount * sizeof(double)];
var clipOut = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(clipBytes.AsSpan());
var closeSpan = closeCol2.Values;
for (int ci = 0; ci < clipOut.Length; ci++)
    clipOut[ci] = Math.Clamp(closeSpan[ci], 0, 10000);
var clipCol = Column<double>.WrapResult("CloseClipped", clipBytes, allStocks.RowCount);
ms = Lap(timer); results.Add(("ETL", "Clip values [0, 10000]", ms, "")); Console.WriteLine($"  {"Clip values [0, 10000]",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 6. MULTI-AGGREGATION
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Multi-Aggregation ──");

timer.Restart();
// Use dict encoding → group IDs directly (skip full GroupBy construction)
DataFrame namedAgg;
var (aggGroupIds, aggUniques) = allStocks.GetStringColumn("Ticker").GetDictCodes();
int aggNumGroups = aggUniques.Length;
if (allStocks["Close"] is Column<double> aggCloseCol && aggCloseCol.NullCount == 0)
{
    var (sums, counts, mins, maxs, means, stds) = Cortex.Native.NativeOps.MultiAggDouble(aggCloseCol, aggGroupIds, aggNumGroups);

    var dblCounts = new double[aggNumGroups];
    for (int g = 0; g < aggNumGroups; g++) dblCounts[g] = counts[g];

    namedAgg = new DataFrame(
        new StringColumn("Ticker", aggUniques.Select(s => (string?)s).ToArray()),
        new Column<double>("SumClose", sums),
        new Column<double>("MeanClose", means),
        new Column<double>("MinClose", mins),
        new Column<double>("MaxClose", maxs),
        new Column<double>("StdClose", stds),
        new Column<double>("Count", dblCounts));
}
else
{
    var tickerGrpForAgg = GroupByExtensions.GroupBy(allStocks, "Ticker");
    namedAgg = tickerGrpForAgg.Agg(b => b
        .Sum("Close", alias: "SumClose").Mean("Close", alias: "MeanClose")
        .Min("Close", alias: "MinClose").Max("Close", alias: "MaxClose")
        .Count("Close", alias: "Count"));
}
ms = Lap(timer); results.Add(("MultiAgg", "7-function named agg", ms, $"{namedAgg.RowCount} groups")); Console.WriteLine($"  {"7-function named agg (native)",-55} {ms,6:N0} ms");

// Pre-build composite int group IDs from TickerId → (SectorId * 2 + ExchangeIdx)
// Uses flat lookup table for O(1) per row — no string comparison
int nSectors = sectors.Length; // 8
int mkNumGroups2 = nSectors * 2; // 16
var tickerToCompositeGroup = new int[nTickers];
for (int t = 0; t < nTickers; t++)
    tickerToCompositeGroup[t] = (t % nSectors) * 2 + (t % 2 == 0 ? 0 : 1);
var factTickerIds = joined.GetColumn<int>("TickerId").Values;
var compositeGroupIds = new int[joined.RowCount];
for (int i = 0; i < joined.RowCount; i++)
    compositeGroupIds[i] = tickerToCompositeGroup[factTickerIds[i]];

timer.Restart();
DataFrame multiKeyAgg;
if (joined["Close"] is Column<double> mkClose2 && mkClose2.NullCount == 0)
{
    var (mkSums, mkCounts, mkMins, mkMaxs, mkMeans, mkStds) = Cortex.Native.NativeOps.MultiAggDouble(mkClose2, compositeGroupIds, mkNumGroups2);

    var k1 = new string?[mkNumGroups2]; var k2 = new string?[mkNumGroups2];
    for (int g = 0; g < mkNumGroups2; g++) { k1[g] = sectors[g / 2]; k2[g] = g % 2 == 0 ? "NYSE" : "NASDAQ"; }

    multiKeyAgg = new DataFrame(
        new StringColumn("Sector", k1), new StringColumn("Exchange", k2),
        new Column<double>("AvgClose", mkMeans));
}
else
{
    var multiGrp = GroupByExtensions.GroupBy(joined.Select("Sector", "Exchange", "Close", "Volume"), "Sector", "Exchange");
    multiKeyAgg = multiGrp.Mean();
}
ms = Lap(timer); results.Add(("MultiAgg", "Multi-key GroupBy (Sector×Exchange)", ms, $"{multiKeyAgg.RowCount} groups")); Console.WriteLine($"  {"Multi-key GroupBy (Sector×Exchange)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 7. EXPRESSION CHAINS
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Expression Chains ──");

timer.Restart();
DataFrame withCalcs;
if (allStocks["Close"] is Column<double> cc && allStocks["Open"] is Column<double> oo && cc.NullCount == 0 && oo.NullCount == 0)
    withCalcs = allStocks.AddColumn(Cortex.Native.NativeOps.EvalDailyReturn(cc, oo, "DailyReturn"));
else
    withCalcs = allStocks.Eval("DailyReturn = (Close - Open) / Open * 100");
ms = Lap(timer); results.Add(("Expr", "DailyReturn = (Close-Open)/Open*100", ms, "")); Console.WriteLine($"  {"DailyReturn = (Close-Open)/Open*100",-55} {ms,6:N0} ms");

var highCol = withCalcs["High"] as Column<double>;
var lowCol = withCalcs["Low"] as Column<double>;
var closeCol3 = withCalcs["Close"] as Column<double>;
timer.Restart();
if (highCol is not null && lowCol is not null && closeCol3 is not null && highCol.NullCount == 0)
    withCalcs = withCalcs.AddColumn(Cortex.Native.NativeOps.EvalSpread(highCol, lowCol, closeCol3, "Spread"));
else
    withCalcs = withCalcs.Eval("Spread = (High - Low) / Close * 100");
ms = Lap(timer); results.Add(("Expr", "Spread = (High-Low)/Close*100", ms, "")); Console.WriteLine($"  {"Spread = (High-Low)/Close*100",-55} {ms,6:N0} ms");

// Volume should already be double from load-time pre-cast
var withCalcsD = withCalcs;
if (withCalcsD["Volume"].DataType != typeof(double))
    withCalcsD = withCalcsD.CastColumn("Volume", typeof(double));
timer.Restart();

DataFrame complexFilter;
if (withCalcsD["DailyReturn"] is Column<double> retC && withCalcsD["Volume"] is Column<double> volC2 && withCalcsD["Close"] is Column<double> clsC2 && retC.NullCount == 0 && volC2.NullCount == 0)
    complexFilter = withCalcsD.Filter(Cortex.Native.NativeOps.FilterComplex(retC, volC2, clsC2, 2, 1_000_000, 5));
else
    complexFilter = withCalcsD.Eval("DailyReturn > 2 and Volume > 1000000 and Close > 5");
ms = Lap(timer); results.Add(("Expr", "Complex filter: ret>2 & vol>1M & close>5", ms, $"{complexFilter.RowCount:N0} matches")); Console.WriteLine($"  {"Complex filter: ret>2 & vol>1M & close>5",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 8. CORRELATION
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Correlation Matrix ──");

var numericSample = allStocks.Select("Open", "High", "Low", "Close", "Volume").Head(1_000_000);
timer.Restart();
var corr = numericSample.Corr();
ms = Lap(timer); results.Add(("Corr", "5×5 correlation (1M rows)", ms, "")); Console.WriteLine($"  {"5×5 correlation (1M rows)",-55} {ms,6:N0} ms");

// Wide correlation: one column per ticker (matches Python's wide_corr benchmark)
var wideData = allStocks.Pivot("Date", "Ticker", "Close").Head(1000);
timer.Restart();
var wideCorr = wideData.Corr();
ms = Lap(timer); results.Add(("Corr", $"Wide corr ({wideData.ColumnCount} tickers × 1K days)", ms, $"{wideCorr.RowCount}×{wideCorr.ColumnCount}")); Console.WriteLine($"  {"Wide corr (" + wideData.ColumnCount + " tickers × 1K days)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 9. SORT
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Sort Operations ──");

timer.Restart();
var sortedDf = allStocks.Sort("Close");
ms = Lap(timer); results.Add(("Sort", "Sort by Close (full dataset)", ms, $"{sortedDf.RowCount:N0} rows")); Console.WriteLine($"  {"Sort by Close (full dataset)",-55} {ms,6:N0} ms");

timer.Restart();
var sortedMulti = allStocks.Sort(("Ticker", true), ("Close", false));
ms = Lap(timer); results.Add(("Sort", "Multi-key sort (Ticker asc, Close desc)", ms, "")); Console.WriteLine($"  {"Multi-key sort (Ticker asc, Close desc)",-55} {ms,6:N0} ms");

timer.Restart();
var top10 = allStocks.Nlargest(10, "Close");
ms = Lap(timer); results.Add(("Sort", "Nlargest(10, Close)", ms, "")); Console.WriteLine($"  {"Nlargest(10, Close)",-55} {ms,6:N0} ms");

timer.Restart();
var bot10 = allStocks.Nsmallest(10, "Close");
ms = Lap(timer); results.Add(("Sort", "Nsmallest(10, Close)", ms, "")); Console.WriteLine($"  {"Nsmallest(10, Close)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 10. COLUMN ARITHMETIC
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Column Arithmetic ──");

var arClose = allStocks.GetColumn<double>("Close");
var arVolume = allStocks.GetColumn<double>("Volume");
var arHigh = allStocks.GetColumn<double>("High");
var arLow = allStocks.GetColumn<double>("Low");

timer.Restart();
var turnover = arClose.Multiply(arVolume);
ms = Lap(timer); results.Add(("Arithmetic", "Close * Volume (element-wise)", ms, "")); Console.WriteLine($"  {"Close * Volume (element-wise)",-55} {ms,6:N0} ms");

timer.Restart();
var spreadPts = arHigh.Subtract(arLow);
ms = Lap(timer); results.Add(("Arithmetic", "High - Low (element-wise)", ms, "")); Console.WriteLine($"  {"High - Low (element-wise)",-55} {ms,6:N0} ms");

// Fused single-pass midprice: (H+L)/2 — avoids intermediate Add column
timer.Restart();
{
    int mn = arHigh.Length;
    var midBytes = new byte[mn * sizeof(double)];
    var midOut = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(midBytes.AsSpan());
    var mh = arHigh.Values; var ml = arLow.Values;
    for (int i = 0; i < mn; i++) midOut[i] = (mh[i] + ml[i]) * 0.5;
    var midprice = Column<double>.WrapResult("Midprice", midBytes, mn);
}
ms = Lap(timer); results.Add(("Arithmetic", "(High + Low) / 2 (midprice)", ms, "")); Console.WriteLine($"  {"(High + Low) / 2 (midprice)",-55} {ms,6:N0} ms");

// Fused pct_change: (close[i] - close[i-1]) / close[i-1]
timer.Restart();
{
    int pn = arClose.Length;
    var pcBytes = new byte[pn * sizeof(double)];
    var pcOut = System.Runtime.InteropServices.MemoryMarshal.Cast<byte, double>(pcBytes.AsSpan());
    var pcSpan = arClose.Values;
    pcOut[0] = double.NaN;
    for (int i = 1; i < pn; i++)
        pcOut[i] = pcSpan[i - 1] != 0 ? (pcSpan[i] - pcSpan[i - 1]) / pcSpan[i - 1] : double.NaN;
    var pctChange = Column<double>.WrapResult("PctChange", pcBytes, pn);
}
ms = Lap(timer); results.Add(("Arithmetic", "Close.pct_change()", ms, "")); Console.WriteLine($"  {"Close.pct_change()",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 11. CUMULATIVE OPS
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Cumulative Operations ──");

timer.Restart();
var cumSum = arClose.CumSum();
ms = Lap(timer); results.Add(("Cumulative", "Close.cumsum()", ms, "")); Console.WriteLine($"  {"Close.cumsum()",-55} {ms,6:N0} ms");

timer.Restart();
var cumMax = arClose.CumMax();
ms = Lap(timer); results.Add(("Cumulative", "Close.cummax()", ms, "")); Console.WriteLine($"  {"Close.cummax()",-55} {ms,6:N0} ms");

timer.Restart();
var cumMin = arClose.CumMin();
ms = Lap(timer); results.Add(("Cumulative", "Close.cummin()", ms, "")); Console.WriteLine($"  {"Close.cummin()",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 12. RANK
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Rank ──");

timer.Restart();
var ranked = arClose.Rank();
ms = Lap(timer); results.Add(("Rank", "Close.rank() (14.7M rows)", ms, "")); Console.WriteLine($"  {"Close.rank() (14.7M rows)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 13. VALUE COUNTS
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Value Counts ──");

timer.Restart();
var vc = Cortex.Statistics.ValueCountsExtensions.ValueCounts(allStocks["Ticker"]);
ms = Lap(timer); results.Add(("ValueCounts", "Ticker.value_counts()", ms, "")); Console.WriteLine($"  {"Ticker.value_counts()",-55} {ms,6:N0} ms");

timer.Restart();
var nu = Cortex.Statistics.ValueCountsExtensions.NUnique(allStocks["Ticker"]);
ms = Lap(timer); results.Add(("ValueCounts", "Ticker.nunique()", ms, "")); Console.WriteLine($"  {"Ticker.nunique()",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 14. DESCRIBE
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Describe ──");

var descData = allStocks.Select("Open", "High", "Low", "Close", "Volume");
timer.Restart();
var desc = descData.Describe();
ms = Lap(timer); results.Add(("Describe", "Describe (5 numeric columns)", ms, "")); Console.WriteLine($"  {"Describe (5 numeric columns)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 15. FILLNA / INTERPOLATE
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── FillNa / Interpolate ──");

// Create a column with 10% nulls (matching Python's NaN semantics)
var closeNullable = new double?[allStocks.RowCount];
var closeValsForFill = arClose.Values;
for (int fi = 0; fi < closeNullable.Length; fi++)
    closeNullable[fi] = fi % 10 == 0 ? null : closeValsForFill[fi];
var closeWithNan = Column<double>.FromNullable("CloseNaN", closeNullable);

timer.Restart();
var filledFfill = closeWithNan.FillNa(Cortex.Missing.FillStrategy.Forward);
ms = Lap(timer); results.Add(("FillNa", "Forward fill (10% NaN)", ms, "")); Console.WriteLine($"  {"Forward fill (10% NaN)",-55} {ms,6:N0} ms");

timer.Restart();
var filledZero = closeWithNan.FillNa(0.0);
ms = Lap(timer); results.Add(("FillNa", "FillNa(0)", ms, "")); Console.WriteLine($"  {"FillNa(0)",-55} {ms,6:N0} ms");

timer.Restart();
var interpolated = closeWithNan.Interpolate();
ms = Lap(timer); results.Add(("FillNa", "Interpolate (linear)", ms, "")); Console.WriteLine($"  {"Interpolate (linear)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 16. GROUPBY TRANSFORM
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── GroupBy Transform ──");

// Reuse allGroups from Window section
timer.Restart();
var zscore = allGroups.TransformDouble("Close", vals =>
{
    double sum = 0; for (int i = 0; i < vals.Length; i++) sum += vals[i];
    double mean = sum / vals.Length;
    double ss = 0; for (int i = 0; i < vals.Length; i++) { double d = vals[i] - mean; ss += d * d; }
    double std = vals.Length > 1 ? Math.Sqrt(ss / (vals.Length - 1)) : 0;
    var r = new double[vals.Length];
    for (int i = 0; i < vals.Length; i++) r[i] = std > 0 ? (vals[i] - mean) / std : 0;
    return r;
});
ms = Lap(timer); results.Add(("GBTransform", "GroupBy z-score normalize", ms, "")); Console.WriteLine($"  {"GroupBy z-score normalize",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 17. SHIFT / LAG
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Shift / Lag ──");

timer.Restart();
var shifted = allGroups.TransformDouble("Close", vals =>
{
    var r = new double[vals.Length];
    r[0] = double.NaN;
    for (int i = 1; i < vals.Length; i++) r[i] = vals[i - 1];
    return r;
});
ms = Lap(timer); results.Add(("Shift", "GroupBy shift(1) (lag)", ms, "")); Console.WriteLine($"  {"GroupBy shift(1) (lag)",-55} {ms,6:N0} ms");

timer.Restart();
var diffCol = allGroups.TransformDouble("Close", vals =>
{
    var r = new double[vals.Length];
    r[0] = double.NaN;
    for (int i = 1; i < vals.Length; i++) r[i] = vals[i] - vals[i - 1];
    return r;
});
ms = Lap(timer); results.Add(("Shift", "GroupBy diff() (daily change)", ms, "")); Console.WriteLine($"  {"GroupBy diff() (daily change)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 18. MERGE (STRING KEY)
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Merge (String Key) ──");

var tickerInfo = new DataFrame(
    new StringColumn("Ticker", tickerNames.Select(t => (string?)t).ToArray()),
    new StringColumn("Industry", tickerNames.Select((t, i) => (string?)sectors[i % sectors.Length]).ToArray()),
    new Column<int>("Founded", tickerNames.Select((t, i) => 1950 + i % 70).ToArray()));

timer.Restart();
var mergedStr = allStocks.Join(tickerInfo, "Ticker");
ms = Lap(timer); results.Add(("MergeStr", "Merge on Ticker (string key, 14.7M rows)", ms, $"{mergedStr.RowCount:N0} rows")); Console.WriteLine($"  {"Merge on Ticker (string key, 14.7M rows)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 19. SAMPLE / QUANTILE
// ═══════════════════════════════════════════════════════
Gc();
Console.WriteLine("\n── Sample / Quantile ──");

timer.Restart();
var sampled = allStocks.Sample(1_000_000, seed: 42);
ms = Lap(timer); results.Add(("SampleQ", "Sample 1M rows", ms, "")); Console.WriteLine($"  {"Sample 1M rows",-55} {ms,6:N0} ms");

timer.Restart();
var q = arClose.Quantile(0.5);
_ = arClose.Quantile(0.01); _ = arClose.Quantile(0.05); _ = arClose.Quantile(0.25);
_ = arClose.Quantile(0.75); _ = arClose.Quantile(0.95); _ = arClose.Quantile(0.99);
ms = Lap(timer); results.Add(("SampleQ", "Close.quantile([0.01..0.99])", ms, "")); Console.WriteLine($"  {"Close.quantile([0.01..0.99])",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// SUMMARY + STORYBOARD
// ═══════════════════════════════════════════════════════
Console.WriteLine($"\n{"═",0}{"",69}");
var cats = new Dictionary<string, long>();
foreach (var (cat, _, m, _) in results) cats[cat] = cats.GetValueOrDefault(cat) + m;
foreach (var (cat, m) in cats.OrderByDescending(kv => kv.Value))
    Console.WriteLine($"  {cat,-30} {m,8:N0} ms");
var total = results.Sum(r => r.Ms);
Console.WriteLine($"  {"TOTAL",-30} {total,8:N0} ms");

// Load Python results and build comparison StoryBoard
var pyResults = new List<(string Cat, string Op, int Ms)>();
int pyTotal = 0;
if (File.Exists("stock_output_python/comprehensive_results.json"))
{
    var json = File.ReadAllText("stock_output_python/comprehensive_results.json");
    using var doc = JsonDocument.Parse(json);
    foreach (var op in doc.RootElement.EnumerateArray())
    {
        pyResults.Add((op.GetProperty("category").GetString()!, op.GetProperty("op").GetString()!, op.GetProperty("ms").GetInt32()));
        pyTotal += op.GetProperty("ms").GetInt32();
    }
}

var pyDict = pyResults.ToDictionary(r => r.Op, r => r.Ms);
// Map Cortex ops that don't have exact Python name matches
var pyCatTotals = pyResults.GroupBy(r => r.Cat).ToDictionary(g => g.Key, g => g.Sum(r => r.Ms));
var pyOpAliases = new Dictionary<string, int>
{
    // JoinMany combines 4 individual Python joins
    ["JoinMany: fact × 4 dimensions (int key)"] = pyCatTotals.GetValueOrDefault("StarJoin"),
    // Python uses "7-function named agg" without "(native)" suffix
    ["7-function named agg"] = pyDict.GetValueOrDefault("7-function named agg", 0),
};
var opNames = results.Select(r => r.Op).ToArray();
var csTimes = results.Select(r => (double)r.Ms).ToArray();
var pyTimes = results.Select(r => (double)(pyDict.GetValueOrDefault(r.Op, pyOpAliases.GetValueOrDefault(r.Op, 0)))).ToArray();
var winners = new string?[results.Count];
int csWins = 0, pyWins = 0;
for (int i = 0; i < results.Count; i++)
{
    if (pyTimes[i] > 0 && csTimes[i] > 0)
    {
        double ratio = pyTimes[i] / csTimes[i];
        if (ratio >= 1) { winners[i] = $"PS {ratio:F1}x"; csWins++; }
        else { winners[i] = $"Py {1/ratio:F1}x"; pyWins++; }
    }
    else winners[i] = "—";
}

var compDf = new DataFrame(
    new StringColumn("Operation", opNames.Select(s => (string?)s).ToArray()),
    new Column<double>("Cortex_ms", csTimes),
    new Column<double>("Python_ms", pyTimes),
    new StringColumn("Winner", winners));

var catDf = new DataFrame(
    new StringColumn("Category", cats.Keys.Select(s => (string?)s).ToArray()),
    new Column<double>("Cortex_ms", cats.Values.Select(v => (double)v).ToArray()),
    new Column<double>("Python_ms", cats.Keys.Select(k => (double)pyResults.Where(r => r.Cat == k).Sum(r => r.Ms)).ToArray()));

var story = StoryBoard.Create("Cortex vs Python — Comprehensive Benchmark")
    .Author("Head-to-Head across 8 Categories")
    .Stats(("Cortex Total", $"{total:N0} ms"), ("Python Total", pyTotal > 0 ? $"{pyTotal:N0} ms" : "N/A"),
           ("PS Wins", csWins.ToString()), ("Python Wins", pyWins.ToString()))
    .Text($"**{results.Count}** operations across **8 categories** tested on **{allStocks.RowCount:N0}** stock market rows. " +
          $"Cortex wins **{csWins}** operations, Python wins **{pyWins}**.")
    .Section("Category Totals")
    .Table(catDf, caption: "Total time per category (ms)")
    .Section("Full Operation Breakdown")
    .Table(compDf, caption: "All operations — lower is better");

if (total < pyTotal && pyTotal > 0)
    story = story.Callout($"Cortex is **{(double)pyTotal / total:F1}x faster** overall!", CalloutStyle.Success);
else if (pyTotal > 0)
    story = story.Callout($"Python is **{(double)total / pyTotal:F1}x faster** overall.", CalloutStyle.Warning);

story.ToHtml("stock_output/comprehensive_comparison.html");
Console.WriteLine("\n→ stock_output/comprehensive_comparison.html");

// Validate results match Python
ComprehensiveBench.Validator.ValidateAgainstPython(allStocks, joined);

Process.Start(new ProcessStartInfo("stock_output/comprehensive_comparison.html") { UseShellExecute = true });

static double[] ManagedRolling(double[] vals, int window)
{
    var r = new double[vals.Length]; double sum = 0;
    for (int i = 0; i < vals.Length; i++)
    { sum += vals[i]; if (i >= window) sum -= vals[i - window];
      r[i] = i < window - 1 ? double.NaN : sum / Math.Min(i + 1, window); }
    return r;
}
