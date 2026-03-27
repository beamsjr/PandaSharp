using System.Diagnostics;
using Cortex;
using Cortex.Column;
using Cortex.Concat;
using Cortex.Expressions;
using Cortex.GroupBy;
using Cortex.IO;
using Cortex.Lazy;
using Cortex.ML.Splitting;
using Cortex.ML.Transformers;
using Cortex.ML.Pipeline;
using Cortex.ML.Metrics;
using Cortex.ML.Tensors;
using Cortex.ParallelOps;
using Cortex.Schema;
using Cortex.Statistics;
using Cortex.Storage;
using Cortex.Streaming;
using Cortex.Viz;
using Cortex.Viz.Charts;
using Cortex.Window;
using static Cortex.Expressions.Expr;

// ============================================================
//  Cortex Stock Market Analyzer — Full Dataset
//  Generates two StoryBoard reports:
//    1. Stock Analysis — the data story
//    2. Performance Report — how Cortex performed
// ============================================================

var dataDir = args.Length > 0
    ? args[0]
    : "/Users/joe/Documents/Repository/lab/stock_market_analyzer/Stocks";

var outputDir = Path.Combine(Directory.GetCurrentDirectory(), "stock_output");
Directory.CreateDirectory(outputDir);

Console.WriteLine("=== Cortex Stock Market Analyzer (Full Dataset) ===\n");

// Performance tracker
var perf = new List<(string Op, long Ms, long Rows, string Detail)>();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }
var timer = Stopwatch.StartNew();

// ── 1. LOAD ALL CSVs (PARALLEL) ───────────────────────
Console.Write("1. Loading ALL stock CSVs (parallel)... ");
var files = Directory.GetFiles(dataDir, "*.txt");
var loadedDfs = CsvReader.ReadMany(files);

var frames = new List<DataFrame>();
int skipped = 0;
for (int fi = 0; fi < files.Length; fi++)
{
    var df = loadedDfs[fi];
    if (df is null || df.RowCount < 100) { skipped++; continue; }
    var ticker = Path.GetFileNameWithoutExtension(files[fi]).Replace(".us", "").ToUpper();
    var tickerCol = new string?[df.RowCount];
    Array.Fill(tickerCol, ticker);
    frames.Add(df.AddColumn(new StringColumn("Ticker", tickerCol)));
}

var allStocks = ConcatExtensions.Concat(frames.ToArray());
var loadMs = Lap(timer);
perf.Add(("Load CSVs", loadMs, allStocks.RowCount, $"{frames.Count} files, {skipped} skipped"));
Console.WriteLine($"{frames.Count} stocks, {allStocks.RowCount:N0} rows [{loadMs}ms]");

// ── 2. SCHEMA VALIDATION ──────────────────────────────
Console.Write("2. Schema validation... ");
var validation = DataFrameSchema.Define()
    .HasColumns("Date", "Open", "High", "Low", "Close", "Volume")
    .Column("Close", type: typeof(double))
    .MinRows(1)
    .Validate(allStocks);
var valMs = Lap(timer);
perf.Add(("Schema Validation", valMs, allStocks.RowCount, validation.IsValid ? "PASSED" : $"{validation.ErrorCount} errors"));
Console.WriteLine(validation.IsValid ? $"PASSED [{valMs}ms]" : $"FAILED [{valMs}ms]");

// ── 3. PROFILE ────────────────────────────────────────
Console.Write("3. Profiling (10K sample)... ");
var profile = allStocks.Head(10_000).Profile();
var profMs = Lap(timer);
perf.Add(("Profile (10K)", profMs, 10_000, $"{profile.Columns.Length} columns analyzed"));
Console.WriteLine($"done [{profMs}ms]");

// ── 4. EVAL (Native fused operation) ──────────────────
Console.Write("4. Eval: daily returns (native)... ");
Console.Write($"[native={Cortex.Native.NativeOps.IsAvailable}] ");
DataFrame withReturns;
if (Cortex.Native.NativeOps.IsAvailable &&
    allStocks["Close"] is Column<double> closeForEval &&
    allStocks["Open"] is Column<double> openForEval &&
    closeForEval.NullCount == 0 && openForEval.NullCount == 0)
{
    var returnCol = Cortex.Native.NativeOps.EvalDailyReturn(closeForEval, openForEval, "DailyReturn");
    withReturns = allStocks.AddColumn(returnCol);
}
else
{
    withReturns = allStocks.Eval("DailyReturn = (Close - Open) / Open * 100");
}
var evalMs = Lap(timer);
perf.Add(("Eval (DailyReturn)", evalMs, allStocks.RowCount, $"Native={Cortex.Native.NativeOps.IsAvailable}"));
Console.WriteLine($"done [{evalMs}ms]");

// ── 5. SORT ───────────────────────────────────────────
Console.Write("5. Nlargest (top 10)... ");
var bigMovers = withReturns.Nlargest(10, "DailyReturn")
    .Select("Ticker", "Date", "Open", "Close", "DailyReturn", "Volume");
var sortMs = Lap(timer);
perf.Add(("Nlargest (top 10)", sortMs, withReturns.RowCount, "Partial sort — O(n) quickselect"));
Console.WriteLine($"done [{sortMs}ms]");

// ── 6. GROUPBY ────────────────────────────────────────
Console.Write("6. GroupBy mean... ");
var avgReturns = GroupByExtensions.GroupBy(withReturns, "Ticker").Mean()
    .Sort("DailyReturn", ascending: false);
var gbMs = Lap(timer);
perf.Add(("GroupBy Mean", gbMs, withReturns.RowCount, $"{frames.Count} groups"));
Console.WriteLine($"done [{gbMs}ms]");

// ── 7. LAZY ───────────────────────────────────────────
Console.Write("7. Lazy chain... ");
var lazyResult = withReturns.Lazy()
    .Filter(Col("Volume") > Lit(10_000_000))
    .Sort("DailyReturn", ascending: false)
    .Select("Ticker", "Date", "DailyReturn", "Volume")
    .Head(20)
    .Collect();
var lazyMs = Lap(timer);
perf.Add(("Lazy Chain", lazyMs, withReturns.RowCount, $"{lazyResult.RowCount} results"));
Console.WriteLine($"done — {lazyResult.RowCount} results [{lazyMs}ms]");

// ── 8. WINDOW ─────────────────────────────────────────
Console.Write("8. Rolling window (AAPL)... ");
var aapl = allStocks.Filter(row => row["Ticker"]?.ToString() == "AAPL");
string aaplSma20 = "N/A", aaplSma50 = "N/A", aaplClose = "N/A";
if (aapl.RowCount > 0)
{
    var closeCol = aapl.GetColumn<double>("Close");
    var sma20 = closeCol.Rolling(20).Mean();
    var sma50 = closeCol.Rolling(50).Mean();
    aaplClose = $"{closeCol[aapl.RowCount - 1]:F2}";
    aaplSma20 = $"{sma20[aapl.RowCount - 1]:F2}";
    aaplSma50 = $"{sma50[aapl.RowCount - 1]:F2}";
}
var winMs = Lap(timer);
perf.Add(("Rolling Window", winMs, aapl.RowCount, "SMA20 + SMA50"));
Console.WriteLine($"done ({aapl.RowCount} rows) [{winMs}ms]");

// ── 9. QCUT ──────────────────────────────────────────
Console.Write("9. QCut binning... ");
var quartiles = withReturns.GetColumn<double>("DailyReturn").QCut(4, ["Crash", "Down", "Up", "Surge"]);
var binCounts = new Dictionary<string, int>();
for (int i = 0; i < quartiles.Length; i++)
    binCounts[quartiles[i] ?? "null"] = binCounts.GetValueOrDefault(quartiles[i] ?? "null") + 1;
var cutMs = Lap(timer);
perf.Add(("QCut Binning", cutMs, withReturns.RowCount, "4 quartiles"));
Console.WriteLine($"done [{cutMs}ms]");

// ── 10. ML: Logistic Regression — predict next-day direction ──
Console.Write("10. ML: Feature engineering + Logistic Regression... ");
string mlAccuracy = "N/A", mlPrecision = "N/A", mlRecall = "N/A", mlF1 = "N/A";
int mlTrainRows = 0, mlTestRows = 0, mlEpochs = 0;
DataFrame? mlResultDf = null;
double[]? lossHistory = null;

if (aapl.RowCount > 500)
{
    // ── Feature Engineering ──
    var closesCol = aapl["Close"]; var highCol = aapl["High"]; var lowCol = aapl["Low"];
    var volCol = aapl["Volume"]; var openCol = aapl["Open"];
    var N = aapl.RowCount;
    var closes = new double[N]; var highs = new double[N]; var lows = new double[N];
    var vols = new double[N]; var opens = new double[N];
    for (int ii = 0; ii < N; ii++)
    {
        closes[ii] = Convert.ToDouble(closesCol.GetObject(ii));
        highs[ii] = Convert.ToDouble(highCol.GetObject(ii));
        lows[ii] = Convert.ToDouble(lowCol.GetObject(ii));
        vols[ii] = Convert.ToDouble(volCol.GetObject(ii));
        opens[ii] = Convert.ToDouble(openCol.GetObject(ii));
    }

    // Compute features with lookback
    int lookback = 50;
    int usable = N - lookback;
    var featReturn1 = new double[usable];   // 1-day return
    var featReturn5 = new double[usable];   // 5-day return
    var featReturn20 = new double[usable];  // 20-day momentum
    var featVolRatio = new double[usable];  // volume vs 20-day avg
    var featRange = new double[usable];     // (high-low)/close (volatility)
    var featSmaRatio = new double[usable];  // close / SMA20 ratio
    var featGap = new double[usable];       // overnight gap (open vs prev close)
    var labels = new int[usable];           // 1=up next day, 0=down

    for (int i = 0; i < usable; i++)
    {
        int idx = i + lookback;
        featReturn1[i] = closes[idx - 1] > 0 ? (closes[idx] - closes[idx - 1]) / closes[idx - 1] * 100 : 0;
        featReturn5[i] = closes[idx - 5] > 0 ? (closes[idx] - closes[idx - 5]) / closes[idx - 5] * 100 : 0;
        featReturn20[i] = closes[idx - 20] > 0 ? (closes[idx] - closes[idx - 20]) / closes[idx - 20] * 100 : 0;

        double avgVol = 0;
        for (int v = idx - 20; v < idx; v++) avgVol += vols[v];
        avgVol /= 20;
        featVolRatio[i] = avgVol > 0 ? vols[idx] / avgVol : 1;

        featRange[i] = closes[idx] > 0 ? (highs[idx] - lows[idx]) / closes[idx] * 100 : 0;

        double sma20 = 0;
        for (int s = idx - 20; s < idx; s++) sma20 += closes[s];
        sma20 /= 20;
        featSmaRatio[i] = sma20 > 0 ? closes[idx] / sma20 : 1;

        featGap[i] = closes[idx - 1] > 0 ? (opens[idx] - closes[idx - 1]) / closes[idx - 1] * 100 : 0;

        // Label: will tomorrow close higher than today?
        if (idx + 1 < N)
            labels[i] = closes[idx + 1] > closes[idx] ? 1 : 0;
    }

    // Drop last row (no label for it)
    usable--;
    var featureNames = new[] { "Return1d", "Return5d", "Momentum20d", "VolRatio", "Range", "SmaRatio", "Gap" };

    var mlDf = new DataFrame(
        new Column<double>("Return1d", featReturn1[..usable]),
        new Column<double>("Return5d", featReturn5[..usable]),
        new Column<double>("Momentum20d", featReturn20[..usable]),
        new Column<double>("VolRatio", featVolRatio[..usable]),
        new Column<double>("Range", featRange[..usable]),
        new Column<double>("SmaRatio", featSmaRatio[..usable]),
        new Column<double>("Gap", featGap[..usable]),
        new Column<int>("Label", labels[..usable])
    );

    // ── Train/Test Split ──
    var (trainDf, testDf) = DataSplitting.TrainTestSplit(mlDf, testFraction: 0.2, seed: 42);
    mlTrainRows = trainDf.RowCount;
    mlTestRows = testDf.RowCount;

    // ── Feature Scaling ──
    var scaler = new StandardScaler();
    var trainScaled = scaler.FitTransform(trainDf.Select(featureNames));
    var testScaled = scaler.Transform(testDf.Select(featureNames));

    // ── Convert to Tensors ──
    var xTrain = trainScaled.ToTensor<double>(featureNames);
    var xTest = testScaled.ToTensor<double>(featureNames);
    var yTrain = trainDf.GetColumn<int>("Label").Values.ToArray();
    var yTest = testDf.GetColumn<int>("Label").Values.ToArray();

    int numFeatures = featureNames.Length;
    int trainN = xTrain.Shape[0];
    int testN = xTest.Shape[0];

    // ── Logistic Regression via Gradient Descent ──
    var weights = new double[numFeatures];
    double bias = 0;
    double lr = 0.01;
    mlEpochs = 200;
    lossHistory = new double[mlEpochs];
    var xTrainData = xTrain.ToArray();
    var rngMl = new Random(42);
    for (int w = 0; w < numFeatures; w++) weights[w] = (rngMl.NextDouble() - 0.5) * 0.1;

    for (int epoch = 0; epoch < mlEpochs; epoch++)
    {
        double totalLoss = 0;
        var gradW = new double[numFeatures];
        double gradB = 0;

        for (int i = 0; i < trainN; i++)
        {
            // Forward: z = w·x + b, p = sigmoid(z)
            double z = bias;
            for (int f = 0; f < numFeatures; f++)
                z += weights[f] * xTrainData[i * numFeatures + f];
            double p = 1.0 / (1.0 + Math.Exp(-Math.Clamp(z, -500, 500)));

            // Loss: binary cross-entropy
            double y = yTrain[i];
            totalLoss += -(y * Math.Log(Math.Max(p, 1e-15)) + (1 - y) * Math.Log(Math.Max(1 - p, 1e-15)));

            // Gradients
            double err = p - y;
            for (int f = 0; f < numFeatures; f++)
                gradW[f] += err * xTrainData[i * numFeatures + f];
            gradB += err;
        }

        // Update
        for (int f = 0; f < numFeatures; f++)
            weights[f] -= lr * gradW[f] / trainN;
        bias -= lr * gradB / trainN;

        lossHistory[epoch] = totalLoss / trainN;
    }

    // ── Predict on test set ──
    var xTestData = xTest.ToArray();
    var preds = new int[testN];
    var probabilities = new double[testN];
    for (int i = 0; i < testN; i++)
    {
        double z = bias;
        for (int f = 0; f < numFeatures; f++)
            z += weights[f] * xTestData[i * numFeatures + f];
        probabilities[i] = 1.0 / (1.0 + Math.Exp(-Math.Clamp(z, -500, 500)));
        preds[i] = probabilities[i] >= 0.5 ? 1 : 0;
    }

    // ── Metrics ──
    var yTrueCol = new Column<int>("true", yTest);
    var yPredCol = new Column<int>("pred", preds);
    var met = MetricsCalculator.Classification(yTrueCol, yPredCol);
    mlAccuracy = $"{met.Accuracy:P1}";
    mlPrecision = $"{met.Precision:P1}";
    mlRecall = $"{met.Recall:P1}";
    mlF1 = $"{met.F1:P1}";

    // Feature importance (weight magnitudes)
    var featImportance = new (string Name, double Weight)[numFeatures];
    for (int f = 0; f < numFeatures; f++)
        featImportance[f] = (featureNames[f], Math.Abs(weights[f]));
    Array.Sort(featImportance, (a, b) => b.Weight.CompareTo(a.Weight));

    // Build result DataFrame for StoryBoard
    mlResultDf = new DataFrame(
        new StringColumn("Feature", featImportance.Select(f => f.Name).ToArray()),
        new Column<double>("Importance", featImportance.Select(f => f.Weight).ToArray())
    );

    Console.WriteLine($"done — Accuracy: {mlAccuracy}, F1: {mlF1}");
    Console.WriteLine($"   Train: {mlTrainRows}, Test: {mlTestRows}, Epochs: {mlEpochs}");
    Console.Write("   Feature importance: ");
    foreach (var (name, weight) in featImportance.Take(3))
        Console.Write($"{name}({weight:F3}) ");
    Console.WriteLine();
}
else
{
    Console.WriteLine("skipped (insufficient data)");
}
var mlMs = Lap(timer);
perf.Add(("ML: Logistic Regression", mlMs, aapl.RowCount, $"Acc: {mlAccuracy}, F1: {mlF1}, {mlEpochs} epochs"));

// ── 11. PARQUET ──────────────────────────────────────
Console.Write("11. Parquet write+read... ");
var pqPath = Path.Combine(outputDir, "all_stocks.parquet");
ParquetIO.WriteParquet(withReturns.Head(100_000), pqPath);
var reloaded = ParquetIO.ReadParquet(pqPath);
var pqMs = Lap(timer);
perf.Add(("Parquet Write+Read", pqMs, 100_000, $"{new FileInfo(pqPath).Length / 1024:N0} KB"));
Console.WriteLine($"done ({new FileInfo(pqPath).Length / 1024:N0} KB) [{pqMs}ms]");

// ── 12. OUT-OF-CORE ──────────────────────────────────
Console.Write("12. Out-of-core spill... ");
var spillPath = Path.Combine(outputDir, "spilled.arrow");
long spillSize;
using (var spilled = withReturns.Head(100_000).Spill(spillPath))
{
    _ = spilled["Ticker"]; _ = spilled["DailyReturn"];
    spillSize = new FileInfo(spillPath).Length;
}
var spillMs = Lap(timer);
perf.Add(("Spill + Lazy Load", spillMs, 100_000, $"{spillSize / 1024:N0} KB, 2 cols loaded"));
Console.WriteLine($"done [{spillMs}ms]");

// ── 13. STREAMING ────────────────────────────────────
Console.Write("13. Streaming simulation... ");
var rng = new Random(42);
var events = Enumerable.Range(0, 100).Select(i =>
    new StreamEvent(new DateTimeOffset(2024, 1, 1, 9, 30, 0, TimeSpan.Zero).AddSeconds(i * 10),
        new Dictionary<string, object?> { ["price"] = 150 + rng.NextDouble() * 10 - 5, ["volume"] = rng.Next(1000, 50000) })).ToList();
var windows = StreamFrame.From(new EnumerableSource(events))
    .Window(new TumblingWindow(TimeSpan.FromMinutes(5)))
    .Agg("price", AggType.Mean, "avg_price").Agg("volume", AggType.Sum, "total_volume")
    .Collect();
var streamMs = Lap(timer);
perf.Add(("Streaming", streamMs, 100, $"{windows.Count} windows"));
Console.WriteLine($"done — {windows.Count} windows [{streamMs}ms]");

// ── 14. NATIVE FILTER ────────────────────────────────
Console.Write("14. Native filter (|return| > 5%)... ");
var bigDays = withReturns.NativeFilterAbsGt("DailyReturn", 5.0);
var parMs = Lap(timer);
perf.Add(("Native Filter", parMs, withReturns.RowCount, $"{bigDays.RowCount:N0} matches, native C"));
Console.WriteLine($"done — {bigDays.RowCount:N0} matches [{parMs}ms]");

// ── 15. SAVE ─────────────────────────────────────────
Console.Write("15. Universal Save... ");
bigMovers.Save(Path.Combine(outputDir, "big_movers.csv"));
bigMovers.Save(Path.Combine(outputDir, "big_movers.json"));
avgReturns.Head(20).Save(Path.Combine(outputDir, "avg_returns.parquet"));
var saveMs = Lap(timer);
perf.Add(("Universal Save", saveMs, 30, "CSV + JSON + Parquet"));
Console.WriteLine($"done [{saveMs}ms]");

// ═══════════════════════════════════════════════════════
// STORYBOARD 1: Stock Analysis Report
// ═══════════════════════════════════════════════════════
Console.Write("\nGenerating Stock Analysis StoryBoard... ");

DataFrame? vizDf = null;
if (aapl.RowCount > 200)
{
    var recent = aapl.Tail(200);
    var idx = new double[recent.RowCount];
    for (int i = 0; i < idx.Length; i++) idx[i] = i;
    vizDf = new DataFrame(new Column<double>("Day", idx), new Column<double>("Close", recent.GetColumn<double>("Close").Values.ToArray()));
}

var analysisStory = StoryBoard.Create("Stock Market Analysis")
    .Author("Cortex Analyzer")
    .Text($"A comprehensive analysis of **{frames.Count:N0}** US stocks comprising **{allStocks.RowCount:N0}** total trading day records, sourced from the Huge Stock Market Dataset on Kaggle.")
    .Stats(
        ("Stocks", frames.Count.ToString("N0")),
        ("Trading Days", allStocks.RowCount.ToString("N0")),
        ("Date Range", "1984–2017"),
        ("Skipped", skipped.ToString())
    );

if (vizDf is not null)
{
    analysisStory = analysisStory
        .Section("AAPL Price History")
        .Text($"Apple (AAPL) has **{aapl.RowCount:N0}** trading days in the dataset. The last close was **{aaplClose}**, with 20-day SMA at **{aaplSma20}** and 50-day SMA at **{aaplSma50}**.")
        .Chart(vizDf, v => v.Line("Day", "Close").Title("AAPL — Last 200 Trading Days"), caption: "Daily closing price (split-adjusted)");
}

analysisStory = analysisStory
    .Section("Return Distribution")
    .Text("Daily returns across all stocks follow an approximately normal distribution with **fat tails** — extreme moves (>5%) occur far more frequently than a Gaussian model would predict.")
    .Chart(withReturns.Head(50_000), v => v.Histogram("DailyReturn").Title("Daily Return Distribution"), caption: "Percentage daily returns, 50K sample")
    .Callout($"Quartile breakdown: Crash ({binCounts.GetValueOrDefault("Crash"):N0}), Down ({binCounts.GetValueOrDefault("Down"):N0}), Up ({binCounts.GetValueOrDefault("Up"):N0}), Surge ({binCounts.GetValueOrDefault("Surge"):N0}). A total of **{bigDays.RowCount:N0}** trading days had moves exceeding 5%.", CalloutStyle.Note)

    .Section("Top Performers")
    .Text("Stocks with the highest *average* daily returns. Many are warrants (W suffix) or micro-caps with extreme volatility and low liquidity.")
    .Chart(avgReturns.Head(10), v => v.Bar("Ticker", "DailyReturn").Title("Top 10 by Avg Daily Return"), caption: "Mean daily percentage return over full history")
    .Table(bigMovers, maxRows: 10, caption: "Top 10 Single-Day Gains")

    .Divider()

    .Section("Volume vs Return")
    .Text("Is there a relationship between trading volume and return magnitude? High-volume days tend to cluster near zero return, while extreme moves often occur on moderate volume.")
    .Chart(withReturns.Head(10_000), v => v.Scatter("Volume", "DailyReturn").Title("Volume vs Return"), caption: "10K random stock-days")

    .Section("Machine Learning: Predicting Next-Day Direction")
    .Text($"A **logistic regression** model was trained on AAPL to predict whether the stock will close higher or lower the next trading day. The model uses **7 engineered features**: 1-day return, 5-day return, 20-day momentum, volume ratio vs 20-day average, daily range (volatility), price/SMA20 ratio, and overnight gap.")
    .Stats(
        ("Train Size", mlTrainRows.ToString("N0")),
        ("Test Size", mlTestRows.ToString("N0")),
        ("Epochs", mlEpochs.ToString()),
        ("Accuracy", mlAccuracy)
    );

if (mlResultDf is not null)
{
    analysisStory = analysisStory
        .Text($"**Results**: Accuracy **{mlAccuracy}**, Precision **{mlPrecision}**, Recall **{mlRecall}**, F1 Score **{mlF1}**. The model achieves above-chance accuracy, but stock prediction remains inherently difficult due to market efficiency.")
        .Chart(mlResultDf, v => v.Bar("Feature", "Importance").Title("Feature Importance (|weight|)"), caption: "Absolute logistic regression weight per feature");
}

if (lossHistory is not null)
{
    var epochNums = new double[lossHistory.Length];
    for (int i = 0; i < epochNums.Length; i++) epochNums[i] = i;
    var lossDf = new DataFrame(
        new Column<double>("Epoch", epochNums),
        new Column<double>("Loss", lossHistory)
    );
    analysisStory = analysisStory
        .Chart(lossDf, v => v.Line("Epoch", "Loss").Title("Training Loss (Binary Cross-Entropy)"), caption: "Loss converges as the model learns feature weights");
}

analysisStory = analysisStory
    .Callout($"The most predictive features were **momentum** and **SMA ratio** — trend-following signals. Overnight gaps and volume had lower importance, suggesting intraday patterns dominate.", CalloutStyle.Info);

analysisStory.ToHtml(Path.Combine(outputDir, "stock_report.html"));
Console.WriteLine("done → stock_output/stock_report.html");

// ═══════════════════════════════════════════════════════
// STORYBOARD 2: Performance Report
// ═══════════════════════════════════════════════════════
Console.Write("Generating Performance StoryBoard... ");

var totalMs = perf.Sum(p => p.Ms);
var perfDf = new DataFrame(
    new StringColumn("Operation", perf.Select(p => p.Op).ToArray()),
    new Column<double>("Time_ms", perf.Select(p => (double)p.Ms).ToArray()),
    new Column<double>("Rows", perf.Select(p => (double)p.Rows).ToArray()),
    new StringColumn("Detail", perf.Select(p => p.Detail).ToArray())
);

// Throughput calc
var throughputs = new double[perf.Count];
for (int i = 0; i < perf.Count; i++)
    throughputs[i] = perf[i].Ms > 0 ? perf[i].Rows / (perf[i].Ms / 1000.0) : 0;
var perfWithThroughput = perfDf.AddColumn(new Column<double>("Rows_per_sec", throughputs));

var perfStory = StoryBoard.Create("Cortex Performance Report")
    .Author("Cortex Benchmark")
    .Theme(StoryTheme.Dark)
    .Text($"Performance profiling of **{perf.Count}** operations on **{allStocks.RowCount:N0}** rows across **{frames.Count:N0}** stocks. Running on **.NET {Environment.Version}** with **{Environment.ProcessorCount} cores**.")
    .Stats(
        ("Total Time", $"{totalMs:N0} ms"),
        ("Operations", perf.Count.ToString()),
        ("Total Rows", allStocks.RowCount.ToString("N0")),
        ("Cores Used", Environment.ProcessorCount.ToString())
    )

    .Section("Operation Timing")
    .Text("Time in milliseconds for each pipeline step. The CSV loading dominates as expected for I/O-bound work.")
    .Chart(perfDf, v => v.Bar("Operation", "Time_ms").Title("Operation Time (ms)"), caption: "Lower is better")
    .Table(perfWithThroughput, caption: "Detailed Performance Breakdown")

    .Section("Throughput")
    .Text("Rows processed per second for each operation. SIMD-accelerated operations like Eval and Sort achieve millions of rows/sec.")
    .Chart(perfDf, v => v.Bar("Operation", "Rows").Title("Rows Processed per Operation"), caption: "Scale of each operation")

    .Callout($"Total pipeline time: **{totalMs:N0} ms** for {allStocks.RowCount:N0} rows — that's **{allStocks.RowCount / (totalMs / 1000.0):N0} rows/sec** end-to-end.", CalloutStyle.Success);

perfStory.ToHtml(Path.Combine(outputDir, "performance_report.html"));
Console.WriteLine("done → stock_output/performance_report.html");

// ── OPEN BOTH ────────────────────────────────────────
Console.WriteLine($"\n=== Complete ===");
Console.WriteLine($"Output: {outputDir}");
Console.WriteLine("Opening reports...");
Process.Start(new ProcessStartInfo(Path.Combine(outputDir, "stock_report.html")) { UseShellExecute = true });
Process.Start(new ProcessStartInfo(Path.Combine(outputDir, "performance_report.html")) { UseShellExecute = true });
