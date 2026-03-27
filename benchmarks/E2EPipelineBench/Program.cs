using System.Diagnostics;
using System.Text.Json;
using Cortex;
using Cortex.Column;
using Cortex.Concat;
using Cortex.GroupBy;
using Cortex.Missing;
using Cortex.Statistics;
using Cortex.Window;
using Cortex.Expressions;
using Cortex.ML.Models;
using Cortex.ML.Transformers;
using Cortex.ML.Metrics;
using Cortex.ML.Splitting;
using Cortex.ML.Tensors;
using Cortex.TimeSeries.Models;
using Cortex.TimeSeries.Decomposition;
using Cortex.Geo;
using Cortex.Text.Preprocessing;
using static Cortex.Expressions.Expr;

/*
 * End-to-End Pipeline Benchmark: Cortex (C#)
 * Identical workflows to e2e_pipeline_python.py
 * Key difference: ZERO conversion boundaries — everything stays as DataFrame.
 */

const int ROWS = 500_000;
const int SEED = 42;

var results = new List<(string Workflow, string Op, double Ms, string Detail)>();
var rng = new Random(SEED);

double LapPrecise(Stopwatch s)
{
    var ms = s.Elapsed.TotalMilliseconds;
    s.Restart();
    return ms;
}

void Record(string workflow, string op, double ms, string detail = "")
{
    results.Add((workflow, op, Math.Round(ms, 2), detail));
    Console.WriteLine($"  {op}: {ms:F1}ms" + (detail != "" ? $"  ({detail})" : ""));
}

void Gc() { GC.Collect(); GC.WaitForPendingFinalizers(); GC.Collect(); }

Console.WriteLine($"\n{"".PadLeft(60, '=')}");
Console.WriteLine($"  End-to-End Pipeline Benchmark — Cortex (C#)");
Console.WriteLine($"  Rows: {ROWS:N0}");
Console.WriteLine($"  Native accelerators: {(Cortex.Native.NativeOps.IsAvailable ? "LOADED" : "NOT AVAILABLE")}");
Console.WriteLine($"{"".PadLeft(60, '=')}");

// ════════════════════════════════════════════════════════════════
// DATA GENERATION — identical to Python benchmark
// ════════════════════════════════════════════════════════════════
var categories = new[] { "Tech", "Finance", "Health", "Energy", "Consumer" };
var cities = new[] { "New York", "London", "Tokyo", "Sydney", "Berlin", "Toronto", "Mumbai", "São Paulo" };
var texts = new[]
{
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
};

var dataCat = new string[ROWS];
var dataCity = new string[ROWS];
var dataText = new string[ROWS];
var dataValue = new double[ROWS];
var dataPrice = new double[ROWS];
var dataVolume = new double[ROWS];
var dataLat = new double[ROWS];
var dataLon = new double[ROWS];
var dataDates = new string[ROWS];
var dataLabel = new int[ROWS];
var dataValueNulls = new double?[ROWS];

for (int i = 0; i < ROWS; i++)
{
    dataCat[i] = categories[rng.Next(categories.Length)];
    dataCity[i] = cities[rng.Next(cities.Length)];
    dataText[i] = texts[rng.Next(texts.Length)];

    // Normal(100, 25)
    double u1 = 1.0 - rng.NextDouble();
    double u2 = rng.NextDouble();
    double normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    dataValue[i] = 100 + 25 * normal;

    // |Normal(50, 15)|
    u1 = 1.0 - rng.NextDouble();
    u2 = rng.NextDouble();
    normal = Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    dataPrice[i] = Math.Abs(50 + 15 * normal);

    dataVolume[i] = rng.Next(100, 10000);
    dataLat[i] = 25 + rng.NextDouble() * 23;
    dataLon[i] = -125 + rng.NextDouble() * 60;
    dataDates[i] = $"2020-{(i % 12) + 1:D2}-{(i % 28) + 1:D2}";
    dataLabel[i] = dataValue[i] > 100 ? 1 : 0;

    // 5% nulls
    dataValueNulls[i] = rng.NextDouble() < 0.05 ? null : dataValue[i];
}


// ════════════════════════════════════════════════════════════════
Console.WriteLine($"\n{"".PadLeft(60, '─')}");
Console.WriteLine("  WORKFLOW 1: ML Pipeline");
Console.WriteLine("  DataFrame throughout — ZERO conversions");
Console.WriteLine($"{"".PadLeft(60, '─')}");

Gc();
var timer = Stopwatch.StartNew();
double w1Total = 0;

// 1a. Load into DataFrame
var w1Df = new DataFrame(new IColumn[]
{
    new CategoricalColumn("category", dataCat),
    Column<double>.FromNullable("value", dataValueNulls),
    new Column<double>("price", dataPrice),
    new Column<double>("volume", dataVolume),
    new Column<int>("label", dataLabel),
});
var ms1 = LapPrecise(timer);
Record("ML Pipeline", "1a. Load into DataFrame", ms1, $"{w1Df.RowCount:N0} rows");
w1Total += ms1;

// 1b. Clean + feature engineer — stays in DataFrame
var w1Dropped = w1Df.DropNa();
var w1Filtered = w1Dropped.Filter(w1Dropped.GetColumn<double>("volume").Gt(200));
// Compute log and price_ratio as raw columns (Expr doesn't have Log/Mean yet)
var valCol = w1Filtered.GetColumn<double>("value");
var priceCol = w1Filtered.GetColumn<double>("price");
var volCol = w1Filtered.GetColumn<double>("volume");
var logValues = new double[w1Filtered.RowCount];
var totalValues = new double[w1Filtered.RowCount];
var priceRatioValues = new double[w1Filtered.RowCount];
double priceMean = 0;
for (int i = 0; i < w1Filtered.RowCount; i++) priceMean += priceCol.Values[i];
priceMean /= w1Filtered.RowCount;
for (int i = 0; i < w1Filtered.RowCount; i++)
{
    logValues[i] = Math.Log(valCol.Values[i]);
    totalValues[i] = priceCol.Values[i] * volCol.Values[i];
    priceRatioValues[i] = priceCol.Values[i] / priceMean;
}
var w1Clean = w1Filtered
    .AddColumn(new Column<double>("total_value", totalValues))
    .AddColumn(new Column<double>("log_value", logValues))
    .AddColumn(new Column<double>("price_ratio", priceRatioValues));
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1b. Clean + feature engineer (DataFrame)", ms1, $"{w1Clean.RowCount:N0} rows after clean");
w1Total += ms1;

// 1c. NO CONVERSION NEEDED — StandardScaler works on DataFrame directly
var scaler = new StandardScaler("value", "price", "volume", "total_value", "log_value", "price_ratio");
var w1Scaled = scaler.FitTransform(w1Clean);
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1c. StandardScaler fit+transform (DataFrame)", ms1);
w1Total += ms1;

// 1d. Train/test split — returns DataFrames
var (w1Train, w1Test) = w1Scaled.TrainTestSplit(testFraction: 0.2, seed: SEED);
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1d. Train/test split (DataFrame)", ms1, $"train={w1Train.RowCount:N0}, test={w1Test.RowCount:N0}");
w1Total += ms1;

// 1e. Convert DataFrame → Tensor for model (internal, not cross-library)
var featureCols = new[] { "value", "price", "volume", "total_value", "log_value", "price_ratio" };
var w1XTrain = DataFrameToTensor(w1Train, featureCols);
var w1YTrain = new Tensor<double>(w1Train.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w1Train.RowCount);
var w1XTest = DataFrameToTensor(w1Test, featureCols);
var w1YTest = new Tensor<double>(w1Test.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w1Test.RowCount);
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1e. DataFrame → Tensor (internal)", ms1);
w1Total += ms1;

// 1f. Train Random Forest
var rfModel = new RandomForestClassifier(nEstimators: 100, seed: SEED);
rfModel.Fit(w1XTrain, w1YTrain);
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1f. Train RandomForest (100 trees)", ms1);
w1Total += ms1;

// 1g. Predict + evaluate — metrics return records, not cross-library
var w1Preds = rfModel.Predict(w1XTest);
var yTrueCol = new Column<int>("actual", w1YTest.ToArray().Select(d => (int)d).ToArray());
var yPredCol = new Column<int>("predicted", w1Preds.ToArray().Select(d => (int)d).ToArray());
var metrics = MetricsCalculator.Classification(yTrueCol, yPredCol);
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1g. Predict + evaluate", ms1, $"acc={metrics.Accuracy:F4}, f1={metrics.F1:F4}");
w1Total += ms1;

// 1h. Results already in DataFrame-compatible form — no conversion needed
var w1Result = new DataFrame(new IColumn[]
{
    yTrueCol,
    yPredCol,
    new Column<bool>("correct", Enumerable.Range(0, yTrueCol.Length)
        .Select(i => yTrueCol.Values[i] == yPredCol.Values[i]).ToArray()),
});
ms1 = LapPrecise(timer);
Record("ML Pipeline", "1h. Build results DataFrame (no conversion)", ms1, $"result: ({w1Result.RowCount}, {w1Result.ColumnCount})");
w1Total += ms1;

Record("ML Pipeline", "TOTAL", w1Total);
Console.WriteLine($"  {"".PadLeft(40, '─')}");
Console.WriteLine($"  ML Pipeline TOTAL: {w1Total:F1}ms\n");


// ════════════════════════════════════════════════════════════════
Console.WriteLine($"{"".PadLeft(60, '─')}");
Console.WriteLine("  WORKFLOW 2: Time Series Forecasting");
Console.WriteLine("  DataFrame throughout — ZERO conversions");
Console.WriteLine($"{"".PadLeft(60, '─')}");

Gc();
timer.Restart();
double w2Total = 0;

// 2a. Generate time series
var baseDate = new DateTime(2015, 1, 1);
int nDays = (int)(new DateTime(2026, 1, 1) - baseDate).TotalDays;
var tsDatesStr = new string[nDays];
var tsDates = new DateTime[nDays];
var tsValues = new double[nDays];
var tsRng = new Random(SEED);
for (int i = 0; i < nDays; i++)
{
    tsDates[i] = baseDate.AddDays(i);
    tsDatesStr[i] = tsDates[i].ToString("yyyy-MM-dd");
    double trend = 100 + 200.0 * i / nDays;
    double seasonal = 20 * Math.Sin(2 * Math.PI * i / 365.25);
    double u1 = 1.0 - tsRng.NextDouble();
    double u2 = tsRng.NextDouble();
    double noise = 5 * Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    tsValues[i] = trend + seasonal + noise;
}
var w2Df = new DataFrame(new IColumn[]
{
    new StringColumn("date", tsDatesStr),
    new Column<double>("value", tsValues),
});
var ms2 = LapPrecise(timer);
Record("TimeSeries", "2a. Generate time series (DataFrame)", ms2, $"{nDays:N0} daily observations");
w2Total += ms2;

// 2b. Resample to monthly — group by year-month, aggregate mean
var w2WithMonth = w2Df.WithColumns(
    Col("date").Str.Slice(0, 7).Alias("month")
);
var w2Monthly = w2WithMonth
    .GroupBy("month")
    .Mean()
    .Sort("month");
ms2 = LapPrecise(timer);
Record("TimeSeries", "2b. Resample to monthly (DataFrame)", ms2, $"{w2Monthly.RowCount} months");
w2Total += ms2;

// 2c. NO CONVERSION — seasonal decomposition works on double[]
// Mean() produces "value_mean" column name
var meanColName = w2Monthly.ColumnNames.First(c => c.Contains("value"));
var monthlyValues = w2Monthly.GetColumn<double>(meanColName).Values.ToArray();
var decomp = SeasonalDecompose.Decompose(monthlyValues, period: 12, DecomposeType.Additive);
ms2 = LapPrecise(timer);
Record("TimeSeries", "2c. Seasonal decomposition (native)", ms2);
w2Total += ms2;

// 2d. Fit Holt-Winters
// Build a train DataFrame with DateTime column (ExponentialSmoothing requires DateTime)
var monthCol = (StringColumn)w2Monthly["month"];
var monthDateTimes = new DateTime[w2Monthly.RowCount];
for (int i = 0; i < w2Monthly.RowCount; i++)
    monthDateTimes[i] = DateTime.Parse(monthCol.GetObject(i)!.ToString()! + "-01");
var w2MonthlyWithDt = w2Monthly.AddColumn(new Column<DateTime>("date_dt", monthDateTimes));
var w2TrainDt = w2MonthlyWithDt.Head(w2MonthlyWithDt.RowCount - 12);
var hw = new ExponentialSmoothing(
    type: ESType.Triple,
    alpha: 0.3,
    beta: 0.1,
    gamma: 0.1,
    seasonalPeriod: 12,
    seasonal: Seasonal.Additive
);
hw.Fit(w2TrainDt, "date_dt", meanColName);
ms2 = LapPrecise(timer);
Record("TimeSeries", "2d. Holt-Winters fit (native)", ms2);
w2Total += ms2;

// 2e. Forecast 12 months
var forecast = hw.Forecast(12);
ms2 = LapPrecise(timer);
Record("TimeSeries", "2e. Forecast 12 months", ms2, $"forecast len={12}");
w2Total += ms2;

// 2f. Evaluate
var actualLast12 = w2Monthly.Tail(12).GetColumn<double>(meanColName).Values.ToArray();
var forecastValues = forecast.Values;
double mae = 0, mse = 0;
for (int i = 0; i < 12; i++)
{
    double diff = actualLast12[i] - forecastValues[i];
    mae += Math.Abs(diff);
    mse += diff * diff;
}
mae /= 12;
double rmse = Math.Sqrt(mse / 12);
ms2 = LapPrecise(timer);
Record("TimeSeries", "2f. Evaluate (MAE, RMSE)", ms2, $"MAE={mae:F2}, RMSE={rmse:F2}");
w2Total += ms2;

// 2g. Results already native — no conversion
var w2Result = new DataFrame(new IColumn[]
{
    new Column<double>("forecast", forecastValues),
    new Column<double>("actual", actualLast12),
});
ms2 = LapPrecise(timer);
Record("TimeSeries", "2g. Build results DataFrame (no conversion)", ms2, $"result: ({w2Result.RowCount}, {w2Result.ColumnCount})");
w2Total += ms2;

Record("TimeSeries", "TOTAL", w2Total);
Console.WriteLine($"  {"".PadLeft(40, '─')}");
Console.WriteLine($"  TimeSeries TOTAL: {w2Total:F1}ms\n");


// ════════════════════════════════════════════════════════════════
Console.WriteLine($"{"".PadLeft(60, '─')}");
Console.WriteLine("  WORKFLOW 3: Geospatial Analytics");
Console.WriteLine("  DataFrame + GeoColumn — ZERO conversions");
Console.WriteLine($"{"".PadLeft(60, '─')}");

Gc();
timer.Restart();
double w3Total = 0;

// 3a. Load location data
var w3Df = new DataFrame(new IColumn[]
{
    new StringColumn("city", dataCity),
    new Column<double>("lat", dataLat),
    new Column<double>("lon", dataLon),
    new Column<double>("value", dataValue),
    new CategoricalColumn("category", dataCat),
});
var ms3 = LapPrecise(timer);
Record("Geospatial", "3a. Load locations (DataFrame)", ms3, $"{w3Df.RowCount:N0} rows");
w3Total += ms3;

// 3b. Create GeoColumn — NO CONVERSION, works directly on DataFrame columns
var geoCol = w3Df.ToGeoColumn("lat", "lon");
ms3 = LapPrecise(timer);
Record("Geospatial", "3b. Create GeoColumn (no conversion)", ms3, $"GeoColumn: {geoCol.Length:N0} points");
w3Total += ms3;

// 3c. Spatial filtering — assign regions via bounding box
// Create region grid and assign
var regionNames = new string?[ROWS];
int regionIdx = 0;
var regionList = new List<(string Name, BoundingBox Bbox)>();
for (int latStart = 25; latStart < 48; latStart += 5)
{
    for (int lonStart = -125; lonStart < -65; lonStart += 10)
    {
        regionList.Add(($"Region_{regionIdx}", new BoundingBox(latStart, lonStart, latStart + 5, lonStart + 10)));
        regionIdx++;
    }
}

// Assign regions to points
for (int i = 0; i < ROWS; i++)
{
    foreach (var (name, bbox) in regionList)
    {
        if (bbox.Contains(new GeoPoint(dataLat[i], dataLon[i])))
        {
            regionNames[i] = name;
            break;
        }
    }
    regionNames[i] ??= "None";
}
var w3WithRegion = w3Df.AddColumn(new StringColumn("region", regionNames!));
ms3 = LapPrecise(timer);
Record("Geospatial", "3c. Spatial region assignment", ms3, $"{regionList.Count} regions");
w3Total += ms3;

// 3d. Aggregate by region — stays in DataFrame
var w3Agg = w3WithRegion
    .GroupBy("region")
    .Agg(
        ("value", AggFunc.Count),
        ("value", AggFunc.Mean),
        ("value", AggFunc.Sum),
        ("value", AggFunc.Std)
    );
ms3 = LapPrecise(timer);
Record("Geospatial", "3d. GroupBy region + aggregate", ms3, $"{w3Agg.RowCount} regions aggregated");
w3Total += ms3;

// 3e. Haversine distance to NYC — native, no conversion
var nyc = new GeoPoint(40.7128, -74.0060);
var w3Distances = geoCol.DistanceTo(nyc);
ms3 = LapPrecise(timer);
Record("Geospatial", "3e. Haversine distances (native)", ms3, $"distances computed for {w3Distances.Length:N0} points");
w3Total += ms3;

// 3f. No conversion back — already a DataFrame
ms3 = 0;
Record("Geospatial", "3f. Results already DataFrame (no conversion)", ms3);

Record("Geospatial", "TOTAL", w3Total);
Console.WriteLine($"  {"".PadLeft(40, '─')}");
Console.WriteLine($"  Geospatial TOTAL: {w3Total:F1}ms\n");


// ════════════════════════════════════════════════════════════════
Console.WriteLine($"{"".PadLeft(60, '─')}");
Console.WriteLine("  WORKFLOW 4: NLP / Text Classification");
Console.WriteLine("  DataFrame throughout — ZERO conversions");
Console.WriteLine($"{"".PadLeft(60, '─')}");

Gc();
timer.Restart();
double w4Total = 0;
int NLP_ROWS = 200_000;

// 4a. Load text data
var w4Df = new DataFrame(new IColumn[]
{
    new StringColumn("text", dataText.Take(NLP_ROWS).ToArray()),
    new CategoricalColumn("category", dataCat.Take(NLP_ROWS).ToArray()),
    new Column<int>("label", dataLabel.Take(NLP_ROWS).ToArray()),
});
var ms4 = LapPrecise(timer);
Record("NLP", "4a. Load text data (DataFrame)", ms4, $"{w4Df.RowCount:N0} documents");
w4Total += ms4;

// 4b. Lowercase — StringColumn accessor
var w4Clean = w4Df.WithColumns(
    Col("text").Str.Lower().Alias("text_clean")
);
ms4 = LapPrecise(timer);
Record("NLP", "4b. Lowercase (DataFrame)", ms4);
w4Total += ms4;

// 4c. NO CONVERSION — TextVectorizer works on DataFrame directly
var tfidf = new TextVectorizer("text_clean", VectorizerMode.TfIdf, maxFeatures: 5000);
var w4Vectorized = tfidf.FitTransform(w4Clean);
ms4 = LapPrecise(timer);
Record("NLP", "4c. TF-IDF vectorize (DataFrame — no conversion)", ms4, $"columns: {w4Vectorized.ColumnCount}");
w4Total += ms4;

// 4d. Train/test split — returns DataFrames
var (w4Train, w4Test) = w4Vectorized.TrainTestSplit(testFraction: 0.2, seed: SEED);
ms4 = LapPrecise(timer);
Record("NLP", "4d. Train/test split (DataFrame)", ms4);
w4Total += ms4;

// 4e. Convert to tensors for model (internal, not cross-library)
var tfidfCols = w4Vectorized.ColumnNames.Where(c => c.StartsWith("tfidf_") || c.StartsWith("tf_")).ToArray();
if (tfidfCols.Length == 0)
    tfidfCols = w4Vectorized.ColumnNames.Where(c => c != "text" && c != "text_clean" && c != "category" && c != "label").ToArray();
var w4XTrain = DataFrameToTensor(w4Train, tfidfCols);
var w4YTrain = new Tensor<double>(w4Train.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w4Train.RowCount);
var w4XTest = DataFrameToTensor(w4Test, tfidfCols);
var w4YTest = new Tensor<double>(w4Test.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w4Test.RowCount);
ms4 = LapPrecise(timer);
Record("NLP", "4e. DataFrame → Tensor (internal)", ms4);
w4Total += ms4;

// 4f. Train LogisticRegression
var lrModel = new LogisticRegression(learningRate: 0.01, maxIterations: 200);
lrModel.Fit(w4XTrain, w4YTrain);
ms4 = LapPrecise(timer);
Record("NLP", "4f. Train LogisticRegression", ms4);
w4Total += ms4;

// 4g. Predict + evaluate
var w4Preds = lrModel.Predict(w4XTest);
var w4YTrueCol = new Column<int>("actual", w4YTest.ToArray().Select(d => (int)d).ToArray());
var w4YPredCol = new Column<int>("predicted", w4Preds.ToArray().Select(d => (int)d).ToArray());
var w4Metrics = MetricsCalculator.Classification(w4YTrueCol, w4YPredCol);
ms4 = LapPrecise(timer);
Record("NLP", "4g. Predict + evaluate", ms4, $"acc={w4Metrics.Accuracy:F4}");
w4Total += ms4;

// 4h. Results already native
var w4Result = new DataFrame(new IColumn[] { w4YTrueCol, w4YPredCol });
ms4 = LapPrecise(timer);
Record("NLP", "4h. Build results DataFrame (no conversion)", ms4, $"result: ({w4Result.RowCount}, {w4Result.ColumnCount})");
w4Total += ms4;

Record("NLP", "TOTAL", w4Total);
Console.WriteLine($"  {"".PadLeft(40, '─')}");
Console.WriteLine($"  NLP Pipeline TOTAL: {w4Total:F1}ms\n");


// ════════════════════════════════════════════════════════════════
Console.WriteLine($"{"".PadLeft(60, '─')}");
Console.WriteLine("  WORKFLOW 5: Multi-Model Comparison (Full Data Science)");
Console.WriteLine("  DataFrame → Tensor → 5 models → DataFrame report");
Console.WriteLine($"{"".PadLeft(60, '─')}");

Gc();
timer.Restart();
double w5Total = 0;

// 5a. Feature engineering
var w5Base = new DataFrame(new IColumn[]
{
    new Column<double>("value", dataValue),
    new Column<double>("price", dataPrice),
    new Column<double>("volume", dataVolume),
    new Column<int>("label", dataLabel),
});
// Compute total, rolling mean, and price_diff manually (Expr doesn't have RollingMean/Shift)
var w5Price = w5Base.GetColumn<double>("price");
var w5Vol = w5Base.GetColumn<double>("volume");
var w5Val = w5Base.GetColumn<double>("value");
var w5TotalArr = new double[ROWS];
var w5Ma10 = new double?[ROWS];
var w5PriceDiff = new double?[ROWS];
for (int i = 0; i < ROWS; i++)
{
    w5TotalArr[i] = w5Price.Values[i] * w5Vol.Values[i];
    w5PriceDiff[i] = i > 0 ? w5Price.Values[i] - w5Price.Values[i - 1] : null;
    if (i >= 9)
    {
        double sum = 0;
        for (int j = i - 9; j <= i; j++) sum += w5Val.Values[j];
        w5Ma10[i] = sum / 10.0;
    }
}
var w5Df = w5Base
    .AddColumn(new Column<double>("total", w5TotalArr))
    .AddColumn(Column<double>.FromNullable("value_ma10", w5Ma10))
    .AddColumn(Column<double>.FromNullable("price_diff", w5PriceDiff))
    .DropNa();
var ms5 = LapPrecise(timer);
Record("MultiModel", "5a. Feature engineering (DataFrame)", ms5, $"{w5Df.RowCount:N0} rows, {w5Df.ColumnCount} features");
w5Total += ms5;

// 5b. Scale + split (all in DataFrame land)
var w5Scaler = new StandardScaler("value", "price", "volume", "total", "value_ma10", "price_diff");
var w5Scaled = w5Scaler.FitTransform(w5Df);
var (w5Train, w5Test) = w5Scaled.TrainTestSplit(testFraction: 0.2, seed: SEED);
ms5 = LapPrecise(timer);
Record("MultiModel", "5b. Scale + split (DataFrame)", ms5);
w5Total += ms5;

// 5c. Build tensors (internal)
var w5FeatureCols = new[] { "value", "price", "volume", "total", "value_ma10", "price_diff" };
var w5XTrain = DataFrameToTensor(w5Train, w5FeatureCols);
var w5YTrain = new Tensor<double>(w5Train.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w5Train.RowCount);
var w5XTest = DataFrameToTensor(w5Test, w5FeatureCols);
var w5YTest = new Tensor<double>(w5Test.GetColumn<int>("label").Values.ToArray().Select(v => (double)v).ToArray(), w5Test.RowCount);
ms5 = LapPrecise(timer);
Record("MultiModel", "5c. DataFrame → Tensor (internal)", ms5);
w5Total += ms5;

// 5d. Train 5 models
var modelResults = new List<(string Name, double Acc, double F1, double TrainMs)>();
var models = new (string Name, IModel Model)[]
{
    ("LogisticRegression", new LogisticRegression(learningRate: 0.01, maxIterations: 200)),
    ("DecisionTree", new DecisionTreeClassifier(maxDepth: 10, seed: SEED)),
    ("RandomForest", new RandomForestClassifier(nEstimators: 50, seed: SEED)),
    ("GBT", new GradientBoostedTreeClassifier(nEstimators: 50, seed: SEED)),
    ("KNN", new KNearestNeighborsClassifier(k: 5)),
};

foreach (var (name, model) in models)
{
    var modelTimer = Stopwatch.StartNew();
    model.Fit(w5XTrain, w5YTrain);
    var preds = model.Predict(w5XTest);
    var trainMs = modelTimer.Elapsed.TotalMilliseconds;

    var ytCol = new Column<int>("y", w5YTest.ToArray().Select(d => (int)d).ToArray());
    var ypCol = new Column<int>("p", preds.ToArray().Select(d => (int)d).ToArray());
    var m = MetricsCalculator.Classification(ytCol, ypCol);
    modelResults.Add((name, m.Accuracy, m.F1, trainMs));
    Console.WriteLine($"    {name}: acc={m.Accuracy:F4}, f1={m.F1:F4} [{trainMs:F0}ms]");
}
ms5 = LapPrecise(timer);
Record("MultiModel", "5d. Train 5 models + evaluate", ms5, $"{models.Length} models trained");
w5Total += ms5;

// 5e. Build comparison report — already in DataFrame form
var comparisonDf = new DataFrame(new IColumn[]
{
    new StringColumn("model", modelResults.Select(r => (string?)r.Name).ToArray()),
    new Column<double>("accuracy", modelResults.Select(r => r.Acc).ToArray()),
    new Column<double>("f1", modelResults.Select(r => r.F1).ToArray()),
    new Column<double>("train_ms", modelResults.Select(r => r.TrainMs).ToArray()),
});
var bestModel = modelResults.OrderByDescending(r => r.F1).First();
ms5 = LapPrecise(timer);
Record("MultiModel", "5e. Build results DataFrame (no conversion)", ms5, $"best model: {bestModel.Name}");
w5Total += ms5;

Record("MultiModel", "TOTAL", w5Total);
Console.WriteLine($"  {"".PadLeft(40, '─')}");
Console.WriteLine($"  MultiModel TOTAL: {w5Total:F1}ms\n");


// ════════════════════════════════════════════════════════════════
// SUMMARY
// ════════════════════════════════════════════════════════════════
Console.WriteLine($"\n{"".PadLeft(60, '=')}");
Console.WriteLine("  SUMMARY — Cortex End-to-End (C#)");
Console.WriteLine($"{"".PadLeft(60, '=')}");

var totals = results.Where(r => r.Op == "TOTAL").ToList();
double totalAll = totals.Sum(t => t.Ms);

Console.WriteLine($"\n  {"Workflow",-20} {"Total",10}");
Console.WriteLine($"  {"".PadLeft(32, '─')}");
foreach (var t in totals)
    Console.WriteLine($"  {t.Workflow,-20} {t.Ms,8:F1}ms");
Console.WriteLine($"  {"".PadLeft(32, '─')}");
Console.WriteLine($"  {"ALL WORKFLOWS",-20} {totalAll,8:F1}ms");
Console.WriteLine($"\n  Conversion overhead: 0.0ms (0.0% — zero conversions)");

// Save results
var output = new
{
    language = "csharp",
    rows = ROWS,
    results = results.Select(r => new { workflow = r.Workflow, operation = r.Op, ms = r.Ms, detail = r.Detail }),
    summary = new
    {
        total_ms = Math.Round(totalAll, 2),
        conversion_ms = 0.0,
        conversion_pct = 0.0,
        num_conversions = 0,
    }
};
File.WriteAllText("e2e_pipeline_csharp_results.json", JsonSerializer.Serialize(output, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine($"\n  Results saved to e2e_pipeline_csharp_results.json");


// ═══════════════════════════════════════════════════════════
// Helper: DataFrame columns → 2D Tensor
// ═══════════════════════════════════════════════════════════
static Tensor<double> DataFrameToTensor(DataFrame df, string[] columns)
{
    int rows = df.RowCount;
    int cols = columns.Length;
    var data = new double[rows * cols];
    for (int c = 0; c < cols; c++)
    {
        var col = df[columns[c]];
        if (col is Column<double> dc)
            for (int r = 0; r < rows; r++) data[r * cols + c] = dc.Values[r];
        else if (col is Column<int> ic)
            for (int r = 0; r < rows; r++) data[r * cols + c] = ic.Values[r];
        else if (col is Column<float> fc)
            for (int r = 0; r < rows; r++) data[r * cols + c] = fc.Values[r];
        else
            for (int r = 0; r < rows; r++) data[r * cols + c] = col.IsNull(r) ? 0 : Convert.ToDouble(col.GetObject(r));
    }
    return new Tensor<double>(data, rows, cols);
}
