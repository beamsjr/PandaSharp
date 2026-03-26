using PandaSharp;
using PandaSharp.Column;
using PandaSharp.TimeSeries.Models;
using PandaSharp.TimeSeries.Decomposition;
using PandaSharp.TimeSeries.Diagnostics;
using System.Diagnostics;
using System.Text.Json;

var results = new List<(string Cat, string Op, long Ms)>();
var timer = Stopwatch.StartNew();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }

// ── Data Generation: trend + seasonal + noise ──
var rng = new Random(42);
int N = 5000;
var series = new double[N];
for (int i = 0; i < N; i++)
{
    double trend = 10.0 + (50.0 - 10.0) * i / (N - 1);
    double seasonal = 5.0 * Math.Sin(2.0 * Math.PI * i / 12.0);
    double noise = NextGaussian(rng) * 0.5;
    series[i] = trend + seasonal + noise;
}

var dates = Enumerable.Range(0, N).Select(i => DateTime.Today.AddDays(i)).ToArray();
var df = new DataFrame(
    new Column<DateTime>("Date", dates),
    new Column<double>("Value", series));

// Smaller subsets for expensive operations
var dates1K = dates.Take(1000).ToArray();
var series1K = series.Take(1000).ToArray();
var df1K = new DataFrame(
    new Column<DateTime>("Date", dates1K),
    new Column<double>("Value", series1K));

Console.WriteLine($"=== PandaSharp.TimeSeries Benchmark ({N} points) ===\n");

void Record(string cat, string name, long ms)
{
    results.Add((cat, name, ms));
    Console.WriteLine($"  {name,-55} {ms,6:N0} ms");
}

// ═══════════════════════════════════════════════════════
// Forecasting
// ═══════════════════════════════════════════════════════
Console.WriteLine("── Forecasting ──");

timer.Restart();
new ARIMA(p: 2, d: 1, q: 1).Fit(df, "Date", "Value").Forecast(50);
Record("Forecast", "ARIMA(2,1,1) fit+forecast(50)", Lap(timer));

timer.Restart();
new ExponentialSmoothing(
    type: ESType.Triple,
    seasonalPeriod: 12,
    seasonal: Seasonal.Additive
).Fit(df, "Date", "Value").Forecast(50);
Record("Forecast", "Holt-Winters fit+forecast(50)", Lap(timer));

timer.Restart();
new AutoARIMA(maxP: 3, maxQ: 3, maxD: 2).Fit(df1K, "Date", "Value");
Record("Forecast", "AutoARIMA (1K points)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Decomposition
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Decomposition ──");

timer.Restart();
SeasonalDecompose.Decompose(series, period: 12, type: DecomposeType.Additive);
Record("Decompose", "Seasonal decompose (additive)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Diagnostics
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Diagnostics ──");

timer.Restart();
StationarityTests.AugmentedDickeyFuller(series);
Record("Diagnostic", "ADF test", Lap(timer));

timer.Restart();
StationarityTests.KPSS(series, regressionType: "c");
Record("Diagnostic", "KPSS test", Lap(timer));

timer.Restart();
AutocorrelationTests.ACF(series, maxLags: 50);
Record("Diagnostic", "ACF (50 lags)", Lap(timer));

timer.Restart();
AutocorrelationTests.PACF(series, maxLags: 50);
Record("Diagnostic", "PACF (50 lags)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Summary
// ═══════════════════════════════════════════════════════
Console.WriteLine($"\n{"",0}{new string('═', 70)}");
var cats = new Dictionary<string, long>();
foreach (var r in results)
    cats[r.Cat] = cats.GetValueOrDefault(r.Cat) + r.Ms;
foreach (var (c, m) in cats.OrderByDescending(kv => kv.Value))
    Console.WriteLine($"  {c,-30} {m,8:N0} ms");
long total = results.Sum(r => r.Ms);
Console.WriteLine($"  {"TOTAL",-30} {total,8:N0} ms");

// ── Save and Compare with Python ──
var jsonResults = results.Select(r => new { category = r.Cat, op = r.Op, ms = r.Ms }).ToArray();
Directory.CreateDirectory("ts_bench_output");
File.WriteAllText("ts_bench_output/csharp_ts_results.json",
    JsonSerializer.Serialize(jsonResults, new JsonSerializerOptions { WriteIndented = true }));

string pyPath = "ts_bench_output/python_ts_results.json";
if (File.Exists(pyPath))
{
    Console.WriteLine($"\n{"",0}{new string('═', 70)}");
    Console.WriteLine("  Python vs C# Comparison");
    Console.WriteLine(string.Format("  {0,-45} {1,8} {2,8} {3,8}", "Operation", "Python", "C#", "Ratio"));
    Console.WriteLine($"  {new string('\u2500', 73)}");

    using var doc = JsonDocument.Parse(File.ReadAllText(pyPath));
    var pyResults = doc.RootElement.EnumerateArray()
        .Select(e => (op: e.GetProperty("op").GetString()!, ms: e.GetProperty("ms").GetInt64()))
        .ToList();

    foreach (var py in pyResults)
    {
        var cs = results.FirstOrDefault(r => r.Op == py.op);
        if (cs != default)
        {
            double ratio = cs.Ms == 0 ? 0 : (double)py.ms / cs.Ms;
            Console.WriteLine(string.Format("  {0,-45} {1,8:N0} {2,8:N0} {3,7:F2}x", py.op, py.ms, cs.Ms, ratio));
        }
        else
        {
            Console.WriteLine(string.Format("  {0,-45} {1,8:N0} {2,>8} {3,>8}", py.op, py.ms, "N/A", ""));
        }
    }

    long pyTotal = pyResults.Sum(r => r.ms);
    double totalRatio = total == 0 ? 0 : (double)pyTotal / total;
    Console.WriteLine($"  {new string('\u2500', 73)}");
    Console.WriteLine(string.Format("  {0,-45} {1,8:N0} {2,8:N0} {3,7:F2}x", "TOTAL", pyTotal, total, totalRatio));
}
else
{
    Console.WriteLine($"\n  (No Python results found at {pyPath} — run timeseries_python.py first)");
}

static double NextGaussian(Random rng)
{
    double u1 = 1.0 - rng.NextDouble();
    double u2 = rng.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
}
