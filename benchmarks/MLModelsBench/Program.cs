using PandaSharp.ML.Tensors;
using PandaSharp.ML.Models;
using PandaSharp.ML.ModelSelection;
using System.Diagnostics;
using System.Text.Json;

var results = new List<(string Cat, string Op, long Ms)>();
var timer = Stopwatch.StartNew();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }

// ── Data Generation ──
var rng = new Random(42);
int N = 50000, P = 20;

var xData = new double[N * P];
for (int i = 0; i < xData.Length; i++) xData[i] = NextGaussian(rng);
var X_reg = new Tensor<double>(xData, N, P);

// y_reg = X @ weights + noise
var weights = new double[P];
for (int i = 0; i < P; i++) weights[i] = NextGaussian(rng);
var yRegData = new double[N];
for (int r = 0; r < N; r++)
{
    double sum = 0;
    for (int c = 0; c < P; c++) sum += xData[r * P + c] * weights[c];
    yRegData[r] = sum + NextGaussian(rng) * 0.1;
}
var y_reg = new Tensor<double>(yRegData, N);

// Classification: y = (x0 + x1 > 0) ? 1 : 0
var yClsData = new double[N];
for (int r = 0; r < N; r++)
    yClsData[r] = (xData[r * P] + xData[r * P + 1]) > 0 ? 1.0 : 0.0;
var y_cls = new Tensor<double>(yClsData, N);
var X_cls = X_reg; // same features

Console.WriteLine("=== PandaSharp.ML Models Benchmark ===");
Console.WriteLine($"  Data: {N} samples, {P} features\n");

void Record(string cat, string name, long ms)
{
    results.Add((cat, name, ms));
    Console.WriteLine($"  {name,-55} {ms,6:N0} ms");
}

// ═══════════════════════════════════════════════════════
// Linear Models
// ═══════════════════════════════════════════════════════
Console.WriteLine("── Linear Models ──");

timer.Restart();
new LinearRegression().Fit(X_reg, y_reg).Predict(X_reg);
Record("Linear", "LinearRegression fit+predict", Lap(timer));

timer.Restart();
new ElasticNet(alpha: 0.1).Fit(X_reg, y_reg).Predict(X_reg);
Record("Linear", "ElasticNet fit+predict", Lap(timer));

timer.Restart();
new Lasso(alpha: 0.1).Fit(X_reg, y_reg).Predict(X_reg);
Record("Linear", "Lasso fit+predict", Lap(timer));

timer.Restart();
new LogisticRegression(maxIterations: 200).Fit(X_cls, y_cls).Predict(X_cls);
Record("Linear", "LogisticRegression fit+predict", Lap(timer));

timer.Restart();
new SGDClassifier(maxEpochs: 100).Fit(X_cls, y_cls).Predict(X_cls);
Record("Linear", "SGDClassifier fit+predict", Lap(timer));

timer.Restart();
new SGDRegressor(maxEpochs: 100).Fit(X_reg, y_reg).Predict(X_reg);
Record("Linear", "SGDRegressor fit+predict", Lap(timer));

// ═══════════════════════════════════════════════════════
// Tree Models
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Tree Models ──");

timer.Restart();
new DecisionTreeClassifier(maxDepth: 10).Fit(X_cls, y_cls).Predict(X_cls);
Record("Tree", "DecisionTreeClassifier fit+predict", Lap(timer));

timer.Restart();
new DecisionTreeRegressor(maxDepth: 10).Fit(X_reg, y_reg).Predict(X_reg);
Record("Tree", "DecisionTreeRegressor fit+predict", Lap(timer));

timer.Restart();
new RandomForestClassifier(nEstimators: 100, maxDepth: 10).Fit(X_cls, y_cls).Predict(X_cls);
Record("Tree", "RandomForestClassifier(100) fit+predict", Lap(timer));

timer.Restart();
new RandomForestRegressor(nEstimators: 100, maxDepth: 10).Fit(X_reg, y_reg).Predict(X_reg);
Record("Tree", "RandomForestRegressor(100) fit+predict", Lap(timer));

timer.Restart();
new GradientBoostedTreeClassifier(nEstimators: 100, maxDepth: 3).Fit(X_cls, y_cls).Predict(X_cls);
Record("Tree", "GBTClassifier(100) fit+predict", Lap(timer));

timer.Restart();
new GradientBoostedTreeRegressor(nEstimators: 100, maxDepth: 3).Fit(X_reg, y_reg).Predict(X_reg);
Record("Tree", "GBTRegressor(100) fit+predict", Lap(timer));

// ═══════════════════════════════════════════════════════
// Distance Models
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Distance Models ──");

var X_knn = X_cls.Slice(0, 0, 5000);
var y_knn = y_cls.Slice(0, 0, 5000);

timer.Restart();
new KNearestNeighborsClassifier(k: 5).Fit(X_knn, y_knn).Predict(X_knn);
Record("Distance", "KNN Classifier(k=5) fit+predict (5K)", Lap(timer));

timer.Restart();
new KNearestNeighborsRegressor(k: 5).Fit(X_knn, y_knn).Predict(X_knn);
Record("Distance", "KNN Regressor(k=5) fit+predict (5K)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Clustering
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Clustering ──");

var X_cluster = X_reg.Slice(0, 0, 10000);

timer.Restart();
new KMeans(nClusters: 5, nInit: 10, maxIter: 300).Fit(X_cluster, Tensor<double>.Zeros(10000));
Record("Cluster", "KMeans(5) fit (10K)", Lap(timer));

timer.Restart();
new DBSCAN(eps: 3.0, minSamples: 5).Fit(X_reg.Slice(0, 0, 5000));
Record("Cluster", "DBSCAN fit (5K)", Lap(timer));

timer.Restart();
new AgglomerativeClustering(nClusters: 5).Fit(X_reg.Slice(0, 0, 2000));
Record("Cluster", "AgglomerativeClustering fit (2K)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Dimensionality Reduction
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Dimensionality Reduction ──");

timer.Restart();
new PCA(nComponents: 5).FitTransform(X_reg);
Record("DimReduce", "PCA(5) fit+transform (50K)", Lap(timer));

timer.Restart();
new TruncatedSVD(nComponents: 5).FitTransform(X_reg);
Record("DimReduce", "TruncatedSVD(5) fit+transform (50K)", Lap(timer));

timer.Restart();
new TSNE(nComponents: 2, perplexity: 30, maxIterations: 250).FitTransform(X_reg.Slice(0, 0, 1000));
Record("DimReduce", "t-SNE(2) fit+transform (1K)", Lap(timer));

// ═══════════════════════════════════════════════════════
// Model Selection
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Model Selection ──");

var X_cv = X_reg.Slice(0, 0, 5000);
var y_cv = y_reg.Slice(0, 0, 5000);

timer.Restart();
CrossValidation.CrossValScore(new LinearRegression(), X_cv, y_cv, nFolds: 5);
Record("ModelSel", "CrossValScore(5-fold, 5K)", Lap(timer));

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

// ── Comparison with Python ──
var jsonResults = results.Select(r => new { category = r.Cat, op = r.Op, ms = r.Ms }).ToArray();
Directory.CreateDirectory("ml_bench_output");
File.WriteAllText("ml_bench_output/csharp_ml_results.json",
    JsonSerializer.Serialize(jsonResults, new JsonSerializerOptions { WriteIndented = true }));

string pyPath = "ml_bench_output/python_ml_results.json";
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

    // Match by operation name
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
    Console.WriteLine($"\n  (No Python results found at {pyPath} — run ml_models_python.py first)");
}

static double NextGaussian(Random rng)
{
    double u1 = 1.0 - rng.NextDouble();
    double u2 = rng.NextDouble();
    return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
}
