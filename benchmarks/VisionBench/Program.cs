using System.Diagnostics;
using System.Text.Json;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;
using Cortex.Vision;
using Cortex.Vision.Transforms;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

const string IMG_DIR = "vision_bench_images";
const int N_IMAGES = 500;
const int IMG_SIZE = 256;

var results = new List<(string Cat, string Op, long Ms)>();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }
var timer = Stopwatch.StartNew();

Console.WriteLine("=== Cortex.Vision Benchmark v2 ===\n");

var imagePaths = Directory.GetFiles(IMG_DIR, "*.png").OrderBy(f => f).Take(N_IMAGES).ToArray();

// ═══════════════════════════════════════════════════════
// 1. IMAGE LOADING
// ═══════════════════════════════════════════════════════
Console.WriteLine("── Image Loading ──");

timer.Restart();
var images = new ImageTensor[N_IMAGES];
Parallel.For(0, N_IMAGES, i => images[i] = ImageIO.Load(imagePaths[i]));
var ms = Lap(timer);
results.Add(("Load", $"Load {N_IMAGES} images (ImageSharp)", ms));
Console.WriteLine($"  {"Load " + N_IMAGES + " images (ImageSharp)",-55} {ms,6:N0} ms");

timer.Restart();
var floatArrays = new float[N_IMAGES][];
Parallel.For(0, N_IMAGES, i => floatArrays[i] = images[i].Span.ToArray());
ms = Lap(timer);
results.Add(("Load", $"Convert to float32 ({N_IMAGES})", ms));
Console.WriteLine($"  {"Convert to float32 (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

timer.Restart();
var batch = ImageIO.Load(imagePaths, IMG_SIZE, IMG_SIZE);
ms = Lap(timer);
results.Add(("Load", $"Stack to batch ({N_IMAGES}x{IMG_SIZE}x{IMG_SIZE}x3)", ms));
Console.WriteLine($"  {"Stack to batch (" + N_IMAGES + "x" + IMG_SIZE + "x" + IMG_SIZE + "x3)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 2. INDIVIDUAL TRANSFORMS (parallel across images)
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Individual Transforms ──");

void Bench(string name, IImageTransformer t)
{
    timer.Restart();
    Parallel.For(0, N_IMAGES, i => t.Transform(images[i]));
    ms = Lap(timer);
    results.Add(("Transform", $"{name} ({N_IMAGES})", ms));
    Console.WriteLine($"  {name + " (" + N_IMAGES + ")",-55} {ms,6:N0} ms");
}

var normalize = Normalize.ImageNet();
var hflip = new RandomHorizontalFlip(1.0, seed: 42);
var vflip = new RandomVerticalFlip(1.0, seed: 42);

Bench("Resize 224x224", new Resize(224, 224));
Bench("CenterCrop 128", new CenterCrop(128, 128));
Bench("RandomCrop 128 pad=4", new RandomCrop(128, 128, padding: 4, seed: 42));
Bench("HorizontalFlip", hflip);
Bench("VerticalFlip", vflip);
Bench("Normalize ImageNet", normalize);
Bench("Grayscale", new Grayscale());
Bench("ColorJitter", new ColorJitter(0.2f, 0.2f, 0.2f, seed: 42));
Bench("GaussianBlur", new GaussianBlur(1.0f));
Bench("RandomRotation ±10°", new RandomRotation(10f, seed: 42));
Bench("RandomErasing", new RandomErasing(1.0, seed: 42));

// ═══════════════════════════════════════════════════════
// 3. BATCH TRANSFORMS
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Batch Transforms ──");

timer.Restart(); normalize.Transform(batch);
ms = Lap(timer); results.Add(("Batch", $"Batch normalize ({N_IMAGES})", ms));
Console.WriteLine($"  {"Batch normalize (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

timer.Restart(); hflip.Transform(batch);
ms = Lap(timer); results.Add(("Batch", $"Batch HFlip ({N_IMAGES})", ms));
Console.WriteLine($"  {"Batch HFlip (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

timer.Restart(); new Grayscale().Transform(batch);
ms = Lap(timer); results.Add(("Batch", $"Batch grayscale ({N_IMAGES})", ms));
Console.WriteLine($"  {"Batch grayscale (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

timer.Restart(); vflip.Transform(batch);
ms = Lap(timer); results.Add(("Batch", $"Batch VFlip ({N_IMAGES})", ms));
Console.WriteLine($"  {"Batch VFlip (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 4. PIPELINES
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Pipelines ──");

var p1 = ImagePipeline.Create().Resize(224, 224).RandomHorizontalFlip(0.5).Normalize(Normalize.ImageNet()).Build();
timer.Restart(); Parallel.For(0, N_IMAGES, i => p1.Transform(images[i]));
ms = Lap(timer); results.Add(("Pipeline", $"Resize+Flip+Normalize ({N_IMAGES})", ms));
Console.WriteLine($"  {"Resize+Flip+Normalize (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

var p2 = ImagePipeline.Create().Resize(224, 224).RandomHorizontalFlip(0.5)
    .Add(new ColorJitter(0.2f, 0.2f, 0.2f, seed: 42)).Normalize(Normalize.ImageNet())
    .Add(new RandomErasing(0.5, seed: 42)).Build();
timer.Restart(); Parallel.For(0, N_IMAGES, i => p2.Transform(images[i]));
ms = Lap(timer); results.Add(("Pipeline", $"Full augmentation ({N_IMAGES})", ms));
Console.WriteLine($"  {"Full augmentation (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

var p3 = ImagePipeline.Create().Resize(224, 224).CenterCrop(224, 224).Normalize(Normalize.ImageNet()).Build();
timer.Restart(); Parallel.For(0, N_IMAGES, i => p3.Transform(images[i]));
ms = Lap(timer); results.Add(("Pipeline", $"Inference pipeline ({N_IMAGES})", ms));
Console.WriteLine($"  {"Inference pipeline (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 5. DATA LOADER SIMULATION
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── DataLoader Simulation ──");

// Build a DataFrame with paths and fake labels
var labels = new string?[N_IMAGES];
for (int i = 0; i < N_IMAGES; i++) labels[i] = (i % 10).ToString();
var df = new DataFrame(
    new StringColumn("path", imagePaths.Select(p => (string?)p).ToArray()),
    new StringColumn("label", labels));

timer.Restart();
var loader = df.ToImageDataLoader("path", "label", batchSize: 32, shuffle: true, seed: 42,
    augmentation: p1, resizeWidth: 224, resizeHeight: 224);
foreach (var (imgs, lbls) in loader) { /* consume batches */ }
ms = Lap(timer); results.Add(("DataLoader", $"Load+augment batches of 32 ({N_IMAGES})", ms));
Console.WriteLine($"  {"Load+augment batches of 32 (" + N_IMAGES + ")",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 6. IMAGE STATISTICS
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Image Statistics ──");

timer.Restart();
var (chMean, chStd) = ImageStats.ComputeNormalization(df, "path", sampleSize: N_IMAGES, seed: 42);
ms = Lap(timer); results.Add(("Stats", $"Compute dataset mean/std ({N_IMAGES} images)", ms));
Console.WriteLine($"  {"Compute dataset mean/std (" + N_IMAGES + " images)",-55} {ms,6:N0} ms");
Console.WriteLine($"    mean=[{chMean[0]:F4}, {chMean[1]:F4}, {chMean[2]:F4}] std=[{chStd[0]:F4}, {chStd[1]:F4}, {chStd[2]:F4}]");

// ═══════════════════════════════════════════════════════
// 7. IMAGE SAVING
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Image Saving ──");

Directory.CreateDirectory("vision_bench_output_cs");
timer.Restart();
Parallel.For(0, 100, i => ImageIO.Save(images[i], $"vision_bench_output_cs/out_{i:D4}.png"));
ms = Lap(timer); results.Add(("Save", "Save 100 PNG", ms));
Console.WriteLine($"  {"Save 100 PNG",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// SUMMARY + COMPARISON
// ═══════════════════════════════════════════════════════
Console.WriteLine($"\n{"═",0}{"",69}");
var cats = new Dictionary<string, long>();
foreach (var (cat, _, m) in results) cats[cat] = cats.GetValueOrDefault(cat) + m;
foreach (var (cat, m) in cats.OrderByDescending(kv => kv.Value))
    Console.WriteLine($"  {cat,-30} {m,8:N0} ms");
var total = results.Sum(r => r.Ms);
Console.WriteLine($"  {"TOTAL",-30} {total,8:N0} ms");

// Load Python v2 results
var pyFile = "vision_bench_output/python_vision_v2_results.json";
if (File.Exists(pyFile))
{
    Console.WriteLine($"\n{"═",0}{"",69}");
    Console.WriteLine("  HEAD-TO-HEAD COMPARISON");
    Console.WriteLine($"{"═",0}{"",69}");

    var pyResults = JsonDocument.Parse(File.ReadAllText(pyFile));
    var pyCats = new Dictionary<string, int>();
    foreach (var op in pyResults.RootElement.EnumerateArray())
        pyCats[op.GetProperty("category").GetString()!] = pyCats.GetValueOrDefault(op.GetProperty("category").GetString()!) + op.GetProperty("ms").GetInt32();

    int csWins = 0, pyWins = 0;
    foreach (var (cat, csMs) in cats.OrderByDescending(kv => kv.Value))
    {
        int pyMs = pyCats.GetValueOrDefault(cat);
        if (pyMs > 0 && csMs > 0)
        {
            string winner = csMs <= pyMs ? $"PS {(double)pyMs/csMs:F1}x" : $"Py {(double)csMs/pyMs:F1}x";
            if (csMs <= pyMs) csWins++; else pyWins++;
            Console.WriteLine($"  {cat,-20} PS: {csMs,6}ms  Py: {pyMs,6}ms  → {winner}");
        }
    }
    Console.WriteLine($"\n  Cortex wins: {csWins}  Python wins: {pyWins}");
    Console.WriteLine($"  Total: PS {total}ms vs Py {pyCats.Values.Sum()}ms → {(double)pyCats.Values.Sum()/total:F1}x faster");
}
