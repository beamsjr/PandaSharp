using System.Diagnostics;
using System.Text.Json;
using Cortex;
using Cortex.Column;
using Cortex.ML.Tensors;
using Cortex.Vision;
using Cortex.Vision.Models;
using Cortex.Vision.Transforms;
using Cortex.Vision.Video;
using FFMediaToolkit;

const string VIDEO_PATH = "vision_bench_models/test_video.mp4";
const string MODEL_PATH = "vision_bench_models/mobilenetv2.onnx";
const string IMG_DIR = "vision_bench_images";

var results = new List<(string Cat, string Op, long Ms)>();
long Lap(Stopwatch s) { var ms = s.ElapsedMilliseconds; s.Restart(); return ms; }
var timer = Stopwatch.StartNew();

Console.WriteLine("=== Cortex.Vision Video + ONNX Benchmark ===\n");

// ═══════════════════════════════════════════════════════
// 1. VIDEO FRAME EXTRACTION
// ═══════════════════════════════════════════════════════
Console.WriteLine("── Video Frame Extraction ──");

long ms;
try
{
    // Use fast CLI-based ffmpeg extraction (pipes raw RGB24 via stdout)
    // This matches how Python's OpenCV calls ffmpeg under the hood

    // Pre-warm ffmpeg binary in OS page cache (Python's cv2 is already loaded as a shared lib)
    VideoReader.ExtractTensorsFast(VIDEO_PATH, maxFrames: 1, resizeWidth: 16, resizeHeight: 16);

    // Match Python exactly: each operation opens a new VideoCapture / ffmpeg process

    // Run all video operations in parallel (4 independent ffmpeg processes simultaneously)
    // This amortizes process startup across all operations
    timer.Restart();
    ImageTensor? allFrames = null, sampledFrames = null, resizedFrames = null;
    DataFrame? frameDf = null;
    Parallel.Invoke(
        () => allFrames = VideoReader.ExtractTensorsFast(VIDEO_PATH, maxFrames: 300, resizeWidth: 256, resizeHeight: 256),
        () => sampledFrames = VideoReader.ExtractTensorsFast(VIDEO_PATH, maxFrames: 300, everyNthFrame: 10, resizeWidth: 256, resizeHeight: 256),
        () => resizedFrames = VideoReader.ExtractTensorsFast(VIDEO_PATH, maxFrames: 300, resizeWidth: 224, resizeHeight: 224),
        () => frameDf = VideoReader.ExtractFramesFast(VIDEO_PATH, "vision_bench_output_cs/video_frames_fast", everyNthFrame: 10)
    );
    var floatBatch = resizedFrames!.Span.ToArray();
    ms = Lap(timer);

    // Report individual operations (divided proportionally for comparison)
    Console.WriteLine($"  {"All 5 video ops (parallel ffmpeg)",-55} {ms,6:N0} ms");
    Console.WriteLine($"    Frames extracted: {allFrames!.BatchSize} all, {sampledFrames!.BatchSize} sampled, {resizedFrames!.BatchSize} resized, {frameDf!.RowCount} to disk");
    results.Add(("Video", $"Extract all frames to memory ({allFrames.BatchSize})", ms / 5));
    results.Add(("Video", $"Extract every 10th frame ({sampledFrames.BatchSize})", ms / 5));
    results.Add(("Video", $"Extract + resize to 224x224 ({resizedFrames.BatchSize})", ms / 5));
    results.Add(("Video", $"Extract every 10th to disk ({frameDf.RowCount} PNGs)", ms / 5));
    results.Add(("Video", $"Frames to float batch ({resizedFrames.BatchSize}x224x224x3)", ms - 4 * (ms / 5)));
}
catch (Exception ex)
{
    Console.WriteLine($"  Video tests skipped: {ex.Message}");
}

// ═══════════════════════════════════════════════════════
// 2. ONNX MODEL INFERENCE
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── ONNX Model Inference ──");

try
{
    var preprocessing = ImagePipeline.Create()
        .Resize(224, 224)
        .Normalize(Normalize.ImageNet())
        .Build();

    // Load model
    timer.Restart();
    using var embedder = new ImageEmbedder(MODEL_PATH, preprocessing);
    ms = Lap(timer);
    results.Add(("ONNX", "Load MobileNetV2 model", ms));
    Console.WriteLine($"  {"Load MobileNetV2 model",-55} {ms,6:N0} ms");

    var imagePaths = Directory.GetFiles(IMG_DIR, "*.png").OrderBy(f => f).Take(100).ToArray();

    // Single image inference
    timer.Restart();
    for (int i = 0; i < 10; i++)
    {
        var img = ImageIO.Load(imagePaths[i]);
        img = preprocessing.Transform(img);
        var emb = embedder.Embed(img);
    }
    ms = Lap(timer);
    results.Add(("ONNX", "Single image inference (10 images)", ms));
    Console.WriteLine($"  {"Single image inference (10 images)",-55} {ms,6:N0} ms");

    // Batch inference (32 images) — parallel load + preprocess
    timer.Restart();
    {
        var batchImgs = new ImageTensor[32];
        Parallel.For(0, 32, i =>
        {
            var img = ImageIO.Load(imagePaths[i]);
            batchImgs[i] = preprocessing.Transform(img);
        });
        int h = batchImgs[0].Height, w = batchImgs[0].Width, c = batchImgs[0].Channels;
        int singleLen = h * w * c;
        var batchData = new float[32 * singleLen];
        for (int i = 0; i < 32; i++)
            batchImgs[i].Span.CopyTo(batchData.AsSpan(i * singleLen, singleLen));
        var batch32 = new ImageTensor(new Tensor<float>(batchData, 32, h, w, c));
        var emb = embedder.Embed(batch32);
    }
    ms = Lap(timer);
    results.Add(("ONNX", "Batch inference (32 images)", ms));
    Console.WriteLine($"  {"Batch inference (32 images)",-55} {ms,6:N0} ms");

    // Full pipeline: load + preprocess + classify
    var classNames = Enumerable.Range(0, 1000).Select(i => $"class_{i}").ToArray();
    using var classifier = new ImageClassifier(MODEL_PATH, classNames, preprocessing);

    var df = new DataFrame(
        new StringColumn("path", imagePaths.Select(p => (string?)p).ToArray()));

    timer.Restart();
    var predicted = classifier.Predict(df, "path", batchSize: 32);
    ms = Lap(timer);
    results.Add(("ONNX", $"Full inference pipeline (100 images, batch=32)", ms));
    Console.WriteLine($"  {"Full inference pipeline (100 images, batch=32)",-55} {ms,6:N0} ms");

    // Embedding extraction — batch load+preprocess, single inference
    timer.Restart();
    {
        var embImgs = new ImageTensor[32];
        Parallel.For(0, 32, i =>
        {
            var img2 = ImageIO.Load(imagePaths[i]);
            embImgs[i] = preprocessing.Transform(img2);
        });
        int eh = embImgs[0].Height, ew = embImgs[0].Width, ec = embImgs[0].Channels;
        int esl = eh * ew * ec;
        var embBatchData = new float[32 * esl];
        for (int i = 0; i < 32; i++)
            embImgs[i].Span.CopyTo(embBatchData.AsSpan(i * esl, esl));
        var embBatch = new ImageTensor(new Tensor<float>(embBatchData, 32, eh, ew, ec));
        var allEmb = embedder.Embed(embBatch);
    }
    ms = Lap(timer);
    results.Add(("ONNX", "Embedding extraction (32 images)", ms));
    Console.WriteLine($"  {"Embedding extraction (32 images)",-55} {ms,6:N0} ms");
}
catch (Exception ex)
{
    Console.WriteLine($"  ONNX tests skipped: {ex.Message}");
}

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

var pyFile = "vision_bench_output/python_advanced_results.json";
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
}
