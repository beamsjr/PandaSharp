using System.Diagnostics;
using System.Text.Json;
using PandaSharp.ML.Tensors;
using PandaSharp.SafeTensors;

const string OutputDir = "st_bench_output";
Directory.CreateDirectory(OutputDir);

var results = new List<(string Cat, string Op, long Ms)>();
var timer = Stopwatch.StartNew();
long Lap() { var ms = timer.ElapsedMilliseconds; timer.Restart(); return ms; }

Console.WriteLine("=== PandaSharp.SafeTensors Benchmark ===\n");

var rng = new Random(42);

// ═══════════════════════════════════════════════════════
// Small model: 10 tensors, 512x512 float (~10MB)
// ═══════════════════════════════════════════════════════
var smallTensors = new List<(string Name, Tensor<float> Data)>();
for (int i = 0; i < 10; i++)
{
    var data = new float[512 * 512];
    for (int j = 0; j < data.Length; j++)
        data[j] = (float)(rng.NextDouble() * 2 - 1);
    smallTensors.Add(($"layer_{i}.weight", new Tensor<float>(data, 512, 512)));
}

timer.Restart();
var smallWriter = new SafeTensorWriter();
foreach (var (name, tensor) in smallTensors)
    smallWriter.Add(name, tensor);
smallWriter.Save(Path.Combine(OutputDir, "small_model.safetensors"));
var ms = Lap();
results.Add(("SafeTensors", "Write 10 tensors (~10MB)", ms));
Console.WriteLine($"  {"Write 10 tensors (~10MB)",-55} {ms,6:N0} ms");

timer.Restart();
using (var reader = SafeTensorReader.Open(Path.Combine(OutputDir, "small_model.safetensors")))
{
    foreach (var tName in reader.GetTensorNames())
        reader.GetTensor<float>(tName);
}
ms = Lap();
results.Add(("SafeTensors", "Read 10 tensors (~10MB)", ms));
Console.WriteLine($"  {"Read 10 tensors (~10MB)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// Large model: 50 tensors, 512x512 float (~50MB)
// ═══════════════════════════════════════════════════════
var largeTensors = new List<(string Name, Tensor<float> Data)>();
for (int i = 0; i < 50; i++)
{
    var data = new float[512 * 512];
    for (int j = 0; j < data.Length; j++)
        data[j] = (float)(rng.NextDouble() * 2 - 1);
    largeTensors.Add(($"layer_{i}.weight", new Tensor<float>(data, 512, 512)));
}

timer.Restart();
var largeWriter = new SafeTensorWriter();
foreach (var (name, tensor) in largeTensors)
    largeWriter.Add(name, tensor);
largeWriter.Save(Path.Combine(OutputDir, "large_model.safetensors"));
ms = Lap();
results.Add(("SafeTensors", "Write 50 tensors (~50MB)", ms));
Console.WriteLine($"  {"Write 50 tensors (~50MB)",-55} {ms,6:N0} ms");

timer.Restart();
using (var reader = SafeTensorReader.Open(Path.Combine(OutputDir, "large_model.safetensors")))
{
    foreach (var tName in reader.GetTensorNames())
        reader.GetTensor<float>(tName);
}
ms = Lap();
results.Add(("SafeTensors", "Read 50 tensors (~50MB)", ms));
Console.WriteLine($"  {"Read 50 tensors (~50MB)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// SUMMARY + COMPARISON
// ═══════════════════════════════════════════════════════
Console.WriteLine($"\n{"═",70}");
var grandTotal = results.Sum(r => r.Ms);
Console.WriteLine($"  {"TOTAL",-30} {grandTotal,8:N0} ms");

// Save results
var jsonResults = results.Select(r => new { category = r.Cat, op = r.Op, ms = r.Ms }).ToArray();
File.WriteAllText(Path.Combine(OutputDir, "csharp_st_results.json"),
    JsonSerializer.Serialize(jsonResults, new JsonSerializerOptions { WriteIndented = true }));

// Load Python results for comparison
var pyResultsPath = Path.Combine(OutputDir, "python_st_results.json");
if (File.Exists(pyResultsPath))
{
    Console.WriteLine($"\n{"═",70}");
    Console.WriteLine("  Python vs C# Comparison:\n");
    Console.WriteLine($"  {"Operation",-45} {"Python",8} {"C#",8} {"Speedup",8}");
    Console.WriteLine($"  {new string('-', 69)}");

    using var doc = JsonDocument.Parse(File.ReadAllText(pyResultsPath));
    var pyResults = new Dictionary<string, long>();
    foreach (var elem in doc.RootElement.EnumerateArray())
    {
        var op = elem.GetProperty("op").GetString()!;
        var pyMs = elem.GetProperty("ms").GetInt64();
        pyResults[op] = pyMs;
    }

    foreach (var r in results)
    {
        if (pyResults.TryGetValue(r.Op, out var pyMs))
        {
            var speedup = pyMs == 0 ? "N/A" : $"{(double)pyMs / r.Ms:F1}x";
            Console.WriteLine($"  {r.Op,-45} {pyMs,7:N0}ms {r.Ms,7:N0}ms {speedup,8}");
        }
        else
        {
            Console.WriteLine($"  {r.Op,-45} {"N/A",8} {r.Ms,7:N0}ms {"",8}");
        }
    }
}
else
{
    Console.WriteLine($"\n  (Run safetensors_python.py first to enable comparison)");
}
