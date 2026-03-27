using System.Diagnostics;
using System.Text.Json;
using Cortex.ML.Tensors;
using Cortex.Text.Tokenizers;
using Cortex.Text.Preprocessing;
using Cortex.Text.Embeddings;

const int CorpusSize = 10_000;
const int EmbeddingCount = 1_000;
const int EmbeddingDim = 384;

var results = new List<(string Cat, string Op, long Ms)>();
var timer = Stopwatch.StartNew();
long Lap() { var ms = timer.ElapsedMilliseconds; timer.Restart(); return ms; }

// Generate corpus: 10K sentences (same as Python benchmark)
var corpus = new string[CorpusSize];
for (int i = 0; i < CorpusSize; i++)
    corpus[i] = $"The quick brown fox jumps over the lazy dog number {i} with extra words for padding and testing tokenization";

Console.WriteLine("=== Cortex.Text Benchmark ===\n");

// ═══════════════════════════════════════════════════════
// 1. TOKENIZATION
// ═══════════════════════════════════════════════════════
Console.WriteLine("── Tokenization ──");

// Pre-train tokenizers (not timed — Python doesn't train either)
var wsTok = new WhitespaceTokenizer(lowercase: true);
wsTok.Train(corpus);
var reTok = new RegexTokenizer(lowercase: true);
reTok.Train(corpus);

// Whitespace tokenize — matches Python: [s.lower().split() for s in corpus]
timer.Restart();
for (int i = 0; i < CorpusSize; i++)
    wsTok.Encode(corpus[i]);
var ms = Lap();
results.Add(("Tokenize", "Whitespace tokenize (10K)", ms));
Console.WriteLine($"  {"Whitespace tokenize (10K)",-55} {ms,6:N0} ms");

// Regex tokenize — matches Python: [re.findall(r'\w+', s.lower()) for s in corpus]
timer.Restart();
for (int i = 0; i < CorpusSize; i++)
    reTok.Encode(corpus[i]);
ms = Lap();
results.Add(("Tokenize", "Regex tokenize (10K)", ms));
Console.WriteLine($"  {"Regex tokenize (10K)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 2. PREPROCESSING
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Preprocessing ──");

var stopRemover = new StopWordRemover("english");
timer.Restart();
for (int i = 0; i < CorpusSize; i++)
    stopRemover.RemoveStopWords(corpus[i]);
ms = Lap();
results.Add(("Preprocess", "Stop word removal (10K)", ms));
Console.WriteLine($"  {"Stop word removal (10K)",-55} {ms,6:N0} ms");

var stemmer = new Stemmer();
timer.Restart();
for (int i = 0; i < CorpusSize; i++)
{
    var words = corpus[i].Split((char[]?)null, StringSplitOptions.RemoveEmptyEntries);
    for (int j = 0; j < words.Length; j++)
        stemmer.Stem(words[j]);
}
ms = Lap();
results.Add(("Preprocess", "Porter stemming (10K)", ms));
Console.WriteLine($"  {"Porter stemming (10K)",-55} {ms,6:N0} ms");

var ngramExtractor = new NGramExtractor(2);
timer.Restart();
for (int i = 0; i < CorpusSize; i++)
    ngramExtractor.Extract(corpus[i]);
ms = Lap();
results.Add(("Preprocess", "Bigram extraction (10K)", ms));
Console.WriteLine($"  {"Bigram extraction (10K)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// 3. SIMILARITY
// ═══════════════════════════════════════════════════════
Console.WriteLine("\n── Similarity ──");

// Build 1K x 384 embedding tensor with random data
var rng = new Random(42);
var embData = new double[EmbeddingCount * EmbeddingDim];
for (int i = 0; i < embData.Length; i++)
    embData[i] = rng.NextDouble() * 2 - 1;
var embeddings = new Tensor<double>(embData, EmbeddingCount, EmbeddingDim);

timer.Restart();
var simMatrix = CosineSimilarity.PairwiseMatrix(embeddings);
ms = Lap();
results.Add(("Similarity", "Pairwise cosine (1Kx1K, 384-dim)", ms));
Console.WriteLine($"  {"Pairwise cosine (1Kx1K, 384-dim)",-55} {ms,6:N0} ms");

// ═══════════════════════════════════════════════════════
// SUMMARY + COMPARISON
// ═══════════════════════════════════════════════════════
Console.WriteLine($"\n{"═",70}");

var cats = new Dictionary<string, long>();
foreach (var r in results)
{
    if (!cats.ContainsKey(r.Cat)) cats[r.Cat] = 0;
    cats[r.Cat] += r.Ms;
}
foreach (var (cat, total) in cats.OrderByDescending(kv => kv.Value))
    Console.WriteLine($"  {cat,-30} {total,8:N0} ms");

var grandTotal = results.Sum(r => r.Ms);
Console.WriteLine($"  {"TOTAL",-30} {grandTotal,8:N0} ms");

// Save results
Directory.CreateDirectory("text_bench_output");
var jsonResults = results.Select(r => new { category = r.Cat, op = r.Op, ms = r.Ms }).ToArray();
File.WriteAllText("text_bench_output/csharp_text_results.json",
    JsonSerializer.Serialize(jsonResults, new JsonSerializerOptions { WriteIndented = true }));

// Load Python results for comparison
var pyResultsPath = "text_bench_output/python_text_results.json";
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
    Console.WriteLine($"\n  (Run text_python.py first to enable comparison)");
}
