import time, json, os, numpy as np
from safetensors.numpy import save_file, load_file

os.makedirs("st_bench_output", exist_ok=True)
results = []
def lap(cat, name, start):
    ms = round((time.perf_counter() - start) * 1000)
    results.append({"category": cat, "op": name, "ms": ms})
    print(f"  {name:<55} {ms:>6} ms")

print("=== Python SafeTensors Benchmark ===\n")

# Small model: 10 tensors, ~10MB
tensors_small = {f"layer_{i}.weight": np.random.randn(512, 512).astype(np.float32) for i in range(10)}
t = time.perf_counter()
save_file(tensors_small, "st_bench_output/small_model.safetensors")
lap("SafeTensors", "Write 10 tensors (~10MB)", t)

t = time.perf_counter()
loaded = load_file("st_bench_output/small_model.safetensors")
lap("SafeTensors", "Read 10 tensors (~10MB)", t)

# Large model: 50 tensors, ~50MB
tensors_large = {f"layer_{i}.weight": np.random.randn(512, 512).astype(np.float32) for i in range(50)}
t = time.perf_counter()
save_file(tensors_large, "st_bench_output/large_model.safetensors")
lap("SafeTensors", "Write 50 tensors (~50MB)", t)

t = time.perf_counter()
loaded = load_file("st_bench_output/large_model.safetensors")
lap("SafeTensors", "Read 50 tensors (~50MB)", t)

# Summary
print(f"\n{'═'*70}")
total = sum(r["ms"] for r in results)
print(f"  {'TOTAL':<30} {total:>8,} ms")
with open("st_bench_output/python_st_results.json", "w") as f: json.dump(results, f, indent=2)
