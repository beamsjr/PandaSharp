"""
PandaSharp vs pandas — Microbenchmark Suite
============================================
Tests the specific operations where architecture differences matter most.
Run: python3 benchmarks/microbenchmark_python.py
"""

import time
import json
import numpy as np
import pandas as pd

results = []
def bench(name, fn, warmup=1, runs=5):
    for _ in range(warmup): fn()
    times = []
    for _ in range(runs):
        t = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t) * 1_000_000)  # microseconds
    median = sorted(times)[len(times)//2]
    results.append({"name": name, "median_us": round(median, 1)})
    print(f"  {name:<45} {median:>12,.1f} μs")

print("=== Python (pandas/numpy) Microbenchmarks ===")
print(f"  pandas {pd.__version__}, numpy {np.__version__}\n")

# ── Setup ──────────────────────────────────────────────
N = 100_000
rng = np.random.default_rng(42)
ids = np.arange(N)
values = rng.random(N) * 1000
categories = rng.choice(["Alpha", "Beta", "Gamma", "Delta", "Epsilon"], N)

df = pd.DataFrame({"Id": ids, "Value": values, "Category": categories})
small_df = df.head(10_000)

# Right side for join
right_ids = np.arange(0, N, 100)
right_labels = [f"Label_{i}" for i in range(len(right_ids))]
right_df = pd.DataFrame({"Id": right_ids, "Label": right_labels})

val_col = df["Value"]
val_array = values.copy()

print("── Aggregates (100K doubles) ──")
bench("Sum", lambda: val_col.sum())
bench("Mean", lambda: val_col.mean())
bench("Std", lambda: val_col.std())
bench("Median", lambda: val_col.median())
bench("Min", lambda: val_col.min())
bench("Max", lambda: val_col.max())

print("\n── Arithmetic (100K doubles) ──")
bench("Column + Column", lambda: val_col + val_col)
bench("Column * Scalar", lambda: val_col * 2.5)
bench("Column + Scalar", lambda: val_col + 100)

print("\n── Filter (100K rows) ──")
bench("Boolean mask filter", lambda: df[df["Value"] > 500])
bench("Lambda filter (apply)", lambda: df[df["Value"].apply(lambda x: x > 500)])

print("\n── Sort ──")
bench("Sort 10K by double", lambda: small_df.sort_values("Value"))
bench("Sort 100K by double", lambda: df.sort_values("Value"))

print("\n── GroupBy (100K rows, 5 groups) ──")
bench("GroupBy Sum", lambda: df.groupby("Category")["Value"].sum())
bench("GroupBy Mean", lambda: df.groupby("Category")["Value"].mean())
bench("GroupBy Count", lambda: df.groupby("Category")["Value"].count())

print("\n── Join (100K × 1K on int key) ──")
bench("Inner Join", lambda: df.merge(right_df, on="Id", how="inner"))

print("\n── Head/Tail/Select ──")
bench("Head(100)", lambda: df.head(100))
bench("Tail(100)", lambda: df.tail(100))
bench("Select 2 cols", lambda: df[["Id", "Value"]])

print("\n── Rolling Window (100K) ──")
bench("Rolling(20).Mean()", lambda: val_col.rolling(20).mean())
bench("Rolling(50).Mean()", lambda: val_col.rolling(50).mean())
bench("Expanding Mean", lambda: val_col.expanding().mean())

print("\n── String Operations (100K) ──")
cat_col = df["Category"]
bench("String contains", lambda: cat_col.str.contains("Alpha"))
bench("String upper", lambda: cat_col.str.upper())
bench("String len", lambda: cat_col.str.len())

print("\n── I/O (100K rows) ──")
import io, tempfile, os

csv_bytes = df.to_csv(index=False).encode()
bench("CSV parse (from bytes)", lambda: pd.read_csv(io.BytesIO(csv_bytes)))

pq_path = os.path.join(tempfile.gettempdir(), "bench.parquet")
df.to_parquet(pq_path)
bench("Parquet read", lambda: pd.read_parquet(pq_path))

print("\n── Nlargest/Nsmallest (100K) ──")
bench("Nlargest(10)", lambda: df.nlargest(10, "Value"))
bench("Nsmallest(10)", lambda: df.nsmallest(10, "Value"))

print("\n── Value Counts / Unique ──")
bench("Value counts (Category)", lambda: cat_col.value_counts())
bench("Nunique (Category)", lambda: cat_col.nunique())
bench("Drop duplicates (Category)", lambda: df.drop_duplicates("Category"))

# Save results
with open("stock_output_python/microbench_results.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to stock_output_python/microbench_results.json")
