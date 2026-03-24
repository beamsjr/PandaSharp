"""
PandaSharp vs pandas benchmark comparison.
Uses identical data and operations to the C# BenchmarkDotNet suite.
100K rows, same random seed, same column types.
"""

import time
import numpy as np
import pandas as pd
import io
import json
import sys

# Ensure reproducible results
np.random.seed(42)

N = 100_000
CATS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

# Build the same DataFrame as the C# benchmark
ids = np.arange(N)
values = np.random.rand(N) * 1000
categories = np.array([CATS[i % len(CATS)] for i in range(N)])

df = pd.DataFrame({
    "Id": ids,
    "Value": values,
    "Category": categories,
})

# Right side for join (same as C#)
right_ids = np.arange(0, 100_000, 100)  # 1000 entries
right_labels = [f"Label_{i}" for i in range(1000)]
right = pd.DataFrame({"Id": right_ids, "Label": right_labels})

# Small DataFrame for sort
df_small = df.head(10_000).copy()

# CSV data for parsing benchmark
csv_buf = io.StringIO()
csv_df = pd.DataFrame({
    "Id": np.arange(N),
    "Value": np.random.rand(N) * 1000,
    "Category": np.array([CATS[np.random.randint(3)] for _ in range(N)]),
})
csv_df.to_csv(csv_buf, index=False)
csv_text = csv_buf.getvalue()


def bench(name, func, iterations=20):
    """Run func multiple times and report median time."""
    times = []
    result = None
    for _ in range(iterations):
        start = time.perf_counter_ns()
        result = func()
        end = time.perf_counter_ns()
        times.append(end - start)
    times.sort()
    median_ns = times[len(times) // 2]
    return name, median_ns, result


results = []

# === Core operations ===
results.append(bench("Head", lambda: df.head(100)))
results.append(bench("Tail", lambda: df.tail(100)))
results.append(bench("Select", lambda: df[["Id", "Value"]]))
results.append(bench("FilterMask", lambda: df[df["Value"] > 500.0]))
results.append(bench("FilterLambda", lambda: df[df.apply(lambda row: row["Value"] > 500.0, axis=1)], iterations=3))
results.append(bench("Sort", lambda: df_small.sort_values("Value")))
results.append(bench("SortLarge", lambda: df.sort_values("Value"), iterations=5))

# === Aggregation ===
results.append(bench("Sum", lambda: df["Value"].sum()))
results.append(bench("Mean", lambda: df["Value"].mean()))
results.append(bench("Median", lambda: df["Value"].median()))
results.append(bench("Std", lambda: df["Value"].std()))

# === GroupBy ===
results.append(bench("GroupBySum", lambda: df.groupby("Category")["Value"].sum()))
results.append(bench("GroupByMean", lambda: df.groupby("Category")["Value"].mean()))

# === Join ===
results.append(bench("JoinInner", lambda: df_small.merge(right, on="Id", how="inner")))
results.append(bench("JoinLarger", lambda: df.merge(right, on="Id", how="inner")))

# === DropDuplicates ===
results.append(bench("DropDuplicates", lambda: df_small.drop_duplicates("Category")))

# === Arithmetic ===
val_col = df["Value"]
results.append(bench("ArithmeticAdd", lambda: val_col + val_col))
results.append(bench("ArithmeticMultiply", lambda: val_col * 2.0))

# === Window ===
results.append(bench("RollingMean100K", lambda: val_col.rolling(10).mean()))
results.append(bench("ExpandingMean100K", lambda: val_col.expanding().mean()))

# === String ===
cat_col = df["Category"]
results.append(bench("StringContains", lambda: cat_col.str.contains("Alpha")))
results.append(bench("StringUpper", lambda: cat_col.str.upper()))
results.append(bench("StringLen", lambda: cat_col.str.len()))

# === CSV ===
results.append(bench("CsvParse", lambda: pd.read_csv(io.StringIO(csv_text)), iterations=5))
results.append(bench("CsvParseWithDtypes", lambda: pd.read_csv(io.StringIO(csv_text), dtype={"Id": int, "Value": float, "Category": str}), iterations=5))

# === Pipeline ===
def pipeline():
    filtered = df[df["Value"] > 500.0]
    return filtered.groupby("Category")["Value"].sum()
results.append(bench("Pipeline", pipeline))

# === GetDummies ===
results.append(bench("GetDummies100K", lambda: pd.get_dummies(df["Category"]), iterations=5))

# === Print results ===
print(f"\npandas {pd.__version__} | numpy {np.__version__} | Python {sys.version.split()[0]}")
print(f"{'Operation':<25} {'Median (μs)':>15} {'Median (ms)':>12}")
print("-" * 55)
for name, ns, _ in results:
    us = ns / 1000
    ms = ns / 1_000_000
    if ms >= 1:
        print(f"{name:<25} {us:>15,.0f} {ms:>11.1f}ms")
    else:
        print(f"{name:<25} {us:>15,.1f} {'':>12}")

# Output JSON for easy comparison
comparison = {name: ns for name, ns, _ in results}
with open("/Users/joe/Documents/Repository/lab/PandaSharp/benchmarks/pandas_results.json", "w") as f:
    json.dump(comparison, f, indent=2)
print("\nResults saved to pandas_results.json")
