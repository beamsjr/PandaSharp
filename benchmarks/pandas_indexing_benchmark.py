"""Benchmark pandas indexing operations for comparison with PandaSharp."""
import time
import numpy as np
import pandas as pd

np.random.seed(42)
N = 100_000
CATS = ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"]

df = pd.DataFrame({
    "Id": np.arange(N),
    "Value": np.random.rand(N) * 1000,
    "Category": [CATS[i % len(CATS)] for i in range(N)],
})

df_small = df.head(10_000).copy()

def bench(name, func, iterations=20):
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)
    times.sort()
    return name, times[len(times) // 2]

results = []

# at — scalar access 1000 times
def at_access():
    last = None
    for i in range(1000):
        last = df.at[i * 100, "Value"]
    return last
results.append(bench("at (1000 accesses)", at_access))

# iat — scalar access 1000 times
def iat_access():
    last = None
    for i in range(1000):
        last = df.iat[i * 100, 1]
    return last
results.append(bench("iat (1000 accesses)", iat_access))

# xs — cross section
df_indexed = df.set_index("Category")
results.append(bench("xs (cross-section)", lambda: df_indexed.xs("Alpha")))

# set_index + reset_index
results.append(bench("set_index+reset_index", lambda: df.set_index("Category").reset_index()))

# MultiIndex set + reset
results.append(bench("multi_set+reset (10K)", lambda: df_small.set_index(["Id", "Category"]).reset_index()))

print(f"\npandas {pd.__version__}")
print(f"{'Operation':<30} {'Median (μs)':>15}")
print("-" * 48)
for name, ns in results:
    us = ns / 1000
    ms = ns / 1_000_000
    if ms >= 1:
        print(f"{name:<30} {us:>15,.0f} ({ms:.1f}ms)")
    else:
        print(f"{name:<30} {us:>15,.1f}")
