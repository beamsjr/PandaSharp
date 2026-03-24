"""Benchmark pandas Parquet/Excel/HTML read for comparison."""
import time, io, tempfile, os
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

tmpdir = tempfile.mkdtemp()

def bench(name, func, iterations=5):
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        func()
        end = time.perf_counter_ns()
        times.append(end - start)
    times.sort()
    return name, times[len(times) // 2]

# Write files
parquet_path = os.path.join(tmpdir, "test.parquet")
df.to_parquet(parquet_path)

excel_path = os.path.join(tmpdir, "test.xlsx")
df.head(10_000).to_excel(excel_path, index=False)  # Excel is slow, use 10K

csv_path = os.path.join(tmpdir, "test.csv")
df.to_csv(csv_path, index=False)

html_str = df.head(1000).to_html()

results = []
results.append(bench("read_parquet", lambda: pd.read_parquet(parquet_path)))
results.append(bench("read_csv", lambda: pd.read_csv(csv_path)))
results.append(bench("read_excel (10K)", lambda: pd.read_excel(excel_path), iterations=3))
results.append(bench("read_html (1K)", lambda: pd.read_html(io.StringIO(html_str)), iterations=3))

print(f"\npandas {pd.__version__}")
print(f"{'Operation':<25} {'Median (ms)':>12}")
print("-" * 40)
for name, ns in results:
    print(f"{name:<25} {ns / 1_000_000:>11.1f}ms")

# Cleanup
import shutil
shutil.rmtree(tmpdir)
