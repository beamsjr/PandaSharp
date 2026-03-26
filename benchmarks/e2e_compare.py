"""
Compare end-to-end pipeline results: PandaSharp vs Python ecosystem.

Usage:
  1. Run: python e2e_pipeline_python.py
  2. Run: dotnet run --project E2EPipelineBench -c Release
  3. Copy e2e_pipeline_csharp_results.json to this dir
  4. Run: python e2e_compare.py

Outputs a formatted comparison table highlighting conversion overhead.
"""

import json
import sys

def load(path):
    with open(path) as f:
        return json.load(f)

try:
    py = load("e2e_pipeline_python_results.json")
    cs = load("e2e_pipeline_csharp_results.json")
except FileNotFoundError as e:
    print(f"Missing file: {e.filename}")
    print("Run both benchmarks first.")
    sys.exit(1)

py_results = {(r["workflow"], r["operation"]): r["ms"] for r in py["results"]}
cs_results = {(r["workflow"], r["operation"]): r["ms"] for r in cs["results"]}

workflows = ["ML Pipeline", "TimeSeries", "Geospatial", "NLP", "MultiModel"]
workflow_labels = {
    "ML Pipeline": "ML Pipeline",
    "TimeSeries": "Time Series",
    "Geospatial": "Geospatial",
    "NLP": "NLP / Text",
    "MultiModel": "Multi-Model",
}

print()
print("=" * 78)
print("  End-to-End Pipeline Benchmark: PandaSharp vs Python Ecosystem")
print("=" * 78)
print(f"  Rows: {py['rows']:,}")
print(f"  Python: pandas + scikit-learn + statsmodels + geopandas")
print(f"  C#:     PandaSharp (integrated — zero library boundary crossings)")
print()

# Per-workflow comparison
print("┌─────────────────┬────────────┬────────────┬────────────────┬─────────────┐")
print("│    Workflow      │  Python    │    C#      │    Speedup     │ Py Convert  │")
print("├─────────────────┼────────────┼────────────┼────────────────┼─────────────┤")

total_py = 0
total_cs = 0
total_convert = 0

for wf in workflows:
    py_total = py_results.get((wf, "TOTAL"), 0)
    cs_total = cs_results.get((wf, "TOTAL"), 0)

    # Sum conversion steps for this workflow
    convert_ms = sum(
        v for (w, op), v in py_results.items()
        if w == wf and "CONVERT" in op
    )

    total_py += py_total
    total_cs += cs_total
    total_convert += convert_ms

    if cs_total > 0:
        ratio = py_total / cs_total
        if ratio >= 1:
            speedup = f"{ratio:.1f}x faster"
        else:
            speedup = f"{1/ratio:.1f}x slower"
    else:
        speedup = "N/A"

    label = workflow_labels.get(wf, wf)
    print(f"│ {label:<15}  │ {py_total:>8.0f}ms │ {cs_total:>8.0f}ms │ {speedup:>14} │ {convert_ms:>8.0f}ms  │")

print("├─────────────────┼────────────┼────────────┼────────────────┼─────────────┤")

overall_ratio = total_py / total_cs if total_cs > 0 else 0
overall_label = f"{overall_ratio:.1f}x faster" if overall_ratio >= 1 else f"{1/overall_ratio:.1f}x slower"
print(f"│ {'TOTAL':<15}  │ {total_py:>8.0f}ms │ {total_cs:>8.0f}ms │ {overall_label:>14} │ {total_convert:>8.0f}ms  │")
print("└─────────────────┴────────────┴────────────┴────────────────┴─────────────┘")

print()
print("  Conversion Overhead Analysis (Python)")
print("  " + "─" * 50)

# List all conversion steps
convert_steps = [
    (r["workflow"], r["operation"], r["ms"])
    for r in py["results"]
    if "CONVERT" in r["operation"]
]

for wf, op, ms in convert_steps:
    # Extract the conversion description
    desc = op.split("★ CONVERT: ")[1] if "★ CONVERT: " in op else op
    print(f"  {wf:<15} {desc:<35} {ms:>8.1f}ms")

print(f"  {'─'*50}")
print(f"  {'TOTAL CONVERSION TAX':<50}  {total_convert:>8.1f}ms")
print(f"  {'% of Python total':<50}  {total_convert/total_py*100:>7.1f}%")

print()
print("  Key Insight:")
print(f"  Python spends {total_convert:.0f}ms ({total_convert/total_py*100:.1f}%) just converting between")
print(f"  library formats. PandaSharp's integrated pipeline eliminates this entirely.")

if total_convert > total_cs - (total_py - total_convert):
    savings = total_convert - (total_cs - (total_py - total_convert))
    print(f"\n  Even if Python's raw compute were identical, the conversion overhead alone")
    print(f"  accounts for a {total_convert:.0f}ms penalty that PandaSharp doesn't pay.")

print()

# Detailed step-by-step comparison
print("=" * 78)
print("  Detailed Step Comparison")
print("=" * 78)

for wf in workflows:
    label = workflow_labels.get(wf, wf)
    print(f"\n  {label}:")
    print(f"  {'Step':<45} {'Python':>8} {'C#':>8} {'Note':>12}")
    print(f"  {'─'*75}")

    py_steps = [(r["operation"], r["ms"]) for r in py["results"] if r["workflow"] == wf and r["operation"] != "TOTAL"]
    cs_steps = [(r["operation"], r["ms"]) for r in cs["results"] if r["workflow"] == wf and r["operation"] != "TOTAL"]

    # Match by step letter (1a, 1b, etc.)
    for py_op, py_ms in py_steps:
        step_id = py_op[:2]  # e.g., "1a"
        cs_match = next(((op, ms) for op, ms in cs_steps if op[:2] == step_id), None)
        cs_ms = cs_match[1] if cs_match else 0

        is_convert = "CONVERT" in py_op
        note = "★ CONVERT" if is_convert else ""

        # Truncate op name
        short_op = py_op[:43]
        print(f"  {short_op:<45} {py_ms:>7.1f} {cs_ms:>7.1f} {note:>12}")

print()
