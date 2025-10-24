#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib

# Try to activate iTerm2 inline backend if available
try:
    import itermplot
    matplotlib.use("module://itermplot")
except ImportError:
    print("⚠️ itermplot not available, using default matplotlib backend.")

# Path to your CSV file
csv_path = "/data/benchmarks/sweep_20251023_173045/benchmark_results.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Normalize numeric types
df["input_len"] = df["input_len"].astype(float).astype(int)
df["output_len"] = df["output_len"].astype(float).astype(int)

# --- Find best throughput per input/output pair ---
best = (
    df.sort_values("output_tokens_per_sec", ascending=False)
      .groupby(["input_len", "output_len"], as_index=False)
      .first()
)

best["token_pair"] = best.apply(lambda r: f"{r['input_len']}/{r['output_len']}", axis=1)
best = best.sort_values(["input_len", "output_len"])

# Print best runs summary
print("\n🏆 Top-performing runs per input/output pair:\n")
print(best[["token_pair", "output_tokens_per_sec", "num_concurrent"]]
      .rename(columns={
          "token_pair": "Input/Output Tokens",
          "output_tokens_per_sec": "Max Throughput (tok/s)",
          "num_concurrent": "Best Concurrency"
      })
      .to_string(index=False))

# --- Plot ---
x = np.arange(len(best))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))
ax2 = ax1.twinx()

# Bars: Max output throughput
ax1.bar(x, best["output_tokens_per_sec"], width, color="#0071C5", label="Max Output Throughput (tokens/s)")

# Line: Concurrency level for that best run
ax2.plot(x, best["num_concurrent"], color="#FF9900", marker="o", linewidth=2, label="Best Concurrency")

# Labels and axes
ax1.set_xlabel("Input / Output Tokens")
ax1.set_ylabel("Output Throughput (tokens/s)")
ax2.set_ylabel("Concurrent Requests (best case)")
ax1.set_xticks(x)
ax1.set_xticklabels(best["token_pair"], rotation=30, ha="right")
ax1.grid(axis='y', linestyle='--', alpha=0.6)
plt.title("vLLM Serving — Max Throughput per Token Length")

# Combine legends
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc="upper right")

plt.tight_layout()

# --- Save and/or show the plot ---
out_path = os.path.join(os.path.dirname(csv_path), "benchmark_plot.png")

# Save safely using Agg backend (no TypeError)
from matplotlib.backends.backend_agg import FigureCanvasAgg
canvas = FigureCanvasAgg(fig)
fig.set_dpi(300)
canvas.print_png(out_path)
print(f"\n✅ Plot saved to: {out_path}")

# Inline display (works in iTerm2/Kitty/etc.)
try:
    plt.show()
except Exception:
    print("⚠️ Inline display not supported in this terminal.")

