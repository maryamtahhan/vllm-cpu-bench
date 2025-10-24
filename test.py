import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load your benchmark results
df = pd.read_csv("/data/benchmarks/sweep_20251023_173045/benchmark_results.csv")

# Combine input/output into a label
df["io_label"] = df.apply(lambda r: f"{r['input_len']}/{r['output_len']}", axis=1)

# Aggregate (mean) throughput per input/output
agg = df.groupby("io_label", as_index=False).agg({
    "output_tokens_per_sec": "mean",
    "num_concurrent": "mean"
})

# Example: If you have two systems, you'd have two dataframes and merge them here

x = np.arange(len(agg))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()

# Bars for throughput (left axis)
bars = ax1.bar(x, agg["output_tokens_per_sec"], width, color="#0071C5", label="Output Throughput (tokens/s)")

# Line for user prompts (right axis)
ax2.plot(x, agg["num_concurrent"], color="#FF9900", marker="o", label="User Prompts")

# Axis labels and formatting
ax1.set_xlabel("Input / Output Tokens")
ax1.set_ylabel("Output Throughput (tokens/s)")
ax2.set_ylabel("User Prompts")
ax1.set_xticks(x)
ax1.set_xticklabels(agg["io_label"])
ax1.set_ylim(0, agg["output_tokens_per_sec"].max() * 1.2)
ax2.set_ylim(0, agg["num_concurrent"].max() * 1.2)

# Add gridlines and title
ax1.grid(axis='y', linestyle='--', alpha=0.6)
plt.title("vLLM Serving Throughput vs Token Length")

# Combine legends from both axes
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.show()

