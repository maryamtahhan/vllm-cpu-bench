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
    print("⚠️  itermplot not available, using default matplotlib backend.")


# Path to your CSV file
csv_path = "/data/benchmarks/sweep_20251023_173045/benchmark_results.csv"

# Read the CSV
df = pd.read_csv(csv_path)

# Create a label for each (input_len, output_len)
df["input_len"] = df["input_len"].astype(float).astype(int)
df["output_len"] = df["output_len"].astype(float).astype(int)

df["token_pair"] = df.apply(lambda r: f"{r['input_len']}/{r['output_len']}", axis=1)



# Aggregate by token pair
agg = (
    df.groupby("token_pair", as_index=False)
      .agg({
          "output_tokens_per_sec": "mean",
          "num_concurrent": "mean"
      })
)

# Sort by approximate total token length
#agg["total_tokens"] = agg["token_pair"].apply(lambda x: sum(map(int, x.split("/"))))
#agg = agg.sort_values("total_tokens")
# Sort by approximate total token length
agg["total_tokens"] = agg["token_pair"].apply(lambda x: int(sum(float(t) for t in x.split("/"))))
agg = agg.sort_values("total_tokens")


# --- Plot ---
x = np.arange(len(agg))
width = 0.35

fig, ax1 = plt.subplots(figsize=(8,5))
ax2 = ax1.twinx()

# Bars for output throughput
ax1.bar(x, agg["output_tokens_per_sec"], width, color="#0071C5", label="Output Throughput (tokens/s)")

# Line for concurrent user prompts
ax2.plot(x, agg["num_concurrent"], color="#FF9900", marker="o", label="User Prompts")

# Labels and axes
ax1.set_xlabel("Input / Output Tokens")
ax1.set_ylabel("Output Throughput (tokens/s)")
ax2.set_ylabel("Concurrent Requests")
ax1.set_xticks(x)
ax1.set_xticklabels(agg["token_pair"])
ax1.grid(axis='y', linestyle='--', alpha=0.6)
plt.title("vLLM Serving Throughput vs Token Length")

# Combine legends
lines_labels = [ax.get_legend_handles_labels() for ax in [ax1, ax2]]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
ax1.legend(lines, labels, loc="upper right")

plt.tight_layout()

# --- Save and/or show the plot ---
out_path = os.path.join(os.path.dirname(csv_path), "benchmark_plot.png")

# 1️⃣ Save safely using a new Agg canvas
from matplotlib.backends.backend_agg import FigureCanvasAgg

canvas = FigureCanvasAgg(fig)
fig.set_dpi(300)
canvas.print_png(out_path)
print(f"✅ Plot saved to: {out_path}")

# 2️⃣ Inline display (works in iTerm2/Kitty/etc.)
try:
    plt.show()
except Exception:
    print("⚠️  Inline display not supported in this terminal.")

## Save figure next to CSV
#out_path = os.path.join(os.path.dirname(csv_path), "benchmark_plot.png")
#plt.savefig(out_path, dpi=300)
#print(f"✅ Plot saved to: {out_path}")
#plt.show()

