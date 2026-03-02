import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re

# Collect all benchmark files
files = glob.glob("./benchmarks/benchmark_depth_*.csv")

plt.figure()

summary_data = []

for file in files:
    df = pd.read_csv(file)

    # Extract depth and width from filename
    filename = os.path.basename(file)

    match = re.search(r"depth_(\d+)_width_(\d+)", filename)
    if match:
        depth = int(match.group(1))
        width = int(match.group(2))
        label = f"{depth}D-{width}W"
    else:
        label = filename

    # Plot runtime per move
    plt.plot(df["iteration"], df["seconds"], label=label)

    # Store summary statistics
    summary_data.append({
        "depth": depth,
        "width": width,
        "avg_time": df["seconds"].mean(),
        "total_time": df["seconds"].sum()
    })

plt.xlabel("Iteration")
plt.ylabel("Time (seconds)")
plt.title("Runtime per Move")
plt.legend()
plt.show()


# --------- Print Summary Table ---------

summary_df = pd.DataFrame(summary_data)
print("\n=== Benchmark Summary ===")
print(summary_df.sort_values("avg_time"))