import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# ----------------- User input: 2 CSV files -----------------
file_normal = "./benchmarks/benchmark_depth_4_width_6_20260301_203606_.csv"
file_hashed = "./benchmarks/benchmark_depth_4_width_6_20260301_203730_hashed.csv"

files = [("Normal", file_normal), ("Hashed", file_hashed)]

plt.figure(figsize=(10,6))

summary_data = []

for label, file in files:
    if not os.path.isfile(file):
        print(f"File not found: {file}")
        continue

    df = pd.read_csv(file)

    # Extract depth & width from filename (optional)
    filename = os.path.basename(file)
    match = re.search(r"depth_(\d+)_width_(\d+)", filename)
    depth, width = (int(match.group(1)), int(match.group(2))) if match else (None, None)

    # Plot per-move runtime
    plt.plot(df["iteration"], df["seconds"], label=f"{label} ({depth}D-{width}W)")

    # Collect summary
    summary_data.append({
        "label": label,
        "depth": depth,
        "width": width,
        "avg_time": df["seconds"].mean(),
        "total_time": df["seconds"].sum()
    })

plt.xlabel("Iteration")
plt.ylabel("Time (seconds)")
plt.title("Benchmark Comparison: Normal vs Hashed")
plt.legend()
plt.grid(True)
plt.show()

# Print summary table
summary_df = pd.DataFrame(summary_data)
print("\n=== Benchmark Summary ===")
print(summary_df.sort_values("avg_time"))