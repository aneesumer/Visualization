import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading and converting to tidy form
raw = pd.read_csv("runtimes.csv", comment="#")
print("Raw shape:", raw.shape)
print(raw.head())
tidy = raw.melt(
    id_vars=["algo", "size"],
    value_vars=["time1", "time2", "time3", "time4", "time5"],
    var_name="thread_col",
    value_name="runtime",
)
tidy["threads"] = tidy["thread_col"].str.extract(r"(\d+)").astype(int)
tidy = tidy.drop(columns="thread_col")
tidy = tidy.dropna(subset=["runtime"])
tidy["algo"] = tidy["algo"].astype("category")
tidy["size"] = tidy["size"].astype("int64")
tidy["threads"] = tidy["threads"].astype("int64")
tidy["runtime"] = tidy["runtime"].astype("float64")
tidy = tidy.sort_values(["algo", "threads", "size"]).reset_index(drop=True)
print("\nTidy shape:", tidy.shape)
print(tidy.head(10))

# Chartfor runtime vs problem size
fig, ax = plt.subplots(figsize=(9, 5))
# single algorithm
single = tidy[tidy["algo"] == "single"]
ax.plot(single["size"], single["runtime"], marker="o",
        label="single", color="black", linewidth=2)
# distributed
dist = tidy[tidy["algo"] == "distributed"]
colors = plt.cm.Blues(np.linspace(0.4, 1.0, dist["threads"].nunique()))

for color, (n_threads, group) in zip(colors, dist.groupby("threads")):
    group_sorted = group.sort_values("size")
    ax.plot(group_sorted["size"], group_sorted["runtime"],
            marker="o", label=f"distributed {n_threads}t", color=color)

ax.set_xscale("log", base=4)
ax.set_yscale("log")
ax.set_xlabel("Problem size (pixels)", fontsize=12)
ax.set_ylabel("Runtime (seconds)", fontsize=12)
ax.set_title("Runtime vs problem size", fontsize=13)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("exercise_1_2_size.png", dpi=150)
plt.show()

# Runtime vs number of threads
# ideal is runtime(n) = runtime(1)/n
fig, ax = plt.subplots(figsize=(8, 5))

sizes = dist["size"].unique()
colors = plt.cm.viridis(np.linspace(0, 0.85, len(sizes)))

for color, size in zip(colors, sorted(sizes)):
    group = dist[dist["size"] == size].sort_values("threads")
    ax.plot(group["threads"], group["runtime"],
            marker="o", color=color, label=f"size {size}")
    t1 = group[group["threads"] == 1]["runtime"].values[0]
    ideal_threads = group["threads"].values
    ax.plot(ideal_threads, t1 / ideal_threads,
            linestyle="--", color=color, alpha=0.5)

ax.plot([], [], linestyle="-",  color="gray", label="actual")
ax.plot([], [], linestyle="--", color="gray", label="ideal")
ax.set_xlabel("Number of threads", fontsize=12)
ax.set_ylabel("Runtime (seconds)", fontsize=12)
ax.set_title("Runtime vs threads of distributed algorithm", fontsize=13)
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("exercise_1_2_threads.png", dpi=150)
plt.show()
