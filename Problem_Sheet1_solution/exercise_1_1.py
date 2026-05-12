import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats

# Loading with explicit dtypes
dtypes = {
    "manufacturer": "category",
    "model":        "category",
    "displ":        "float64",
    "year":         "int64",
    "cyl":          "int64",
    "trans":        "category",
    "drv":          "category",
    "cty":          "int64",
    "hwy":          "int64",
    "fl":           "category",
    "class":        "category",
}

df = pd.read_csv("mpg-data.csv", dtype=dtypes)
print(df.dtypes)

# Linear regression per class
regressions = {}
for car_class, group in df.groupby("class", observed=True):
    result = scipy.stats.linregress(group["displ"], group["hwy"])
    regressions[car_class] = result
    print(f"{car_class:12s}  slope={result.slope:.2f}  intercept={result.intercept:.2f}  r²={result.rvalue**2:.3f}")

# Scatter plot and regression lines
classes = df["class"].cat.categories.tolist()
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_map = dict(zip(classes, colors))
fig, ax = plt.subplots(figsize=(9, 6))

for car_class in classes:
    group = df[df["class"] == car_class]
    c = color_map[car_class]
    ax.scatter(group["displ"], group["hwy"], c=c,
               marker="x", label=car_class)
    x_min, x_max = group["displ"].min(), group["displ"].max()
    x_line = np.linspace(x_min, x_max, 100)
    reg = regressions[car_class]
    y_line = reg.slope * x_line + reg.intercept
    ax.plot(x_line, y_line, color=c, linewidth=1.8)

ax.set_xlabel("Engine displacement (litres)", fontsize=12)
ax.set_ylabel("Highway fuel efficiency (mpg)", fontsize=12)
ax.set_title("Highway efficiency vs displacement by car class", fontsize=13)
ax.legend(title="Class", bbox_to_anchor=(1.01, 1), loc="upper left")
plt.tight_layout()
plt.savefig("exercise_1_1_scatter.png", dpi=150)
plt.show()

# Grouped median table
median_table = (
    df.groupby(["class", "year"], observed=True)["hwy"]
    .median()
    .rename("median_hwy")
    .reset_index()
)

print("\nMedian highway mpg by class and year")
print(median_table.to_string(index=False))
