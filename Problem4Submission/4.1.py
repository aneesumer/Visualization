import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path


HERE = Path(__file__).parent        
CSV  = HERE / "population_us.csv"
OUT1 = HERE / "fig1_pyramids.png"
OUT2 = HERE / "fig2_heatmaps.png"


df = pd.read_csv(CSV)
df["sex_label"] = df["sex"].map({1: "Male", 2: "Female"})

years = sorted(df["year"].unique())   
ages  = sorted(df["age"].unique())    

df["share"] = df["people"] / df.groupby("year")["people"].transform("sum") * 100

MALE_COL   = "#4FC3F7"
FEMALE_COL = "#F48FB1"
BG         = "#0d1117"
TEXT       = "#e6edf3"
GRID       = "#21262d"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor":   GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color":  TEXT, "grid.color":   GRID,
    "font.family": "DejaVu Sans",
})


selected = [1850, 1900, 1950, 2000]
fig1, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)
fig1.suptitle("US Population Pyramids  ·  1850 – 2000",
              fontsize=15, fontweight="bold", color=TEXT, y=1.02)

max_share = df["share"].max() * 1.08

for ax, yr in zip(axes, selected):
    sub = df[df["year"] == yr]
    m = sub[sub["sex"] == 1].set_index("age")["share"].reindex(ages, fill_value=0)
    f = sub[sub["sex"] == 2].set_index("age")["share"].reindex(ages, fill_value=0)

    ax.barh(ages, -m.values, height=4.2, color=MALE_COL,   alpha=0.85, label="Male")
    ax.barh(ages,  f.values, height=4.2, color=FEMALE_COL, alpha=0.85, label="Female")

    ax.set_xlim(-max_share, max_share)
    ax.set_title(str(yr), fontsize=13, fontweight="bold", color=TEXT, pad=8)
    ax.axvline(0, color=TEXT, linewidth=0.7, alpha=0.35)
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)

  
    tick_vals = np.linspace(-max_share, max_share, 5)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{abs(v):.1f}%" for v in tick_vals], fontsize=7)
    ax.set_xlabel("Share of total population", fontsize=8)

    if ax is axes[0]:
        ax.set_yticks(ages)
        ax.set_yticklabels([str(a) for a in ages], fontsize=7)
        ax.set_ylabel("Age group (years)", fontsize=9)

    ax.text(-max_share * 0.5, 93, "← Male",
            ha="center", va="bottom", fontsize=7.5, color=MALE_COL)
    ax.text( max_share * 0.5, 93, "Female →",
            ha="center", va="bottom", fontsize=7.5, color=FEMALE_COL)

fig1.tight_layout()
fig1.savefig(OUT1, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT1}")


def pivot(sex_val):
    return (df[df["sex"] == sex_val]
            .pivot_table(index="age", columns="year", values="share")
            .reindex(index=ages, columns=years))

M_mat = pivot(1)
F_mat = pivot(2)
D_mat = M_mat - F_mat  

fig2 = plt.figure(figsize=(18, 14))
fig2.patch.set_facecolor(BG)
gs = gridspec.GridSpec(3, 2, width_ratios=[22, 1], hspace=0.5, wspace=0.06)

panels = [
    (M_mat, "Male age distribution  (% of total population)",   "Blues",  False),
    (F_mat, "Female age distribution  (% of total population)", "RdPu",   False),
    (D_mat, "Male − Female share  (percentage points)",          "RdBu_r", True),
]

for row, (mat, title, cmap, diverging) in enumerate(panels):
    ax  = fig2.add_subplot(gs[row, 0])
    cax = fig2.add_subplot(gs[row, 1])

    if diverging:
        absmax = np.nanmax(np.abs(mat.values))
        norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                       origin="lower", norm=norm)
    else:
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap,
                       origin="lower", vmin=0, vmax=mat.values.max())

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ages)))
    ax.set_yticklabels(ages, fontsize=7)
    ax.set_xlabel("Year", fontsize=9)
    ax.set_ylabel("Age group (years)", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=TEXT, pad=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)

    cb = fig2.colorbar(im, cax=cax)
    cb.ax.yaxis.set_tick_params(color=TEXT, labelsize=7)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=TEXT)
    cb.set_label("M − F  (pp)" if diverging else "% of population",
                 fontsize=7, color=TEXT)

fig2.suptitle("US Population Age Distribution  ·  1850 – 2000",
              fontsize=14, fontweight="bold", color=TEXT, y=1.01)

fig2.savefig(OUT2, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved → {OUT2}")