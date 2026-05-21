import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import norm
from pathlib import Path

rng = np.random.default_rng(seed=42)

def sample_XN(M: int, N: int, rng=rng) -> np.ndarray:
    xi = rng.choice([-1, 1], size=(M, N))  
    return xi.sum(axis=1) / np.sqrt(N)     

M  = 5000
Ns = [1, 3, 10, 30, 100]

BG      = "#0f0e17"
TEXT    = "#fffffe"
GRID    = "#1e1e2e"
NORMAL  = "#ff8906"
COLORS  = ["#4cc9f0", "#4361ee", "#a855f7", "#f72585", "#e63946"]

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
    "text.color": TEXT, "grid.color": GRID,
    "font.family": "DejaVu Sans",
})

fig = plt.figure(figsize=(20, 11))
fig.patch.set_facecolor(BG)
gs = gridspec.GridSpec(2, 6, figure=fig,
                       hspace=0.55, wspace=0.35,
                       left=0.05, right=0.97, top=0.90, bottom=0.08)

x_grid     = np.linspace(-4, 4, 600)
normal_pdf = norm.pdf(x_grid)


def gaussian_kde(samples, x_grid):
    n = len(samples)
    h = 1.06 * samples.std(ddof=1) * n**(-1/5)
    kde = np.mean(norm.pdf((x_grid[:, None] - samples[None, :]) / h), axis=1) / h
    return kde


for k, (N, col) in enumerate(zip(Ns, COLORS)):
    ax = fig.add_subplot(gs[0, k])
    samples = sample_XN(M, N)
    kde     = gaussian_kde(samples, x_grid)

    ax.hist(samples, bins=40, density=True, color=col, alpha=0.22, linewidth=0)
    ax.fill_between(x_grid, kde, alpha=0.16, color=col)
    ax.plot(x_grid, kde, color=col, linewidth=2.0, label="KDE")
    ax.plot(x_grid, normal_pdf, color=NORMAL, linewidth=1.6,
            linestyle="--", alpha=0.9, label="N(0,1)")

    ax.set_xlim(-4, 4)
    ax.set_ylim(bottom=0)
    ax.set_title(f"N = {N}", fontsize=12, fontweight="bold", color=col, pad=6)
    ax.set_xlabel("$X_N$", fontsize=9)
    if k == 0:
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7, framealpha=0.15, loc="upper left")
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", linewidth=0.4, alpha=0.4)
    ax.spines[["top", "right"]].set_visible(False)


ax_all = fig.add_subplot(gs[1, :4])
for N, col in zip(Ns, COLORS):
    samples = sample_XN(M, N)
    kde = gaussian_kde(samples, x_grid)
    ax_all.plot(x_grid, kde, color=col, linewidth=2.2, label=f"N = {N}")

ax_all.plot(x_grid, normal_pdf, color=NORMAL, linewidth=2.2,
            linestyle="--", label="N(0,1)  [limit]")
ax_all.set_xlim(-4, 4)
ax_all.set_xlabel("$X_N$", fontsize=10)
ax_all.set_ylabel("Density", fontsize=10)
ax_all.set_title("KDE of $X_N$ for all N  —  convergence to N(0,1)",
                 fontsize=12, fontweight="bold", color=TEXT, pad=8)
ax_all.legend(fontsize=9, framealpha=0.15, ncol=3)
ax_all.grid(axis="y", linewidth=0.4, alpha=0.4)
ax_all.spines[["top", "right"]].set_visible(False)
ax_all.tick_params(labelsize=8)


ax_txt = fig.add_subplot(gs[1, 4:])
ax_txt.axis("off")


lines = [
    "Central Limit Theorem",
    "",
    r"$X_N = \frac{1}{\sqrt{N}}\sum_{i=1}^{N} x_i$",
    "",
    r"$x_i \in \{-1,+1\}$ with prob $\frac{1}{2}$",
    r"$\mathrm{E}[x_i]=0$,   $\mathrm{Var}(x_i)=1$",
    "",
    "As $N \\to \\infty$:",
    r"  $X_N \overset{d}{\longrightarrow} \mathcal{N}(0,1)$",
    "",
    f"Samples per curve:  M = {M:,}",
    "Density estimator: Gaussian KDE",
    "(Silverman bandwidth rule)",
]
note = "\n".join(lines)

ax_txt.text(0.05, 0.95, note, transform=ax_txt.transAxes,
            fontsize=9.5, va="top", ha="left", color=TEXT, linespacing=1.7,
            bbox=dict(boxstyle="round,pad=0.7", facecolor="#1e1e2e",
                      edgecolor=NORMAL, linewidth=1.3, alpha=0.88))

fig.suptitle(
    "Visualising the Central Limit Theorem via Kernel Density Estimation",
    fontsize=14, fontweight="bold", color=TEXT, y=0.97)

OUT = Path(__file__).parent / "fig_clt.png"
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=BG)
print(f"Saved -> {OUT}")