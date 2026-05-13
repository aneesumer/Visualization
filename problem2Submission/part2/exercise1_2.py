import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from sklearn.kernel_ridge import KernelRidge

#1. Load & type-convert 
from pathlib import Path
DATA_DIR = Path(__file__).parent
df = pd.read_csv(DATA_DIR / 'data_astroids.csv')
df['first_obs'] = pd.to_datetime(df['first_obs'])   
df['class'] = df['class'].astype('category')        
df = df.dropna(subset=['H', 'first_obs', 'a', 'diameter'])

#2. H vs first_obs (coloured by a) 
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(df['first_obs'], df['H'], c=df['a'], cmap='viridis',
                s=6, alpha=0.5, linewidths=0)
cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Semi-major axis a (AU)')
ax.set_xlabel('Date of first observation')
ax.set_ylabel('Absolute magnitude H')
ax.set_title('H vs. First Observation, coloured by semi-major axis a')
ax.xaxis.set_major_locator(mdates.YearLocator(5))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig('plot_H_firstobs_a.png', dpi=150)

#3. Split into 5 equal sized a-groups via qcut 
df['a_group'] = pd.qcut(df['a'], q=5)
epoch = pd.Timestamp('1970-01-01')
df['obs_days'] = (df['first_obs'] - epoch).dt.days.astype(float)

#4. Grouped plot + kernel regression (RBF kernel ridge)
colors = ['#e41a1c', '#ff7f00', '#4daf4a', '#377eb8', '#984ea3']
fig2, ax2 = plt.subplots(figsize=(11, 7))

for (name, grp), color in zip(df.groupby('a_group', observed=True), colors):
    # scatter
    ax2.scatter(grp['first_obs'], grp['H'], c=color, s=5, alpha=0.35,
                linewidths=0)

    
    X = grp['obs_days'].values.reshape(-1, 1)
    y = grp['H'].values
    X_min, X_max = X.min(), X.max()
    X_n = (X - X_min) / (X_max - X_min)          # normalise to [0,1]

    kr = KernelRidge(kernel='rbf', gamma=5, alpha=0.5)
    kr.fit(X_n, y)

    x_grid = np.linspace(X_min, X_max, 300).reshape(-1, 1)
    x_grid_n = (x_grid - X_min) / (X_max - X_min)
    y_pred = kr.predict(x_grid_n)
    dates_grid = pd.to_datetime(x_grid.ravel(), unit='D', origin='unix')
    ax2.plot(dates_grid, y_pred, color=color, lw=2.0, label=f'a ∈ {name}')

ax2.set_xlabel('Date of first observation')
ax2.set_ylabel('Absolute magnitude H')
ax2.set_title('H vs. First Observation by semi-major axis group\n(lines = kernel regression)')
ax2.xaxis.set_major_locator(mdates.YearLocator(5))
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig2.autofmt_xdate()
legend_elements = [
    Line2D([0], [0], color=c, lw=2, marker='o', markersize=5,
           label=f'a ∈ {n}', markerfacecolor=c, alpha=0.8)
    for (n, _), c in zip(df.groupby('a_group', observed=True), colors)
]
ax2.legend(handles=legend_elements, title='Semi-major axis group (AU)',
           fontsize=8, title_fontsize=9, loc='upper left')
fig2.tight_layout()
fig2.savefig('plot_grouped_regression.png', dpi=150)

#5. diameter vs H — logarithmic y-axis 
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(df['H'], df['diameter'], s=5, alpha=0.3, linewidths=0,
            color='steelblue')
ax3.set_yscale('log')
ax3.set_xlabel('Absolute magnitude H')
ax3.set_ylabel('Diameter (km, log scale)')
ax3.set_title('Diameter vs. Absolute Magnitude H\n(logarithmic diameter axis)')
ax3.grid(True, which='both', alpha=0.3, linestyle='--')
fig3.tight_layout()
fig3.savefig('plot_diameter_H.png', dpi=150)

plt.show()
