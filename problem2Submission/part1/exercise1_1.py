import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

#1. Load data
from pathlib import Path
DATA_DIR = Path(__file__).parent
df = pd.read_csv(DATA_DIR / 'data_sin.csv').dropna()
x = df['x'].values
y = df['y'].values

#Plot 1: Scatter with error bars (σ = 0.2)
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.errorbar(x, y, yerr=0.2, fmt='o', color='steelblue', ecolor='lightblue',
             elinewidth=1.5, capsize=3, markersize=4, label='Observed data (σ = 0.2)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_title('Scatter plot with error bars')
ax1.legend()
ax1.grid(True, alpha=0.3)
fig1.tight_layout()
fig1.savefig('plot1_scatter_errorbars.png', dpi=150)

#2. Gaussian Process Regression 
#Kernel: squared-exponential (RBF) + white noise
kernel = 1.0 * RBF(length_scale=0.1) + WhiteKernel(noise_level=0.04)
gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=9)
X = x.reshape(-1, 1)
gp.fit(X, y)

X_pred = np.linspace(0, 1, 300).reshape(-1, 1)
y_mean, y_std = gp.predict(X_pred, return_std=True)

fig2, ax2 = plt.subplots(figsize=(8, 5))
ax2.errorbar(x, y, yerr=0.2, fmt='o', color='steelblue', ecolor='lightblue',
             elinewidth=1.5, capsize=3, markersize=4, label='Observed data (σ = 0.2)', zorder=3)
ax2.plot(X_pred.ravel(), y_mean, color='darkorange', lw=2, label='GP mean prediction')
ax2.fill_between(X_pred.ravel(),
                 y_mean - 1.96 * y_std,
                 y_mean + 1.96 * y_std,
                 alpha=0.25, color='darkorange', label='GP 95% confidence interval')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Gaussian Process Regression')
ax2.legend()
ax2.grid(True, alpha=0.3)
fig2.tight_layout()
fig2.savefig('plot2_gp_regression.png', dpi=150)

#3. Rolling mean and standard deviation
window = 5
roll_x   = df['x'].rolling(window, center=True).mean()
roll_y   = df['y'].rolling(window, center=True).mean()
roll_std = df['y'].rolling(window, center=True).std()

fig3, ax3 = plt.subplots(figsize=(8, 5))
ax3.errorbar(x, y, yerr=0.2, fmt='o', color='steelblue', ecolor='lightblue',
             elinewidth=1.5, capsize=3, markersize=4, label='Observed data (σ = 0.2)', zorder=3)
ax3.plot(roll_x, roll_y, color='forestgreen', lw=2, label='Rolling mean (window = 5)')
ax3.fill_between(roll_x,
                 roll_y - roll_std,
                 roll_y + roll_std,
                 alpha=0.25, color='forestgreen', label='Rolling ±1 std dev (window = 5)')
ax3.set_xlabel('x')
ax3.set_ylabel('y')
ax3.set_title('Rolling Mean and Standard Deviation (window = 5)')
ax3.legend()
ax3.grid(True, alpha=0.3)
fig3.tight_layout()
fig3.savefig('plot3_rolling.png', dpi=150)

plt.show()
