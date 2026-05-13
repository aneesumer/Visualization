# Plot Descriptions

## Plot 1 — Scatter Plot with Error Bars

The data is loaded using `pd.read_csv`. Since the noise standard deviation is known (`σ = 0.2`), the uncertainty for each observation is directly visualized using:

```python
ax.errorbar(..., yerr=0.2)
```

This represents the **±1σ uncertainty** around every data point.

---

## Plot 2 — Gaussian Process Regression

A **Gaussian Process (GP)** model is fitted using:

- an **RBF kernel** to capture smooth underlying trends, and
- a **WhiteKernel** to model observation noise.

Predictions are generated at 300 evenly spaced points using:

```python
gp.predict(..., return_std=True)
```

This returns:

- the **posterior mean**, and
- the **posterior standard deviation**.

The shaded confidence region is computed as:

```python
mean ± 1.96 × σ
```

which corresponds to the **95% confidence interval** of the estimated mean function.

### Characteristics of the GP uncertainty band

- **Shrinks** in regions where training data is dense
- **Expands** in sparse or extrapolated regions

---

## Plot 3 — Rolling Mean and Standard Deviation

Rolling statistics are computed using:

```python
df.rolling(window=5, center=True)
```

This applies a sliding window of **5 consecutive points**, centered on each row.

The shaded region represents:

```python
rolling_mean ± rolling_std
```

Unlike the GP confidence interval, this band reflects the **local variability of the observed data itself** rather than uncertainty in a predictive model.

### Characteristics of the rolling statistics band

- Wider in noisier regions
- Directly dependent on local fluctuations in the data
- Does **not** naturally shrink near boundaries or dense regions like GP uncertainty does