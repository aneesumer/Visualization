# Data Processing and Plots

## 1. Type Conversions

- `first_obs` is parsed using `pd.to_datetime()`.  
- Its class is then cast to **category**.  
- All rows with missing values in the key columns are dropped, leaving **all 20,000 rows intact**.

---

## 2. H vs `first_obs` coloured by `a`

A scatter plot is created with:

- **x-axis:** `first_obs`  
- **y-axis:** `H`  
- **Colour:** semi-major axis `a` (using the **viridis** colormap)

**Observations:**

- Objects with larger `a` (outer belt, purple/yellow) tend to have **lower H** (brighter/larger).  
- Most discoveries occur **after ~2000**, as survey programs ramped up.

---

## 3 & 4. Grouped Regression

- `pd.qcut(df['a'], q=5)` divides the 20,000 objects into **5 equally-populated bins** by `a`.  
- For each group, a **Kernel Ridge Regression** with an **RBF kernel** is fitted to `(first_obs → H)`.  
- This acts as a **smooth local regression**, similar to LOESS.

**Trend revealed by regression lines:**

- Asteroids discovered **later** tend to have **higher H** (fainter/smaller).  
- Large, bright objects were found first; surveys progressively target **smaller objects** over time.

---

## 5. Diameter vs H — Log Scale

- Diameter spans nearly **4 orders of magnitude** (0.008 km to 270 km), so a **logarithmic y-axis** is essential.  
- The plot shows the expected **inverse power-law relationship**:  

  - Higher H (fainter) → smaller diameter  
  - Forms a tight, nearly linear band on the log-linear scale