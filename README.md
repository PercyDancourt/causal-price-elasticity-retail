# Causal Price Elasticity of Demand — Dominick's Cereal
### A Double Machine Learning & Causal Forest Analysis of Scanner Data

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![econml](https://img.shields.io/badge/econml-0.16.0-green.svg)](https://github.com/microsoft/EconML)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project estimates the **causal price elasticity of demand** for ready-to-eat (RTE) cereals using weekly scanner data from **Dominick's Finer Foods** (1989–1997). The core challenge in retail pricing research is that observed price variation is confounded by simultaneous promotional activity — any naive regression of sales on price mixes the true price effect with the promotional advertising effect.

We address this with a two-stage causal inference pipeline:

1. **Double Machine Learning (DML)** — estimates the Average Treatment Effect (ATE) of price on log-units sold, orthogonalizing both variables from a rich confounder matrix (promotions, competitor prices, lagged price dynamics, seasonality, brand, store zone).
2. **Causal Forest DML** — estimates Conditional Average Treatment Effects (CATEs) by price zone, revealing heterogeneous price sensitivity across the store network.
3. **Simulation Validation** — a synthetic Data Generating Process (DGP) with known true elasticity confirms that the DML pipeline correctly recovers causal parameters under confounding, and quantifies how weak overlap breaks identification.

### Key Results

| Analysis / Price Zone | Elasticity | Interpretation |
|---|---|---|
| Naive OLS (Entire Network) | −0.34 | Biased — captures promotional sales spikes |
| DML ATE (Average Shelf) | **−0.088** | Causal shelf elasticity (no promotions) |
| CATE — High price zone | −0.117 | Most price-sensitive stores (higher income/substitution) |
| CATE — Low price zone | −0.104 | Price-sensitive stores |
| CATE — Medium price zone | −0.092 | Average price-sensitive stores |
| CATE — CubFighter price zone | −0.060 | Competitor-focused discount format (lowest sensitivity) |

Our estimates are deliberately lower in magnitude than the canonical literature (Hausman 1997: −0.9 to −2.5; Bijmolt et al. 2005: −2.62 mean), because we isolate **baseline shelf elasticity** — the demand response to a quiet price change with no advertising support — which is the strategically actionable number for everyday pricing decisions.

Since the estimated price elasticity is highly inelastic ($\epsilon > -1.0$) across all price zones, demand is relatively insensitive to price changes. A pricing simulation using the exact power-law demand formula confirms that **any price increase will lead to an increase in overall revenue** (e.g., a differentiated price increase of 3–7% yielded a 4.51% revenue increase across the simulated network).

---

## Project Structure

```
.
├── notebooks/
│   ├── 01_data_preparation.ipynb       # Load, clean, and merge scanner data
│   ├── 02_exploratory_analysis.ipynb   # EDA: price distributions, brand shares, seasonality
│   ├── 03_baseline_naive.ipynb         # OLS baseline and confounding diagnosis
│   ├── 03_5_feature_engineering.ipynb  # Confounder matrix construction
│   ├── 04_dml_estimation.ipynb         # Double Machine Learning ATE estimation
│   ├── 05_causal_forest_cate.ipynb     # Causal Forest CATE by price zone
│   └── 06_simulation_validation.ipynb  # Synthetic DGP validation of the pipeline
│
├── src/
│   ├── data_loader.py                  # Functions to load raw Dominick's files
│   └── feature_engineering.py          # add_features() and BRAND_MAPPING constant
│
├── data/
│   ├── raw/                            # NOT INCLUDED (see Data section)
│   └── processed/                      # NOT INCLUDED (regenerate from notebooks)
│
├── reports/
│   └── figures/
│       └── cate_by_zone.png
│
├── requirements.txt
└── README.md
```

---

## Data

This project uses the **Dominick's Finer Foods** weekly scanner dataset, publicly available through the **Kilts Center for Marketing** at the University of Chicago Booth School of Business.

> **Data not included in this repository.** The raw data files are not redistributable and must be downloaded directly from the source or secondary repositories.

### How to obtain the data

1. Register (free) at the [Kilts Center Data Portal](https://www.chicagobooth.edu/research/kilts/datasets/dominicks)
2. Download the **Cereal** category files:
   - `wcer.csv` — Weekly movement data (price, units, promotions)
   - `upccer.csv` — UPC-level product lookup for Cereal (descriptions, sizes)
3. Download the following metadata files from the public repository `eurostat/dff` (as they are only found inside the PDF manual on the Kilts portal):
   - `stores.csv` — Store price zone attributes
   - `weeks.csv` — Week-level calendar data (holidays, dates)
4. Place all four files in `data/raw/`
5. Run notebooks `01` through `06` in order

---

## Methodology

### Sample Size Rationale
* **Double Machine Learning (DML - Notebook 04):** We run the model on a representative subset of $N=100,000$ transactions. This provides substantial statistical power for estimating the overall ATE while keeping 5-fold cross-fitting and hyperparameter tuning runtimes fast.
* **Causal Forest (CATE - Notebook 05):** We expand the sample to $N=500,000$ observations. Since CATE estimates are conditional on 4 pricing zones (CubFighter, Low, High, Medium), a larger sample ensures enough observations per zone to grow stable forest splitting trees without the risk of dummy variable collisions in $X$.

### Causal Identification Strategy

The core estimand is the causal effect of log-price ($T$) on log-units sold ($Y$), controlling for a confounder matrix $W$ that captures all observable sources of simultaneous price and demand variation:

$$Y = \theta \cdot T + g(W) + \varepsilon$$
$$T = f(W) + \eta$$

**DML** (Chernozhukov et al., 2018) estimates $\hat{\theta}$ from the residualized regression:

$$\tilde{Y} = \theta \cdot \tilde{T} + \varepsilon, \quad \tilde{Y} = Y - \hat{E}[Y|W], \quad \tilde{T} = T - \hat{E}[T|W]$$

where the nuisance functions $\hat{E}[Y|W]$ and $\hat{E}[T|W]$ are estimated with `HistGradientBoostingRegressor` via 5-fold cross-fitting.

### Confounder Matrix $W$

| Variable | Type | Purpose |
|---|---|---|
| `promotion` | Binary | Direct promotional effect on price and sales |
| `log_competitor_price` | Continuous | Cross-price substitution effects |
| `lag_price_change` | Continuous | Price dynamics without leaking treatment ($R^2_T \approx 0.18$) |
| `log_lag_units` | Continuous | Own-brand demand persistence |
| `brand` dummies | Categorical | Brand fixed effects (Kellogg's, Post, General Mills, …) |
| `month × price_zone` | Interaction | Seasonal × format heterogeneity |
| `time_trend` | Continuous | Non-linear temporal drift |
| Holiday indicators | Binary | Thanksgiving, Christmas, Back-to-School |

> **Weak Overlap Note:** Including `log_lag_price` instead of `lag_price_change` pushes the residual variance ratio of $T$ from 0.82 to 0.074 (near-deterministic treatment), causing DML to produce implausible estimates (ATE = −2.48). Notebook 06 documents this failure mode with a synthetic DGP.

---

## Installation

```bash
git clone https://github.com/your-username/Pricing_Elasticity.git
cd Pricing_Elasticity

python3 -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows

pip install -r requirements.txt
```

### Requirements

| Package | Version |
|---|---|
| `pandas` | ≥ 2.0.0 |
| `numpy` | ≥ 1.24.0 |
| `scipy` | ≥ 1.10.0 |
| `scikit-learn` | ≥ 1.2.0 |
| `econml` | == 0.16.0 |
| `statsmodels` | ≥ 0.14.0 |
| `matplotlib` | ≥ 3.7.0 |
| `seaborn` | ≥ 0.12.0 |
| `jupyter` | ≥ 1.0.0 |

---

## Reproducing the Results

```bash
# Start Jupyter
jupyter notebook

# Run in order:
# 01_data_preparation → 02_exploratory_analysis → 03_baseline_naive
# → 03_5_feature_engineering → 04_dml_estimation
# → 05_causal_forest_cate → 06_simulation_validation
```

Each notebook is self-contained and includes markdown cells explaining every modeling decision.

---

## References

- **Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., & Robins, J.** (2018). Double/debiased machine learning for treatment and structural parameters. *The Econometrics Journal*, 21(1), C1–C68.

- **Hausman, J.A.** (1997). Valuation of new goods under perfect and imperfect competition. In T.F. Bresnahan & R.J. Gordon (Eds.), *The Economics of New Goods*, NBER Studies in Income and Wealth, Vol. 58 (pp. 209–248). University of Chicago Press.

- **Bijmolt, T.H.A., van Heerde, H.J., & Pieters, R.G.M.** (2005). New empirical generalizations on the determinants of price elasticity. *Journal of Marketing Research*, 42(2), 141–156.

- **Nijs, V.R., Dekimpe, M.G., Steenkamp, J.B.E.M., & Hanssens, D.M.** (2001). The category-demand effects of price promotions. *Marketing Science*, 20(1), 1–22.

- **Microsoft Research** (2023). EconML: A Python Package for ML-Based Heterogeneous Treatment Effects Estimation. [https://github.com/microsoft/EconML](https://github.com/microsoft/EconML)

---

## License

MIT License .

The Dominick's dataset is subject to the terms of use of the Kilts Center for Marketing, University of Chicago Booth School of Business.
