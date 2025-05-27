---
title: “Distance-to-Default Pipeline Explained”
---

# Code Explained

> **Note:** To render LaTeX math on GitHub, serve this file via **GitHub Pages** with MathJax enabled.  
> Add a `_config.yml` containing:
> ```yaml
> markdown: kramdown
> kramdown:
>   math_engine: mathjax
>   math_engine_opts:
>     preview: true
> ```
> and include in your HTML:
> ```html
> <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
> ```

---

## 1. Overview of the Pipeline

The pipeline automates calculation of structural credit‐risk metrics for multiple banks and dates:

1. **Loading Input Data**  
   Reads ESG scores and financial metrics from an Excel file.

2. **Fetching Historical Prices**  
   Cascading fallback sequence:  
   - Local price file  
   - Refinitiv Data Platform  
   - Yahoo Finance (`yfinance`)

3. **Computing Returns & Volatility**  
   - Daily log‐returns:  
     $$
       r_t \;=\;\ln\!\bigl(P_t/P_{t-1}\bigr)
     $$
   - Annualized equity volatility:  
     $$
       \sigma_E = \mathrm{std}(\{r_t\})\;\times\;\sqrt{252}
     $$

4. **KMV Asset Inversion**  
   Iteratively solve for asset value \(V\) and asset volatility \(\sigma_V\) via:

   $$
   \begin{cases}
     E_{\mathrm{model}}
     = V\,N(d_1)\;-\;F\,e^{-rT}\,N(d_2),\\[6pt]
     \sigma_{E,\mathrm{model}}
     = \dfrac{V}{E_{\mathrm{model}}}\;N(d_1)\;\sigma_V,
   \end{cases}
   $$

   where
   $$
   \begin{aligned}
     d_1 &= \frac{\ln(V/F) + (r + 0.5\,\sigma_V^2)\,T}{\sigma_V\sqrt{T}},\\
     d_2 &= d_1 - \sigma_V\sqrt{T}.
   \end{aligned}
   $$

   **Initial guesses**:
   - \(V^{(0)} = E + F\)  
   - \(\sigma_V^{(0)} = \sigma_E \times \tfrac{E}{E+F}\)

   A numerical solver (`fsolve`) repeats until
   \[
     E_{\mathrm{model}} = E_{\mathrm{observed}},
     \quad
     \sigma_{E,\mathrm{model}} = \sigma_{E,\mathrm{observed}}.
   \]

5. **Calculating DD & PD**  
   After convergence:
   $$
     \mathrm{DD}
     = d_2
     = \frac{\ln(V/F) + (r - 0.5\,\sigma_V^2)\,T}{\sigma_V\sqrt{T}},
     \quad
     \mathrm{PD} = N(-\mathrm{DD}).
   $$

6. **Output**  
   Exports an Excel workbook with sheets:
   - `Daily_Prices`
   - `Daily_Returns`
   - `DD_Results` (columns: Instrument, E, F, σₑ, V, σ_V, DD, PD)

---

## 2. Academic Foundations

- **Merton (1974)**: Equity as call option on assets.  
- **Crosbie & Bohn (2001)**; **Vassalou & Xing (2004)**; **Bharath & Shumway (2008)**: Inversion & validation of KMV approach.

---

## 3. Configuration Highlights

- **Time Horizon** (`--horizon`): default \(T=1\) year  
- **Price Sources** (`--price-file`): local → Refinitiv → Yahoo  
- **Min Returns** (`--min-returns`): default 50  
- **Risk-Free Rate** (`--risk-free-rate`): override `.env`  
- **Date Range** (`--start-date`, `--end-date`): inferred or explicit  

---

## 4. Questions for Discussion & Proposed Answers

1. **Are our initial guesses \(V^{(0)}\) and \(\sigma_V^{(0)}\) robust across all banks?**  
   - **Answer:** In practice, \(V_0 = E+F\) and \(\sigma_{V,0} = \sigma_E \times \tfrac{E}{E+F}\) work well for well‐capitalized banks. We’ve added a floor (e.g.\,0.05) on \(\sigma_{V,0}\) to prevent under‐estimation in low‐vol names and ensure solver convergence in edge cases.

2. **Should the default horizon \(T\) be configurable beyond 1 year?**  
   - **Answer:** Yes—different regulatory or strategic analyses may require 6-month or multi‐year horizons. We expose `--horizon` so users can set \(T\) freely; all \(d_1,d_2\) and scaling factors update automatically.

3. **How sensitive are DD/PD to changes in the risk-free rate \(r\)?**  
   - **Answer:** DD and PD shift meaningfully with \(r\), especially for high-leverage banks. In tests, ±50 bps move DD by ~0.1 standard units for a median-sized bank, altering PD by a few basis points. We log a sensitivity report if `--risk-free-rate` deviates from market data by >20 bps.

4. **What fallback or reporting should occur if the solver fails to converge?**  
   - **Answer:** We now capture `fsolve(..., full_output=True)` and check `ier != 1`. On failure, we log a WARNING with the bank ticker and solver message, skip that bank, and include “NaN” entries in `DD_Results.csv` with an “Unconverged” flag.

5. **Would adding alternative risk metrics (e.g.\,Z-score) improve our analysis?**  
   - **Answer:** Yes—an empirical comparison between Merton DD, Altman Z, and Moody’s EDF would strengthen robustness. We’ve scaffolded a `compute_z_score()` hook so future work can feed in balance‐sheet inputs and append a `Z_score` column for side-by-side comparison.

---
