# Code Explained

This document provides a clear, step-by-step narrative of the distance-to-default (DD) pipeline implementation, designed to be accessible to financial specialists.

---

## 1. Overview of the Pipeline

The pipeline automates calculation of structural credit risk metrics for multiple banks over time by:

1. **Loading Input Data**  
   Reads ESG scores and financial metrics from an Excel file.

2. **Fetching Historical Prices**  
   Obtains price data via a cascading fallback:
   - Local price file  
   - Refinitiv Data Platform  
   - Yahoo Finance (`yfinance`)

3. **Computing Returns and Volatility**  
   - Daily log-returns:  
     ```math
     r_t = \ln\bigl(P_t / P_{t-1}\bigr)
     ```  
   - Annualized equity volatility:  
     ```math
     \sigma_E = \mathrm{std}(\{r_t\}) \times \sqrt{252}
     ```

4. **KMV Asset Inversion**  
   Iteratively solves these two equations for each bank and date:

   **(1) Equity Value Equation**  
   $$
   E_{\mathrm{model}}
   = V\,N(d_1)
   - F e^{-rT}\,N(d_2)
   $$

   **(2) Equity Volatility Equation**  
   $$
   \sigma_{E,\mathrm{model}}
   = \frac{V}{E_{\mathrm{model}}}\,N(d_1)\,\sigma_V
   $$

   where  
   $$
   d_1 = \frac{\ln(V/F) + (r + 0.5\,\sigma_V^2)\,T}{\sigma_V\sqrt{T}},\quad
   d_2 = d_1 - \sigma_V\sqrt{T}.
   $$

   Initial guesses:
   - \(V^{(0)} = E + F\)  
   - \(\sigma_V^{(0)} = \sigma_E \times \tfrac{E}{E + F}\)

   A numerical solver (`fsolve`) repeats until  
   \(E_{\mathrm{model}} = E_{\mathrm{observed}}\) and  
   \(\sigma_{E,\mathrm{model}} = \sigma_{E,\mathrm{observed}}\).

5. **Calculating DD & PD**  
   After convergence:
   $$
   \mathrm{DD}
   = d_2
   = \frac{\ln(V/F) + (r - 0.5\,\sigma_V^2)\,T}{\sigma_V \sqrt{T}},
   \quad
   \mathrm{PD} = N(-\mathrm{DD}).
   $$

6. **Output**  
   Exports an Excel workbook with three sheets:
   - **Daily_Prices**  
   - **Daily_Returns**  
   - **DD_Results** (with E, F, \(\sigma_E\), V, \(\sigma_V\), DD, PD)

---

## 2. Academic Foundations

- **Merton (1974)**: Treats equity as a call option on firm value.  
- **Crosbie & Bohn (2001)**: KMV-style practical inversion method.  
- **Vassalou & Xing (2004)**: Highlights DD’s link to systematic risk.  
- **Bharath & Shumway (2008)**: Validates Merton DD model forecasts.

---

## 3. Configuration Highlights

- **Time Horizon** (`--horizon`): default \(T=1\) year.  
- **Price Sources** (`--price-file`): local → Refinitiv → Yahoo.  
- **Min Returns** (`--min-returns`): default 50 observations.  
- **Risk-Free Rate** (`--risk-free-rate`): override `.env` default.  
- **Date Range** (`--start-date`, `--end-date`): inferred or explicit.

---

## 4. Discussion Questions

1. Are initial guesses \(V^{(0)}\) and \(\sigma_V^{(0)}\) suitable across banks?  
2. Should the default horizon \(T\) remain fixed at one year?  
3. How sensitive are results to the risk-free rate \(r\)?  
4. What fallbacks for non-convergence should be reported?  
5. Would adding alternative default measures (e.g., Z-score) improve robustness?

---

*End of Document.*
