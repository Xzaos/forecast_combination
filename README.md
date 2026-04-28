# Forecast Combination Framework

Combines multiple imperfect signals into a single composite forecast that is more
accurate and more robust than any individual component. Built on top of the
earnings prediction engine from Project 1.

## Core idea

No single model or signal dominates across all market regimes. This project takes
5–10 diverse signals (momentum, value, quality, earnings revision, mean-reversion,
short interest proxies) and combines them using methods ranging from equal weighting
to regime-conditional Bayesian model averaging.

The intellectual core is **bias identification and correction** — analyst optimism,
look-ahead bias, survivorship bias, recency bias in model fitting, and overfitting
of combination weights.

## Data sources (all free)

- **Project 1 pipeline**: SUE signal, fundamental features, price data (EDGAR + yfinance)
- **FRED**: Macro variables for regime identification (yield curve slope, credit spreads,
  unemployment rate, economic surprise proxies)
- **CBOE via yfinance**: VIX for regime features
- **AQR Data Library**: Published factor returns (value, momentum, quality, low-risk)
  at https://www.aqr.com/Insights/Datasets — CSV downloads, no API key needed
- **yfinance**: Price/volume data for constructing momentum, mean-reversion, and
  volatility signals at the stock level

## Universe

Same as Project 1: S&P 500 constituents, 2010–present, point-in-time membership.
~40 quarters × 500 names ≈ 20,000 firm-quarter observations.

## Architecture

```
forecast_combination/
├── src/
│   ├── signals.py          # construct each base signal (the signal library)
│   ├── signal_diagnostics.py  # IC time series, turnover, factor exposures
│   ├── bias_correction.py  # analyst optimism, winsorize, shrinkage
│   ├── combination.py      # equal-weight, inv-vol, ridge, BMA
│   ├── regime.py           # HMM on macro vars, regime detection
│   ├── walkforward.py      # strict out-of-sample evaluation engine
│   ├── macro_data.py       # FRED + VIX pulls
│   └── aqr_data.py         # AQR factor returns loader
├── tests/
│   ├── test_signals.py
│   ├── test_combination.py
│   ├── test_regime.py
│   └── test_walkforward.py
├── notebooks/
│   ├── 01_signal_library.ipynb
│   ├── 02_signal_diagnostics.ipynb
│   ├── 03_bias_analysis.ipynb
│   ├── 04_combination_methods.ipynb
│   ├── 05_regime_detection.ipynb
│   └── 06_walkforward_evaluation.ipynb
├── data/
│   ├── raw/        # FRED downloads, AQR CSVs, cached API responses
│   ├── interim/    # individual signal panels
│   └── processed/  # combined signal matrix, regime labels
└── docs/
    └── signal_dictionary.md
```

## Signal library (target: 8–10 signals)

| # | Signal | Source | Information type | Freq |
|---|--------|--------|-----------------|------|
| 1 | Time-series SUE | Project 1 | Earnings surprise | Q |
| 2 | Earnings revision (proxy) | yfinance consensus | Analyst sentiment | Q |
| 3 | Price momentum (12-1) | yfinance | Trend | M |
| 4 | Short-term reversal (1m) | yfinance | Mean-reversion | M |
| 5 | Earnings yield (E/P) | EDGAR + yfinance | Value | Q |
| 6 | ROE stability | EDGAR | Quality | Q |
| 7 | Accruals (Sloan) | Project 1 | Quality/fraud | Q |
| 8 | Volatility (realized 60d) | yfinance | Risk | M |
| 9 | AQR value factor loading | AQR + yfinance | Value (style) | M |
|10 | AQR momentum factor loading | AQR + yfinance | Momentum (style) | M |

Signals must be diversified in the information they draw on. The ten above span:
accounting data, price data, analyst estimates, and published factor returns.

## Combination methods (in order of complexity)

1. **Equal weight** — benchmark, surprisingly hard to beat OOS
2. **Inverse-IC-volatility** — weight ∝ 1/σ(IC) over trailing window
3. **OLS** — regress realized returns on all signals; coefficients = weights
4. **Ridge regression** — regularized OLS, penalizes extreme weights
5. **Elastic net** — L1+L2 penalty, allows some signals to be zeroed out
6. **Bayesian Model Averaging** — prior probabilities on each signal, updated
   by recent performance
7. **Regime-conditional weights** — HMM detects regime, separate weights per regime

## Regime model

Hidden Markov Model on 3–4 macro features:
- VIX level
- Yield curve slope (10Y - 2Y Treasury)
- Credit spread (BAA - AAA)
- Unemployment rate or economic surprise index

Two or three latent states (low-vol trending, high-vol mean-reverting, crisis).
Regime labels are used to condition combination weights — a momentum signal that
works in regime 1 may be destructive in regime 2.

## Evaluation

Walk-forward only. At each quarterly step:
1. Estimate combination weights using only data up to time t
2. Apply weights to signals at time t to produce composite forecast
3. Measure IC of composite vs realized outcome at t+1

Metrics:
- **Composite IC** (Spearman rank correlation, quarterly)
- **IC improvement** over best single signal
- **IC stability** (lower std across quarters = better)
- **Quintile spread** (long top quintile, short bottom)
- **Regime-conditional IC** (does the composite work in all regimes?)
- **Turnover** (how much does the composite ranking change quarter to quarter?)

## Build order (for Claude Code prompts)

1. `src/macro_data.py` + `src/aqr_data.py` — data pulls
2. `src/signals.py` — construct all 10 signals
3. `src/signal_diagnostics.py` — IC, turnover, exposures
4. `src/bias_correction.py` — winsorize, shrinkage, optimism adjustment
5. `src/combination.py` — all combination methods
6. `src/regime.py` — HMM fitting and regime labeling
7. `src/walkforward.py` — the evaluation engine
8. Notebooks 01–06 — research narrative

## Bias corrections (the intellectual core)

**Analyst optimism:** Sell-side systematically overestimates growth. Correct by
subtracting the rolling median forecast error from each consensus estimate before
computing the revision signal.

**Look-ahead bias:** Use only information available at each point in time. All
signals must be lagged by at least one period relative to the evaluation target.
The walk-forward engine enforces this structurally.

**Survivorship bias:** Use point-in-time S&P 500 membership (from Project 1's
`universe.py`). Firms that delisted are included for the quarters they were alive.

**Recency bias in fitting:** Use at least 20 quarters of rolling history for
weight estimation. Shorter windows overfit to the most recent regime.

**Overfitting of weights:** Ridge/elastic-net regularization. Compare every
sophisticated method to equal-weight benchmark. If it doesn't beat equal-weight
OOS, don't use it.

## Dependencies

- Everything from Project 1 (pandas, numpy, scikit-learn, yfinance, etc.)
- `fredapi` — FRED data pulls (requires free API key from fred.stlouisfed.org)
- `hmmlearn` — Hidden Markov Models
- `statsmodels` — for OLS and diagnostic regressions
- `scipy` — optimization, statistics

## Citations

- Bates, J. M., & Granger, C. W. J. (1969). The combination of forecasts.
  *Operational Research Quarterly*, 20(4), 451–468.
- Timmermann, A. (2006). Forecast combinations. *Handbook of Economic
  Forecasting*, 1, 135–196.
- Hamilton, J. D. (1989). A new approach to the economic analysis of
  nonstationary time series and the business cycle. *Econometrica*, 57(2), 357–384.
- Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased estimation
  for nonorthogonal problems. *Technometrics*, 12(1), 55–67.
