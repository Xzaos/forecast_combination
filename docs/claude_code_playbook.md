# Claude Code Prompt Playbook — Forecast Combination

Follow these steps in order. Each step is a prompt to paste into Claude Code.
Commit after each step passes its tests.

---

## Prerequisites

This project depends on Project 1's pipeline (earnings_prediction). You need:
- A working `src/targets.py` (SUE computation)
- A working `src/features_fund.py` (Sloan accruals, fundamentals)
- A working `src/prices_pull.py` (yfinance price data)
- A working `src/universe.py` (S&P 500 membership)

If Project 1 isn't complete yet, you can still build and test this scaffold
using synthetic data — every test in this repo uses fixtures, not real data.

---

## Step 0: Setup

```
cd ~/Desktop/forecast_combination
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
git init && git add . && git commit -m "Initial scaffold"
```

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html.
Create a `.env` file: `FRED_API_KEY=your_key_here`

---

## Step 1: Verify the scaffold

Prompt to Claude Code:

> Read `README.md` and `docs/signal_dictionary.md` to orient yourself.
> Then run the full test suite: `python -m pytest tests/ -v`.
> All tests should pass on synthetic data. If any fail, fix them.
> Do not change the test assertions — only fix the source modules.
> Stop and report results.

Expected: all tests pass. Commit: `git add . && git commit -m "All scaffold tests pass"`

---

## Step 2: Build macro data puller

Prompt:

> Build `src/macro_data.py`. Read the existing scaffold in `src/macro_data.py`
> for the spec. Then actually test it end-to-end:
>
> 1. Load my FRED API key from the `.env` file (use `python-dotenv`).
> 2. Run `build_macro_panel()` and verify it returns a DataFrame with
>    columns [yield_curve, credit_spread, vix, unemployment], monthly
>    frequency, starting from 2005.
> 3. Print the shape, first 5 rows, and last 5 rows.
> 4. Check for gaps: are there any months with NaN after forward-fill?
> 5. Save to `data/raw/macro_panel.parquet`.
>
> Keep the code minimal — the scaffold already has the structure,
> you just need to make sure it actually runs with real data.

Commit: `git add . && git commit -m "Macro data puller verified with real data"`

---

## Step 3: Build signal library (connecting to Project 1)

Prompt:

> We need to connect `src/signals.py` to real data from Project 1.
> The Project 1 repo is at `~/Desktop/earnings_prediction`.
>
> Create a new module `src/data_bridge.py` that:
> 1. Imports from the Project 1 repo (add its path to sys.path)
> 2. Loads the processed data from Project 1: SUE targets, accruals,
>    price data, fundamental features
> 3. Reshapes each into the format expected by `src/signals.py`:
>    - SUE: DataFrame with [ticker, date, sue]
>    - Prices: DataFrame with [ticker, date, adj_close]
>    - Fundamentals: DataFrame with [ticker, date, eps_trailing_4q, roe]
>    - Accruals: DataFrame with [ticker, date, accruals_scaled]
> 4. If Project 1 data isn't available yet, fall back to generating
>    synthetic data with the same schema so the pipeline can still run.
>
> Keep it simple. No complex transformations — just load and reshape.
> Write a test in tests/test_data_bridge.py that verifies the output schemas.

Commit: `git add . && git commit -m "Data bridge to Project 1"`

---

## Step 4: Build AQR factor loader

Prompt:

> Look at `src/aqr_data.py`. The fallback_factors function uses ETF proxies
> as crude factor approximations. Test it:
>
> 1. Run `_fallback_factors()` and verify it returns a DataFrame with
>    columns [HML, UMD] at monthly frequency.
> 2. Print shape and summary statistics.
> 3. If it works, we'll use this as our factor data for now. If/when I
>    download actual AQR CSVs, the `load_aqr_factors()` function will
>    pick them up automatically.
>
> Keep the code minimal. Don't add complexity.

Commit: `git add . && git commit -m "AQR factor loader tested"`

---

## Step 5: Signal diagnostics on real data

Prompt:

> Create `notebooks/02_signal_diagnostics.ipynb`. For each of the 10 signals
> in the signal dictionary:
>
> 1. Construct the signal using `src/signals.py` and data from `src/data_bridge.py`
> 2. Compute IC time series using `src/signal_diagnostics.py`
> 3. Plot IC over time (matplotlib, simple line chart)
> 4. Print IC summary (mean, std, IR, pct positive)
> 5. Compute turnover
>
> Use the outcome = next-quarter stock return (simple forward return from yfinance).
>
> At the end of the notebook, create a summary table showing all 10 signals'
> IC stats side by side. This tells us which signals are worth combining and
> which are noise.
>
> Keep plots simple. No fancy formatting. The goal is information, not aesthetics.

Commit: `git add . && git commit -m "Signal diagnostics notebook"`

---

## Step 6: Bias analysis

Prompt:

> Create `notebooks/03_bias_analysis.ipynb`. Using the real data:
>
> 1. Run `check_survivorship()` from `src/bias_correction.py` on our signal
>    panel. Report: how many removed firms are we missing? What's our
>    survivorship coverage ratio?
>
> 2. Run `correct_analyst_optimism()` on the earnings revision signal.
>    Compare the uncorrected vs corrected revision signal IC. Plot both
>    IC time series on the same chart.
>
> 3. Run `detect_recency_bias()` on the top 3 signals by IC. Identify
>    periods where short-term IC diverged from long-term IC. What was
>    happening in markets during those periods?
>
> 4. Test shrinkage: take the OLS weights from `combine_ols()` and apply
>    `shrink_weights()` at shrinkage = 0.0, 0.25, 0.5, 0.75, 1.0.
>    For each, run walk-forward and plot OOS IC. Where does shrinkage help?
>
> Keep the notebook focused on findings, not code volume. Each section
> should end with a one-sentence takeaway.

Commit: `git add . && git commit -m "Bias analysis notebook"`

---

## Step 7: Combination methods comparison

Prompt:

> Create `notebooks/04_combination_methods.ipynb`. Using `src/walkforward.py`:
>
> 1. Run `compare_methods()` with all 6 methods: equal_weight, inverse_ic_vol,
>    ols, ridge (alpha=1.0), elastic_net (alpha=0.1, l1_ratio=0.5), bma.
> 2. Print the summary table (method, mean_ic, std_ic, IR, pct_positive).
> 3. Plot the cumulative IC for each method over time on one chart.
> 4. For the ridge method, show how the weights evolve over time (plot
>    weight trajectories for each signal across walk-forward steps).
> 5. Key question to answer: does anything beat equal weight out of sample?
>
> Keep the notebook minimal. Data speaks.

Commit: `git add . && git commit -m "Combination methods comparison"`

---

## Step 8: Regime detection

Prompt:

> Create `notebooks/05_regime_detection.ipynb`:
>
> 1. Load real macro data using `build_macro_panel()`.
> 2. Fit a 2-regime HMM using `fit_regime_model()`.
> 3. Print `regime_summary()` — what characterizes each regime?
> 4. Plot the regime labels over time alongside VIX.
> 5. Run `regime_ic()` from signal_diagnostics for each signal.
>    Which signals work in which regimes?
> 6. Fit a 3-regime model and compare: does adding a third regime
>    improve the story, or is it just fitting noise?
>
> The key insight we're looking for: are there signals that are helpful
> in one regime and destructive in another? If yes, regime-conditional
> combination has a reason to exist.

Commit: `git add . && git commit -m "Regime detection notebook"`

---

## Step 9: Walk-forward with regime conditioning

Prompt:

> Create `notebooks/06_walkforward_evaluation.ipynb`. The final evaluation:
>
> 1. Run `walkforward_evaluate()` with ridge regression (unconditional).
> 2. Run `walkforward_regime_evaluate()` with ridge regression (regime-conditional).
> 3. Compare: does regime conditioning improve OOS IC? At what cost in stability?
> 4. Plot both IC time series on one chart.
> 5. Compute quintile spreads: form quintiles by composite signal, measure
>    average next-quarter return per quintile. Is the spread monotone?
> 6. Final summary table: method | mean_ic | std_ic | IR | quintile_spread
>
> End the notebook with a one-paragraph conclusion: what worked, what didn't,
> and what you'd do differently with better data.

Commit: `git add . && git commit -m "Final walk-forward evaluation"`

---

## What "done" looks like

After Step 9, you should have:
- A signal library with 8-10 diverse signals
- Documented bias corrections with measured impact
- A comparison of 6+ combination methods, all evaluated walk-forward
- Evidence for/against regime-conditional combination
- A clear conclusion about whether sophisticated methods beat equal weight

The honest finding for most people is: equal weight is very hard to beat.
Ridge regression with shrinkage sometimes improves stability. Regime conditioning
helps when there are genuinely regime-dependent signals, but adds estimation
error from the HMM. These are all valid research findings.
