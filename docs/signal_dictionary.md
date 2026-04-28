# Signal Dictionary

Exact construction for each base signal. All signals are cross-sectionally
rank-normalized (0 to 1) before combination.

## 1. Time-Series SUE

```
SUE_t = (EPS_t - E[EPS_t]) / σ(residuals)
E[EPS_t] = EPS_{t-4} + drift
drift = mean(EPS_{t-k} - EPS_{t-k-4}) for k in 1..N
σ = std of residuals over trailing 8-20 quarters
```

Source: Project 1 `src/targets.py`
Frequency: Quarterly
Information: Earnings surprise

## 2. Earnings Revision Proxy

```
RevisionProxy_t = (consensus_now - consensus_90d_ago) / |consensus_90d_ago|
```

If 90-day-ago consensus unavailable, use: `surprise_history[-1] / price`
Source: yfinance `.info` fields
Frequency: Quarterly
Information: Analyst sentiment shift

## 3. Price Momentum (12-1)

```
Mom_t = cumulative_return(t-252, t-21) — skip most recent month
```

Standard Jegadeesh-Titman construction. Skip last 21 trading days to avoid
short-term reversal contamination.
Source: yfinance adjusted close
Frequency: Monthly (resample to quarterly for combination)

## 4. Short-Term Reversal (1 month)

```
STR_t = -1 * return(t-21, t)
```

Negative sign: past losers are expected to revert upward.
Source: yfinance adjusted close
Frequency: Monthly

## 5. Earnings Yield (E/P)

```
EP_t = EPS_trailing_4q / price_t
```

Use trailing four quarters of reported EPS (from EDGAR), divided by current price.
Source: EDGAR EPS + yfinance price
Frequency: Quarterly

## 6. ROE Stability

```
ROE_t = net_income_trailing_4q / avg(book_equity_{t}, book_equity_{t-4})
ROE_stability = 1 / std(ROE over trailing 8 quarters)
```

Higher stability = higher quality. Inverse of ROE volatility.
Source: EDGAR (net income, stockholders equity)
Frequency: Quarterly

## 7. Sloan Accruals

```
Accruals = (ΔCA - ΔCash) - (ΔCL - ΔSTD - ΔTP) - Dep
Accruals_scaled = Accruals / avg(total_assets)
Signal = -1 * Accruals_scaled  (low accruals = good)
```

Source: Project 1 `src/features_fund.py`
Frequency: Quarterly (YoY deltas)

## 8. Realized Volatility (60-day)

```
RVol_t = std(daily_returns over trailing 60 trading days) * sqrt(252)
Signal = -1 * RVol  (low vol = good, defensive signal)
```

Source: yfinance daily returns
Frequency: Monthly

## 9. AQR Value Factor Loading

```
β_value = rolling_regression(stock_excess_return ~ AQR_HML, window=60 months)
Signal = β_value  (high loading on value factor = cheap stock)
```

Source: AQR "Value" factor returns + yfinance stock returns
Frequency: Monthly

## 10. AQR Momentum Factor Loading

```
β_mom = rolling_regression(stock_excess_return ~ AQR_UMD, window=60 months)
Signal = β_mom  (high loading on momentum factor)
```

Source: AQR "Momentum" factor returns + yfinance stock returns
Frequency: Monthly

## Cross-sectional normalization

Before combination, every signal is rank-normalized cross-sectionally each period:

```python
signal_ranked = signal.groupby('date').rank(pct=True)
```

This makes signals comparable regardless of their raw scale and makes the
combination weights interpretable.
