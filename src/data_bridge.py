"""
Synthetic data bridge.

Generates realistic synthetic data so the pipeline can run end-to-end
without depending on Project 1 or any external data source.
"""

import numpy as np
import pandas as pd


def load_all_signal_inputs(
    n_tickers: int = 100,
    start: str = "2012-01-01",
    end: str = "2024-12-31",
) -> dict:
    """
    Generate synthetic but realistic signal inputs.

    Returns
    -------
    dict with keys:
        sue_df              : [ticker, date, sue]
        prices              : [ticker, date, adj_close]
        fundamentals        : [ticker, date, eps_trailing_4q, roe]
        accruals            : [ticker, date, accruals_scaled]
        earnings_estimates  : [ticker, date, actual, estimate]
    """
    rng = np.random.default_rng(42)
    tickers = [f"STOCK_{i:03d}" for i in range(n_tickers)]

    quarter_ends = pd.date_range(start, end, freq="QE")
    biz_days = pd.bdate_range(start, end)

    # --- prices: daily random walk with drift ---
    price_rows = []
    for ticker in tickers:
        returns = rng.normal(0.0004, 0.018, len(biz_days))
        prices = 50 * np.cumprod(1 + returns)
        for d, p in zip(biz_days, prices):
            price_rows.append({"ticker": ticker, "date": d, "adj_close": p})
    prices_df = pd.DataFrame(price_rows)

    # --- quarterly frames ---
    sue_rows, fund_rows, acc_rows, est_rows = [], [], [], []
    for ticker in tickers:
        eps_base = rng.uniform(0.5, 4.0)
        roe_base = rng.uniform(0.05, 0.25)
        for qdate in quarter_ends:
            sue_rows.append({
                "ticker": ticker, "date": qdate,
                "sue": rng.normal(0.0, 1.0),
            })
            eps = eps_base + rng.normal(0, 0.3)
            fund_rows.append({
                "ticker": ticker, "date": qdate,
                "eps_trailing_4q": eps,
                "roe": roe_base + rng.normal(0, 0.02),
            })
            acc_rows.append({
                "ticker": ticker, "date": qdate,
                "accruals_scaled": rng.normal(0.0, 0.05),
            })
            actual = eps_base + rng.normal(0, 0.3)
            # estimates systematically 5% above actuals on average
            estimate = actual * (1.05 + rng.normal(0, 0.03))
            est_rows.append({
                "ticker": ticker, "date": qdate,
                "actual": actual,
                "estimate": estimate,
            })

    return {
        "sue_df": pd.DataFrame(sue_rows),
        "prices": prices_df,
        "fundamentals": pd.DataFrame(fund_rows),
        "accruals": pd.DataFrame(acc_rows),
        "earnings_estimates": pd.DataFrame(est_rows),
    }
