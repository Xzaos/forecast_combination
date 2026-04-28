"""
Signal library: construct each base signal from raw data.

Every function takes cleaned input data and returns a DataFrame with columns:
    ticker, date, signal_value

All signals are cross-sectionally rank-normalized (0 to 1) before output.
This makes them comparable and the combination weights interpretable.

Signals are constructed with strict point-in-time discipline: no future data
is used in any computation. Dates represent the date at which the signal
value was *known*, not the period it refers to.
"""

import pandas as pd
import numpy as np


def rank_normalize(df: pd.DataFrame, value_col: str = "signal") -> pd.Series:
    """Cross-sectional percentile rank within each date."""
    return df.groupby("date")[value_col].rank(pct=True)


# ---------------------------------------------------------------------------
# Signal 1: Time-Series SUE (from Project 1)
# ---------------------------------------------------------------------------
def signal_sue(sue_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap the SUE target from Project 1 as a signal.

    Parameters
    ----------
    sue_df : DataFrame with columns [ticker, date, sue]
        Output of Project 1's targets.time_series_sue().
        'date' should be the filing date (point-in-time).

    Returns
    -------
    DataFrame with columns [ticker, date, signal]
    """
    out = sue_df[["ticker", "date"]].copy()
    out["signal"] = sue_df["sue"]
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 2: Earnings Revision Proxy
# ---------------------------------------------------------------------------
def signal_earnings_revision(earnings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Proxy earnings revision signal.

    Uses the most recent earnings surprise as a fraction of price.
    With free data, we can't get true revision time series, so we
    use: surprise_pct = (actual - estimate) / |estimate|

    Parameters
    ----------
    earnings_df : DataFrame with columns [ticker, date, actual, estimate]

    Returns
    -------
    DataFrame with [ticker, date, signal]
    """
    out = earnings_df[["ticker", "date"]].copy()
    denom = earnings_df["estimate"].abs().clip(lower=0.01)
    out["signal"] = (earnings_df["actual"] - earnings_df["estimate"]) / denom
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 3: Price Momentum (12-1 month)
# ---------------------------------------------------------------------------
def signal_momentum(prices: pd.DataFrame) -> pd.DataFrame:
    """
    12-minus-1 month momentum (Jegadeesh-Titman).

    Parameters
    ----------
    prices : DataFrame with columns [ticker, date, adj_close]
        Daily adjusted close prices.

    Returns
    -------
    DataFrame with [ticker, date, signal] at monthly frequency.
    """
    # Pivot to wide, resample to month-end
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    monthly = wide.resample("ME").last()

    # 12-month return skipping most recent month
    ret_12 = monthly.shift(1) / monthly.shift(12) - 1

    # Melt back to long
    out = ret_12.stack().reset_index()
    out.columns = ["date", "ticker", "signal"]
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 4: Short-Term Reversal (1 month)
# ---------------------------------------------------------------------------
def signal_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    """
    1-month reversal: negative of trailing 1-month return.

    Parameters
    ----------
    prices : DataFrame with [ticker, date, adj_close], daily.

    Returns
    -------
    DataFrame with [ticker, date, signal], monthly.
    """
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    monthly = wide.resample("ME").last()
    ret_1m = monthly / monthly.shift(1) - 1
    reversal = -1 * ret_1m

    out = reversal.stack().reset_index()
    out.columns = ["date", "ticker", "signal"]
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 5: Earnings Yield (E/P)
# ---------------------------------------------------------------------------
def signal_earnings_yield(fundamentals: pd.DataFrame,
                          prices: pd.DataFrame) -> pd.DataFrame:
    """
    Trailing earnings yield = EPS_trailing_4q / price.

    Parameters
    ----------
    fundamentals : DataFrame with [ticker, date, eps_trailing_4q]
        'date' is the filing date (point-in-time).
    prices : DataFrame with [ticker, date, adj_close]

    Returns
    -------
    DataFrame with [ticker, date, signal]
    """
    # Get price as-of each fundamental date
    fund = fundamentals[["ticker", "date", "eps_trailing_4q"]].copy()
    fund = fund.sort_values(["ticker", "date"])

    # Merge with nearest prior price
    prices_daily = prices.pivot(index="date", columns="ticker",
                                values="adj_close")
    # For each fundamental row, get the price on that date (or nearest prior)
    price_vals = []
    for _, row in fund.iterrows():
        ticker_prices = prices_daily.get(row["ticker"])
        if ticker_prices is None:
            price_vals.append(np.nan)
            continue
        valid = ticker_prices.loc[:row["date"]].dropna()
        price_vals.append(valid.iloc[-1] if len(valid) > 0 else np.nan)

    fund["price"] = price_vals
    fund["signal"] = fund["eps_trailing_4q"] / fund["price"].clip(lower=0.01)

    out = fund[["ticker", "date", "signal"]].dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 6: ROE Stability
# ---------------------------------------------------------------------------
def signal_roe_stability(fundamentals: pd.DataFrame) -> pd.DataFrame:
    """
    Quality signal: inverse of ROE volatility over trailing 8 quarters.

    Parameters
    ----------
    fundamentals : DataFrame with [ticker, date, roe]
        Quarterly ROE = net_income / avg_book_equity.

    Returns
    -------
    DataFrame with [ticker, date, signal]
    """
    df = fundamentals.sort_values(["ticker", "date"])
    df["roe_std"] = df.groupby("ticker")["roe"].transform(
        lambda x: x.rolling(8, min_periods=4).std()
    )
    out = df[["ticker", "date"]].copy()
    out["signal"] = 1.0 / df["roe_std"].clip(lower=0.001)
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 7: Sloan Accruals (from Project 1)
# ---------------------------------------------------------------------------
def signal_accruals(accruals_df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrap Sloan accruals from Project 1. Low accruals = good.

    Parameters
    ----------
    accruals_df : DataFrame with [ticker, date, accruals_scaled]

    Returns
    -------
    DataFrame with [ticker, date, signal]
    """
    out = accruals_df[["ticker", "date"]].copy()
    out["signal"] = -1.0 * accruals_df["accruals_scaled"]  # negate: low = good
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 8: Realized Volatility (60-day)
# ---------------------------------------------------------------------------
def signal_realized_vol(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Defensive signal: low realized volatility.

    Parameters
    ----------
    prices : DataFrame with [ticker, date, adj_close], daily.

    Returns
    -------
    DataFrame with [ticker, date, signal], monthly.
    """
    wide = prices.pivot(index="date", columns="ticker", values="adj_close")
    daily_ret = wide.pct_change()

    # 60-day rolling std, annualized
    vol = daily_ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    monthly_vol = vol.resample("ME").last()

    # Negate: low vol = high signal
    out = (-1 * monthly_vol).stack().reset_index()
    out.columns = ["date", "ticker", "signal"]
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Signal 9 & 10: Factor Loadings (AQR value + momentum)
# ---------------------------------------------------------------------------
def signal_factor_loading(stock_returns: pd.DataFrame,
                          factor_returns: pd.Series,
                          window: int = 60) -> pd.DataFrame:
    """
    Rolling beta of each stock on a given factor.

    Parameters
    ----------
    stock_returns : DataFrame, monthly returns with ticker columns, date index.
    factor_returns : Series, monthly factor returns with date index.
    window : int, rolling window in months.

    Returns
    -------
    DataFrame with [ticker, date, signal]
    """
    # Align dates
    common = stock_returns.index.intersection(factor_returns.index)
    sr = stock_returns.loc[common]
    fr = factor_returns.loc[common]

    betas = pd.DataFrame(index=sr.index, columns=sr.columns, dtype=float)

    for ticker in sr.columns:
        stock = sr[ticker].dropna()
        if len(stock) < window:
            continue
        # Rolling OLS beta
        for i in range(window, len(stock)):
            y = stock.iloc[i - window:i].values
            x = fr.loc[stock.index[i - window:i]].values
            if len(x) < window or np.isnan(x).any() or np.isnan(y).any():
                continue
            cov = np.cov(y, x)
            if cov[1, 1] > 0:
                betas.loc[stock.index[i], ticker] = cov[0, 1] / cov[1, 1]

    out = betas.stack().reset_index()
    out.columns = ["date", "ticker", "signal"]
    out["signal"] = out["signal"].astype(float)
    out = out.dropna()
    out["signal"] = rank_normalize(out)
    return out


# ---------------------------------------------------------------------------
# Combine all signals into a single panel
# ---------------------------------------------------------------------------
def build_signal_panel(signals: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Merge all individual signal DataFrames into one wide panel.

    Parameters
    ----------
    signals : dict mapping signal_name -> DataFrame with [ticker, date, signal]

    Returns
    -------
    DataFrame with [ticker, date, sig_1, sig_2, ...] — one column per signal.
    """
    merged = None
    for name, df in signals.items():
        renamed = df.rename(columns={"signal": name})
        if merged is None:
            merged = renamed
        else:
            merged = merged.merge(renamed, on=["ticker", "date"], how="outer")

    return merged.sort_values(["date", "ticker"]).reset_index(drop=True)
