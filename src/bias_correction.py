"""
Bias identification and correction.

This is the intellectual core of the project. Signals are systematically
biased in ways that, if uncorrected, silently degrade forecast quality.

Corrections implemented:
1. Analyst optimism bias: subtract rolling median forecast error
2. Winsorization: cap extreme values to reduce outlier influence
3. Shrinkage: pull combination weights toward equal-weight prior
4. Recency bias detection: flag when recent performance diverges
   sharply from long-run average
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore


def correct_analyst_optimism(estimates: pd.DataFrame,
                             actuals: pd.DataFrame,
                             window: int = 20) -> pd.DataFrame:
    """
    Correct systematic analyst optimism.

    Sell-side analysts systematically overestimate earnings growth.
    Subtract the rolling median forecast error from each consensus
    estimate before computing the revision signal.

    Parameters
    ----------
    estimates : DataFrame with [ticker, date, estimate]
    actuals : DataFrame with [ticker, date, actual]
    window : int
        Rolling window (in quarters) for computing median error.

    Returns
    -------
    DataFrame with [ticker, date, estimate_corrected]
    """
    merged = estimates.merge(actuals, on=["ticker", "date"])
    merged["error"] = merged["actual"] - merged["estimate"]

    # Rolling median error per ticker (how much does this ticker's
    # consensus systematically miss by?)
    merged = merged.sort_values(["ticker", "date"])
    merged["median_error"] = merged.groupby("ticker")["error"].transform(
        lambda x: x.rolling(window, min_periods=4).median()
    )

    # Corrected estimate = original estimate + historical median error
    # (if analysts always overestimate by 0.05, add -0.05 bias correction)
    merged["estimate_corrected"] = merged["estimate"] + merged["median_error"]

    return merged[["ticker", "date", "estimate_corrected"]].dropna()


def winsorize_signal(signal: pd.Series, limits: tuple = (0.01, 0.99)) -> pd.Series:
    """
    Winsorize a signal cross-sectionally at given percentiles.

    Caps extreme values to reduce the influence of outliers without
    discarding observations entirely.
    """
    lower = signal.quantile(limits[0])
    upper = signal.quantile(limits[1])
    return signal.clip(lower=lower, upper=upper)


def winsorize_panel(signal_df: pd.DataFrame, value_col: str = "signal",
                    limits: tuple = (0.01, 0.99)) -> pd.DataFrame:
    """Winsorize signal cross-sectionally within each date."""
    df = signal_df.copy()
    df[value_col] = df.groupby("date")[value_col].transform(
        winsorize_signal, limits=limits
    )
    return df


def detect_recency_bias(ic_series: pd.Series,
                        short_window: int = 8,
                        long_window: int = 40) -> pd.DataFrame:
    """
    Flag periods where recent IC diverges sharply from long-run IC.

    If short-window IC is much higher than long-run, a recency-biased
    model would overweight this signal. If much lower, it would underweight.
    The flag helps decide when to trust vs distrust recent performance.

    Returns
    -------
    DataFrame with [date, ic_short, ic_long, divergence, flag]
    """
    ic_short = ic_series.rolling(short_window, min_periods=4).mean()
    ic_long = ic_series.rolling(long_window, min_periods=12).mean()
    divergence = ic_short - ic_long

    out = pd.DataFrame({
        "ic_short": ic_short,
        "ic_long": ic_long,
        "divergence": divergence,
        "flag": divergence.abs() > 2 * divergence.std(),
    }, index=ic_series.index)

    return out


def shrink_weights(raw_weights: np.ndarray,
                   shrinkage: float = 0.5) -> np.ndarray:
    """
    Shrink combination weights toward equal weighting.

    w_shrunk = (1 - shrinkage) * w_raw + shrinkage * w_equal

    Parameters
    ----------
    raw_weights : array of signal weights (from OLS, ridge, etc.)
    shrinkage : float in [0, 1]. 0 = no shrinkage, 1 = full equal weight.

    Returns
    -------
    Shrunk weight vector, normalized to sum to 1.
    """
    n = len(raw_weights)
    equal = np.ones(n) / n
    shrunk = (1 - shrinkage) * raw_weights + shrinkage * equal
    # Normalize
    shrunk = shrunk / shrunk.sum()
    return shrunk


def check_survivorship(panel: pd.DataFrame,
                       universe_dates: pd.DataFrame) -> dict:
    """
    Audit survivorship bias in the signal panel.

    Checks whether the panel includes firms that later delisted or
    were removed from the index. Returns statistics on coverage.

    Parameters
    ----------
    panel : DataFrame with [ticker, date, ...]
    universe_dates : DataFrame with [ticker, added_date, removed_date]

    Returns
    -------
    dict with survivorship metrics
    """
    # Firms in panel
    panel_tickers = set(panel["ticker"].unique())

    # Firms that were removed before the panel end date
    panel_end = panel["date"].max()
    removed = universe_dates[
        universe_dates["removed_date"] < panel_end
    ]["ticker"].unique()
    removed_set = set(removed)

    # How many removed firms are actually in our panel?
    removed_in_panel = panel_tickers & removed_set
    removed_missing = removed_set - panel_tickers

    return {
        "total_tickers": len(panel_tickers),
        "removed_firms_total": len(removed_set),
        "removed_firms_included": len(removed_in_panel),
        "removed_firms_missing": len(removed_missing),
        "survivorship_coverage": (
            len(removed_in_panel) / len(removed_set)
            if removed_set else 1.0
        ),
    }
