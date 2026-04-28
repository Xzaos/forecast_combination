"""
Diagnostic toolkit for individual signals.

Before combining signals, understand each one independently:
- IC time series (Spearman rank correlation with outcome)
- IC decay (how quickly does predictive power fade?)
- Turnover (how much does the ranking change each period?)
- Sector exposure (does the signal just pick one sector?)
- Regime dependency (does it work in all market environments?)
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr


def compute_ic_timeseries(signal: pd.DataFrame,
                          outcome: pd.DataFrame) -> pd.Series:
    """
    Cross-sectional Spearman IC for each date.

    Parameters
    ----------
    signal : DataFrame with [ticker, date, signal]
    outcome : DataFrame with [ticker, date, outcome]

    Returns
    -------
    Series indexed by date, values are IC per period.
    """
    merged = signal.merge(outcome, on=["ticker", "date"], how="inner")
    ics = {}
    for date, group in merged.groupby("date"):
        if len(group) < 10:
            continue
        ic, _ = spearmanr(group["signal"], group["outcome"])
        ics[date] = ic
    return pd.Series(ics, name="IC").sort_index()


def ic_summary(ic_series: pd.Series) -> dict:
    """Summary statistics for an IC time series."""
    return {
        "mean_ic": ic_series.mean(),
        "std_ic": ic_series.std(),
        "ir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
        "pct_positive": (ic_series > 0).mean(),
        "n_periods": len(ic_series),
    }


def compute_turnover(signal: pd.DataFrame) -> pd.Series:
    """
    Rank turnover: average absolute change in percentile rank
    from one period to the next.

    High turnover = expensive to trade on.
    """
    df = signal.sort_values(["ticker", "date"])
    df["prev_signal"] = df.groupby("ticker")["signal"].shift(1)
    df = df.dropna()

    turnover = df.groupby("date").apply(
        lambda g: (g["signal"] - g["prev_signal"]).abs().mean()
    )
    return turnover.rename("turnover")


def sector_ic(signal: pd.DataFrame, outcome: pd.DataFrame,
              sectors: pd.DataFrame) -> pd.DataFrame:
    """
    IC computed within each sector.

    Parameters
    ----------
    signal : DataFrame with [ticker, date, signal]
    outcome : DataFrame with [ticker, date, outcome]
    sectors : DataFrame with [ticker, sector]

    Returns
    -------
    DataFrame with sector-level IC statistics.
    """
    merged = signal.merge(outcome, on=["ticker", "date"])
    merged = merged.merge(sectors, on="ticker")

    results = []
    for sector, group in merged.groupby("sector"):
        ic_ts = {}
        for date, sub in group.groupby("date"):
            if len(sub) < 5:
                continue
            ic, _ = spearmanr(sub["signal"], sub["outcome"])
            ic_ts[date] = ic
        ic_s = pd.Series(ic_ts)
        results.append({
            "sector": sector,
            "mean_ic": ic_s.mean(),
            "std_ic": ic_s.std(),
            "n_periods": len(ic_s),
        })
    return pd.DataFrame(results)


def regime_ic(signal: pd.DataFrame, outcome: pd.DataFrame,
              regime_labels: pd.Series) -> pd.DataFrame:
    """
    IC broken down by market regime.

    Parameters
    ----------
    signal : DataFrame with [ticker, date, signal]
    outcome : DataFrame with [ticker, date, outcome]
    regime_labels : Series indexed by date, values are regime integers.

    Returns
    -------
    DataFrame with regime-level IC statistics.
    """
    ic_ts = compute_ic_timeseries(signal, outcome)
    # Align regime labels to IC dates
    common = ic_ts.index.intersection(regime_labels.index)
    ic_ts = ic_ts.loc[common]
    regimes = regime_labels.loc[common]

    results = []
    for regime in sorted(regimes.unique()):
        mask = regimes == regime
        ic_regime = ic_ts[mask]
        results.append({
            "regime": regime,
            "mean_ic": ic_regime.mean(),
            "std_ic": ic_regime.std(),
            "n_periods": len(ic_regime),
        })
    return pd.DataFrame(results)
