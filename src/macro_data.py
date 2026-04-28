"""
Macro data for regime identification.

Pulls four series from FRED + VIX from yfinance:
- Yield curve slope: GS10 - GS2 (10Y minus 2Y Treasury)
- Credit spread: BAA minus AAA corporate bond yields
- VIX: CBOE volatility index
- Unemployment rate: UNRATE

All series are monthly. Output is a single DataFrame indexed by month-end date.

Requires a FRED API key. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html
Store it in a .env file as FRED_API_KEY=your_key_here or pass directly.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw")


def pull_fred_series(series_id: str, fred_api_key: str,
                     start: str = "2005-01-01") -> pd.Series:
    """Pull a single FRED series. Returns monthly Series."""
    from fredapi import Fred
    fred = Fred(api_key=fred_api_key)
    s = fred.get_series(series_id, observation_start=start)
    s.name = series_id
    return s


def pull_vix(start: str = "2005-01-01") -> pd.Series:
    """Pull VIX monthly close from yfinance."""
    import yfinance as yf
    vix = yf.download("^VIX", start=start, progress=False)
    # Resample to month-end, take last close
    monthly = vix["Close"].resample("ME").last()
    monthly.name = "VIX"
    return monthly


def build_macro_panel(fred_api_key: str, start: str = "2005-01-01",
                      cache: bool = True) -> pd.DataFrame:
    """
    Build a monthly macro panel for regime detection.

    Columns: yield_curve, credit_spread, vix, unemployment
    Index: month-end dates

    Starts from 2005 to give the HMM enough pre-sample history
    before the 2010+ signal evaluation period.
    """
    cache_path = DATA_DIR / "macro_panel.parquet"
    if cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    # FRED series
    gs10 = pull_fred_series("GS10", fred_api_key, start)
    gs2 = pull_fred_series("GS2", fred_api_key, start)
    baa = pull_fred_series("BAA", fred_api_key, start)
    aaa = pull_fred_series("AAA", fred_api_key, start)
    unrate = pull_fred_series("UNRATE", fred_api_key, start)

    # VIX
    vix = pull_vix(start)

    # Combine into monthly panel
    panel = pd.DataFrame({
        "yield_curve": gs10 - gs2,
        "credit_spread": baa - aaa,
        "unemployment": unrate,
    })
    # Resample FRED to month-end to align with VIX
    panel = panel.resample("ME").last()
    panel["vix"] = vix

    # Forward-fill short gaps (FRED sometimes has missing months)
    panel = panel.ffill(limit=3)
    panel = panel.dropna()

    if cache:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        panel.to_parquet(cache_path)

    return panel
