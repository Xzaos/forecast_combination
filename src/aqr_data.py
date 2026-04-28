"""
Load AQR published factor returns.

Downloads CSV files from AQR's Data Library:
https://www.aqr.com/Insights/Datasets

We use two datasets:
- "Betting Against Beta" or "Quality Minus Junk" factors
- "Momentum" (UMD) factor

AQR publishes these as Excel/CSV with monthly returns for US equities.
The exact URLs may change — if they do, download manually and place in data/raw/.
"""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("data/raw/aqr")


def load_aqr_factors(filepath: str = None) -> pd.DataFrame:
    """
    Load AQR factor returns from a local CSV/Excel file.

    Expected columns after loading: date, HML (value), UMD (momentum),
    QMJ (quality), BAB (betting against beta).

    Since AQR's download URLs change, the workflow is:
    1. User downloads the CSV from AQR's website manually
    2. Places it in data/raw/aqr/
    3. This function reads it

    If no file exists, returns a synthetic placeholder using Fama-French
    factors from Ken French's data library (available via pandas-datareader
    or direct CSV download).

    Parameters
    ----------
    filepath : str, optional
        Path to the AQR CSV. If None, looks in data/raw/aqr/ for any CSV.

    Returns
    -------
    pd.DataFrame
        Monthly factor returns with datetime index.
    """
    if filepath is None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        csvs = list(DATA_DIR.glob("*.csv"))
        if csvs:
            filepath = str(csvs[0])
        else:
            return _fallback_factors()

    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    # Standardize column names
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if "val" in cl or "hml" in cl:
            col_map[col] = "HML"
        elif "mom" in cl or "umd" in cl:
            col_map[col] = "UMD"
        elif "qual" in cl or "qmj" in cl:
            col_map[col] = "QMJ"
        elif "bab" in cl or "beta" in cl:
            col_map[col] = "BAB"
    df = df.rename(columns=col_map)
    return df


def _fallback_factors() -> pd.DataFrame:
    """
    Fallback: construct simple value and momentum factors from
    yfinance sector ETFs. This is a rough proxy, not a proper
    factor model, but allows the pipeline to run without AQR data.
    """
    import yfinance as yf

    # Use broad market and a value ETF as crude proxies
    tickers = ["SPY", "IWD", "IWF", "MTUM"]  # market, value, growth, momentum
    data = yf.download(tickers, start="2005-01-01", progress=False)["Close"]

    if data.empty:
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=["HML", "UMD"], dtype=float)

    returns = data.pct_change().dropna()
    monthly = returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)

    factors = pd.DataFrame(index=monthly.index)
    # Crude value = value ETF - growth ETF
    if "IWD" in monthly.columns and "IWF" in monthly.columns:
        factors["HML"] = monthly["IWD"] - monthly["IWF"]
    # Crude momentum = momentum ETF - market
    if "MTUM" in monthly.columns and "SPY" in monthly.columns:
        factors["UMD"] = monthly["MTUM"] - monthly["SPY"]

    return factors.dropna()
