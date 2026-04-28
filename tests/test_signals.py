"""Tests for the signal library."""

import numpy as np
import pandas as pd
import pytest
from src.signals import (
    rank_normalize, signal_momentum, signal_reversal,
    signal_realized_vol, build_signal_panel,
)


@pytest.fixture
def sample_prices():
    """Synthetic daily price panel for 10 stocks over 2 years."""
    np.random.seed(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    tickers = [f"STOCK_{i}" for i in range(10)]

    rows = []
    for ticker in tickers:
        # Random walk with drift
        returns = np.random.normal(0.0005, 0.02, len(dates))
        prices = 100 * np.cumprod(1 + returns)
        for d, p in zip(dates, prices):
            rows.append({"ticker": ticker, "date": d, "adj_close": p})

    return pd.DataFrame(rows)


class TestRankNormalize:
    def test_output_range(self):
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 5,
            "signal": [10, 20, 30, 40, 50],
        })
        ranked = rank_normalize(df)
        assert ranked.min() >= 0
        assert ranked.max() <= 1

    def test_preserves_order(self):
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 5,
            "signal": [1, 2, 3, 4, 5],
        })
        ranked = rank_normalize(df)
        assert list(ranked) == sorted(ranked)


class TestMomentum:
    def test_returns_expected_columns(self, sample_prices):
        result = signal_momentum(sample_prices)
        assert set(result.columns) == {"date", "ticker", "signal"}

    def test_no_nans_in_output(self, sample_prices):
        result = signal_momentum(sample_prices)
        assert result["signal"].isna().sum() == 0

    def test_signal_is_rank_normalized(self, sample_prices):
        result = signal_momentum(sample_prices)
        # Each date should have ranks between 0 and 1
        for _, group in result.groupby("date"):
            assert group["signal"].min() >= 0
            assert group["signal"].max() <= 1


class TestReversal:
    def test_returns_expected_columns(self, sample_prices):
        result = signal_reversal(sample_prices)
        assert set(result.columns) == {"date", "ticker", "signal"}


class TestRealizedVol:
    def test_returns_monthly(self, sample_prices):
        result = signal_realized_vol(sample_prices)
        # Should have fewer rows than daily data
        assert len(result) < len(sample_prices)


class TestBuildPanel:
    def test_merge_signals(self):
        sig1 = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "date": pd.to_datetime(["2020-01-31"] * 2 + ["2020-02-28"] * 2),
            "signal": [0.2, 0.8, 0.6, 0.4],
        })
        sig2 = pd.DataFrame({
            "ticker": ["A", "B", "A", "B"],
            "date": pd.to_datetime(["2020-01-31"] * 2 + ["2020-02-28"] * 2),
            "signal": [0.5, 0.5, 0.3, 0.7],
        })
        panel = build_signal_panel({"momentum": sig1, "reversal": sig2})
        assert "momentum" in panel.columns
        assert "reversal" in panel.columns
        assert len(panel) == 4
