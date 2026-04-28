"""Tests for the synthetic data bridge."""

import pytest
import pandas as pd
from src.data_bridge import load_all_signal_inputs


@pytest.fixture(scope="module")
def data():
    return load_all_signal_inputs(n_tickers=20, start="2015-01-01", end="2018-12-31")


class TestSchema:
    def test_sue_columns(self, data):
        assert set(data["sue_df"].columns) >= {"ticker", "date", "sue"}

    def test_prices_columns(self, data):
        assert set(data["prices"].columns) >= {"ticker", "date", "adj_close"}

    def test_fundamentals_columns(self, data):
        assert set(data["fundamentals"].columns) >= {"ticker", "date", "eps_trailing_4q", "roe"}

    def test_accruals_columns(self, data):
        assert set(data["accruals"].columns) >= {"ticker", "date", "accruals_scaled"}

    def test_estimates_columns(self, data):
        assert set(data["earnings_estimates"].columns) >= {"ticker", "date", "actual", "estimate"}


class TestDtypes:
    def test_prices_adj_close_numeric(self, data):
        assert pd.api.types.is_float_dtype(data["prices"]["adj_close"])

    def test_sue_numeric(self, data):
        assert pd.api.types.is_float_dtype(data["sue_df"]["sue"])

    def test_dates_are_datetimes(self, data):
        assert pd.api.types.is_datetime64_any_dtype(data["prices"]["date"])
        assert pd.api.types.is_datetime64_any_dtype(data["sue_df"]["date"])


class TestConsistentTickers:
    def test_same_tickers_across_all_frames(self, data):
        base = set(data["sue_df"]["ticker"].unique())
        for key in ("prices", "fundamentals", "accruals", "earnings_estimates"):
            assert set(data[key]["ticker"].unique()) == base, f"Ticker mismatch in {key}"


class TestOptimismBias:
    def test_positive_mean_forecast_error(self, data):
        # forecast error = estimate - actual; should be positive (optimism)
        df = data["earnings_estimates"]
        mean_error = (df["estimate"] - df["actual"]).mean()
        assert mean_error > 0, f"Expected positive forecast error, got {mean_error:.4f}"
