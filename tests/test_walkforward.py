"""Tests for the walk-forward evaluation engine."""

import numpy as np
import pandas as pd
import pytest
from src.combination import combine_equal_weight, combine_ridge
from src.walkforward import walkforward_evaluate, compare_methods


@pytest.fixture
def panel_with_outcome():
    """
    Synthetic panel: 20 stocks, 40 quarterly dates, 3 signals.
    Outcome is weakly correlated with sig_a.
    """
    np.random.seed(123)
    tickers = [f"T{i:02d}" for i in range(20)]
    dates = pd.date_range("2015-03-31", periods=40, freq="QE")
    rows = []
    for d in dates:
        for t in tickers:
            rows.append({
                "ticker": t,
                "date": d,
                "sig_a": np.random.randn(),
                "sig_b": np.random.randn(),
                "sig_c": np.random.randn(),
            })
    df = pd.DataFrame(rows)
    df["outcome"] = 0.1 * df["sig_a"] + np.random.randn(len(df)) * 0.9
    return df


@pytest.fixture
def signal_cols():
    return ["sig_a", "sig_b", "sig_c"]


class TestWalkforward:
    def test_returns_results(self, panel_with_outcome, signal_cols):
        results = walkforward_evaluate(
            panel_with_outcome, signal_cols, "outcome",
            combine_equal_weight, min_train_periods=10,
        )
        assert len(results) > 0
        assert "composite_ic" in results.columns

    def test_no_future_data(self, panel_with_outcome, signal_cols):
        """
        The first evaluation date should use only the first
        min_train_periods of data. If we change the future data,
        the first evaluation should be identical.
        """
        results_full = walkforward_evaluate(
            panel_with_outcome, signal_cols, "outcome",
            combine_ridge, min_train_periods=10, alpha=1.0,
        )

        # Corrupt the last 10 dates' outcomes
        corrupted = panel_with_outcome.copy()
        dates = sorted(corrupted["date"].unique())
        late_dates = dates[-10:]
        mask = corrupted["date"].isin(late_dates)
        corrupted.loc[mask, "outcome"] = 999.0

        results_corrupted = walkforward_evaluate(
            corrupted, signal_cols, "outcome",
            combine_ridge, min_train_periods=10, alpha=1.0,
        )

        # First few evaluation ICs should be identical (they don't see late data)
        first_n = 5
        np.testing.assert_array_almost_equal(
            results_full["composite_ic"].iloc[:first_n].values,
            results_corrupted["composite_ic"].iloc[:first_n].values,
            decimal=10,
        )

    def test_weights_dict_present(self, panel_with_outcome, signal_cols):
        results = walkforward_evaluate(
            panel_with_outcome, signal_cols, "outcome",
            combine_equal_weight, min_train_periods=10,
        )
        assert "weights" in results.columns
        # Each weight dict should have all signal columns
        for w in results["weights"]:
            assert set(w.keys()) == set(signal_cols)


class TestCompareMethods:
    def test_returns_summary(self, panel_with_outcome, signal_cols):
        methods = {
            "equal": (combine_equal_weight, {}),
            "ridge": (combine_ridge, {"alpha": 1.0}),
        }
        summary = compare_methods(
            panel_with_outcome, signal_cols, "outcome",
            methods, min_train_periods=10,
        )
        assert len(summary) == 2
        assert "mean_ic" in summary.columns
        assert "method" in summary.columns
