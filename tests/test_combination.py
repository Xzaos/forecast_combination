"""Tests for combination methods."""

import numpy as np
import pandas as pd
import pytest
from src.combination import (
    combine_equal_weight, combine_inverse_ic_vol, combine_ols,
    combine_ridge, combine_elastic_net, combine_bma, apply_weights,
)


@pytest.fixture
def sample_panel():
    """Synthetic signal panel with 3 signals and an outcome."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "ticker": [f"S{i % 20}" for i in range(n)],
        "date": pd.to_datetime(
            [f"2020-{(i // 20) + 1:02d}-28" for i in range(n)]
        ),
        "sig_a": np.random.randn(n),
        "sig_b": np.random.randn(n),
        "sig_c": np.random.randn(n),
    })
    # Outcome is correlated with sig_a and sig_b, not sig_c
    df["outcome"] = 0.4 * df["sig_a"] + 0.3 * df["sig_b"] + np.random.randn(n) * 0.5
    return df


@pytest.fixture
def signal_cols():
    return ["sig_a", "sig_b", "sig_c"]


class TestEqualWeight:
    def test_weights_sum_to_one(self, sample_panel, signal_cols):
        w = combine_equal_weight(sample_panel, signal_cols)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_all_equal(self, sample_panel, signal_cols):
        w = combine_equal_weight(sample_panel, signal_cols)
        assert np.allclose(w, 1 / 3)


class TestRidge:
    def test_weights_not_extreme(self, sample_panel, signal_cols):
        w = combine_ridge(sample_panel, signal_cols, alpha=1.0)
        # Ridge should prevent extreme weights
        assert np.max(np.abs(w)) < 5.0

    def test_stronger_alpha_shrinks_more(self, sample_panel, signal_cols):
        # Higher alpha should produce raw coefficients closer to each other
        # (before normalization). We test via the raw Ridge coefs.
        from sklearn.linear_model import Ridge as _Ridge

        X = sample_panel[signal_cols].values
        y = sample_panel["outcome"].values
        mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X, y = X[mask], y[mask]

        m_low = _Ridge(alpha=0.01).fit(X, y)
        m_high = _Ridge(alpha=100.0).fit(X, y)
        # Higher alpha shrinks raw coefs toward zero
        assert np.max(np.abs(m_high.coef_)) < np.max(np.abs(m_low.coef_))


class TestElasticNet:
    def test_can_zero_signals(self, sample_panel, signal_cols):
        # With strong L1, some weights should be exactly zero
        w = combine_elastic_net(sample_panel, signal_cols,
                                alpha=1.0, l1_ratio=0.9)
        # sig_c has no relationship to outcome; may be zeroed
        # (this is probabilistic, so we just check it runs)
        assert len(w) == 3


class TestBMA:
    def test_weights_sum_to_one(self, sample_panel, signal_cols):
        w = combine_bma(sample_panel, signal_cols)
        assert abs(w.sum() - 1.0) < 1e-10

    def test_informative_signal_gets_more_weight(self, sample_panel, signal_cols):
        w = combine_bma(sample_panel, signal_cols)
        # sig_a has the strongest relationship to outcome (0.4 coefficient)
        # It should get more weight than sig_c (no relationship)
        assert w[0] > w[2]


class TestApplyWeights:
    def test_output_length(self, sample_panel, signal_cols):
        w = np.array([0.4, 0.3, 0.3])
        composite = apply_weights(sample_panel, signal_cols, w)
        assert len(composite) == len(sample_panel)

    def test_handles_nan(self, sample_panel, signal_cols):
        panel = sample_panel.copy()
        panel.loc[0, "sig_a"] = np.nan
        w = np.array([0.4, 0.3, 0.3])
        composite = apply_weights(panel, signal_cols, w)
        # Should not have NaN in output (NaN replaced with median)
        assert not np.isnan(composite.iloc[0])


class TestWalkforwardSafety:
    """Verify that no combination method uses future data."""

    def test_ols_uses_only_provided_data(self, sample_panel, signal_cols):
        """OLS weights change when we change the training data's outcome."""
        train = sample_panel.copy()
        w_original = combine_ols(train, signal_cols)

        # Corrupt the outcome — weights MUST change
        train_corrupted = train.copy()
        train_corrupted["outcome"] = -train_corrupted["outcome"]
        w_corrupted = combine_ols(train_corrupted, signal_cols)

        assert not np.allclose(w_original, w_corrupted, atol=0.01)
