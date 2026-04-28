"""Tests for regime detection."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_macro():
    """
    Two-regime synthetic macro panel.
    Regime 0: low VIX, positive yield curve.
    Regime 1: high VIX, flat yield curve.
    """
    np.random.seed(42)
    dates = pd.date_range("2010-01-31", periods=120, freq="ME")

    # Alternate regimes every 30 months
    regime_true = np.array([0] * 30 + [1] * 30 + [0] * 30 + [1] * 30)

    data = pd.DataFrame(index=dates)
    data["yield_curve"] = np.where(regime_true == 0,
                                    np.random.normal(1.5, 0.3, 120),
                                    np.random.normal(0.2, 0.5, 120))
    data["credit_spread"] = np.where(regime_true == 0,
                                      np.random.normal(1.0, 0.2, 120),
                                      np.random.normal(2.5, 0.5, 120))
    data["vix"] = np.where(regime_true == 0,
                            np.random.normal(15, 3, 120),
                            np.random.normal(30, 5, 120))
    data["unemployment"] = np.where(regime_true == 0,
                                     np.random.normal(4.5, 0.5, 120),
                                     np.random.normal(7.0, 1.0, 120))

    return data, regime_true


class TestRegimeDetection:
    def test_fit_returns_labels(self, synthetic_macro):
        from src.regime import fit_regime_model
        macro, _ = synthetic_macro
        model, labels, scaler = fit_regime_model(macro, n_regimes=2)
        assert len(labels) == len(macro)
        assert set(labels.unique()) == {0, 1}

    def test_regime_ordering(self, synthetic_macro):
        """Regime 0 should have lower mean VIX than regime 1."""
        from src.regime import fit_regime_model
        macro, _ = synthetic_macro
        _, labels, _ = fit_regime_model(macro, n_regimes=2)

        vix_r0 = macro.loc[labels == 0, "vix"].mean()
        vix_r1 = macro.loc[labels == 1, "vix"].mean()
        assert vix_r0 < vix_r1

    def test_reasonable_accuracy(self, synthetic_macro):
        """
        With cleanly separated regimes, HMM should recover them
        with >80% accuracy.
        """
        from src.regime import fit_regime_model
        macro, true_regimes = synthetic_macro
        _, labels, _ = fit_regime_model(macro, n_regimes=2)

        accuracy = (labels.values == true_regimes).mean()
        assert accuracy > 0.80

    def test_predict_new_data(self, synthetic_macro):
        from src.regime import fit_regime_model, predict_regime
        macro, _ = synthetic_macro

        # Fit on first 80, predict on last 40
        train = macro.iloc[:80]
        test = macro.iloc[80:]

        model, _, scaler = fit_regime_model(train, n_regimes=2)
        preds = predict_regime(model, scaler, test)
        assert len(preds) == 40
        assert set(preds.unique()).issubset({0, 1})
