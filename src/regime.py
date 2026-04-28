"""
Regime detection using Hidden Markov Models.

Fits an HMM on macro variables (VIX, yield curve, credit spread,
unemployment) to identify latent market regimes. These regimes
condition the signal combination weights.

Key design choice: the HMM is fit on macro variables only, NOT on
signal performance. This prevents the circular dependency of using
signal IC to define regimes and then using regimes to weight signals.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def fit_regime_model(macro_panel: pd.DataFrame,
                     n_regimes: int = 2,
                     n_iter: int = 100) -> tuple:
    """
    Fit a Gaussian HMM on the macro panel.

    Parameters
    ----------
    macro_panel : DataFrame with columns [yield_curve, credit_spread, vix, unemployment]
        Monthly frequency, datetime index.
    n_regimes : int
        Number of latent states (2 or 3).
    n_iter : int
        EM iterations.

    Returns
    -------
    (model, regime_labels, scaler)
        model : fitted GaussianHMM
        regime_labels : Series indexed by date with integer regime labels
        scaler : fitted StandardScaler (needed for prediction on new data)
    """
    from hmmlearn.hmm import GaussianHMM

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(macro_panel.values)

    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=42,
    )
    model.fit(X)

    labels = model.predict(X)

    # Order regimes by mean VIX (regime 0 = lowest VIX = calm)
    vix_col = list(macro_panel.columns).index("vix")
    regime_vix_means = [
        macro_panel.iloc[labels == r]["vix"].mean()
        for r in range(n_regimes)
    ]
    order = np.argsort(regime_vix_means)
    label_map = {old: new for new, old in enumerate(order)}
    labels = np.array([label_map[l] for l in labels])

    regime_labels = pd.Series(
        labels, index=macro_panel.index, name="regime"
    )

    return model, regime_labels, scaler


def predict_regime(model, scaler, new_macro: pd.DataFrame) -> pd.Series:
    """
    Predict regime for new macro data using a fitted model.

    Parameters
    ----------
    model : fitted GaussianHMM
    scaler : fitted StandardScaler
    new_macro : DataFrame with same columns as training data

    Returns
    -------
    Series of regime labels.
    """
    X = scaler.transform(new_macro.values)
    labels = model.predict(X)
    return pd.Series(labels, index=new_macro.index, name="regime")


def regime_summary(macro_panel: pd.DataFrame,
                   regime_labels: pd.Series) -> pd.DataFrame:
    """
    Summary statistics per regime.

    Returns
    -------
    DataFrame with mean/std of each macro variable per regime.
    """
    combined = macro_panel.copy()
    combined["regime"] = regime_labels

    summary = combined.groupby("regime").agg(["mean", "std", "count"])
    return summary


def combine_by_regime(signal_panel: pd.DataFrame,
                      signal_cols: list[str],
                      outcome_col: str,
                      regime_labels: pd.Series,
                      combination_fn,
                      **kwargs) -> dict[int, np.ndarray]:
    """
    Estimate separate combination weights for each regime.

    Parameters
    ----------
    signal_panel : DataFrame with signal columns + outcome column
    signal_cols : list of signal column names
    outcome_col : name of the outcome column
    regime_labels : Series indexed by date with regime integers
    combination_fn : callable from combination.py (e.g. combine_ridge)

    Returns
    -------
    dict mapping regime_int -> weight_vector
    """
    # Attach regime labels to panel
    panel = signal_panel.copy()
    panel["regime"] = panel["date"].map(regime_labels)
    panel = panel.dropna(subset=["regime"])

    regime_weights = {}
    for regime in sorted(panel["regime"].unique()):
        regime_data = panel[panel["regime"] == regime]
        if len(regime_data) < 50:
            # Too few observations — fall back to equal weight
            regime_weights[int(regime)] = np.ones(len(signal_cols)) / len(signal_cols)
        else:
            weights = combination_fn(regime_data, signal_cols,
                                     outcome_col=outcome_col, **kwargs)
            regime_weights[int(regime)] = weights

    return regime_weights
