"""
Signal combination methods.

Takes a panel of rank-normalized signals and an outcome, produces
combination weights. Methods ordered by complexity:

1. Equal weight (benchmark)
2. Inverse-IC-volatility
3. OLS
4. Ridge regression
5. Elastic net
6. Bayesian Model Averaging (BMA)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, ElasticNet
from scipy.optimize import minimize


def combine_equal_weight(signal_panel: pd.DataFrame,
                         signal_cols: list[str]) -> np.ndarray:
    """
    Equal weight: 1/N for each signal.
    Benchmark — surprisingly hard to beat OOS.
    """
    n = len(signal_cols)
    return np.ones(n) / n


def combine_inverse_ic_vol(ic_dict: dict[str, pd.Series],
                           signal_cols: list[str],
                           window: int = 12) -> np.ndarray:
    """
    Weight each signal by the inverse of its recent IC volatility.
    Signals with more stable IC get higher weight.

    Parameters
    ----------
    ic_dict : dict mapping signal_name -> IC time series (pd.Series)
    signal_cols : list of signal names (ordering must match)
    window : trailing window for computing IC std

    Returns
    -------
    Weight vector, normalized to sum to 1.
    """
    inv_vols = []
    for col in signal_cols:
        ic = ic_dict[col]
        vol = ic.iloc[-window:].std()
        inv_vols.append(1.0 / max(vol, 0.001))

    weights = np.array(inv_vols)
    return weights / weights.sum()


def combine_ols(signal_panel: pd.DataFrame, signal_cols: list[str],
                outcome_col: str = "outcome") -> np.ndarray:
    """
    OLS: regress outcome on all signals. Coefficients = weights.
    No regularization — will overfit with many correlated signals.
    """
    X = signal_panel[signal_cols].values
    y = signal_panel[outcome_col].values

    # Drop rows with NaN
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    # OLS via normal equations
    XtX = X.T @ X
    Xty = X.T @ y
    try:
        weights = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Fallback to equal weight if singular
        return combine_equal_weight(signal_panel, signal_cols)

    # Normalize
    if weights.sum() != 0:
        weights = weights / np.abs(weights).sum()
    return weights


def combine_ridge(signal_panel: pd.DataFrame, signal_cols: list[str],
                  outcome_col: str = "outcome",
                  alpha: float = 1.0) -> np.ndarray:
    """
    Ridge regression: L2-penalized OLS. Shrinks extreme weights.

    Parameters
    ----------
    alpha : regularization strength. Higher = more shrinkage toward zero.
    """
    X = signal_panel[signal_cols].values
    y = signal_panel[outcome_col].values

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    weights = model.coef_

    # Normalize
    if np.abs(weights).sum() > 0:
        weights = weights / np.abs(weights).sum()
    return weights


def combine_elastic_net(signal_panel: pd.DataFrame, signal_cols: list[str],
                        outcome_col: str = "outcome",
                        alpha: float = 0.1,
                        l1_ratio: float = 0.5) -> np.ndarray:
    """
    Elastic net: L1+L2 penalty. Can zero out useless signals.

    Parameters
    ----------
    alpha : overall penalty strength
    l1_ratio : balance between L1 (lasso) and L2 (ridge). 1.0 = pure lasso.
    """
    X = signal_panel[signal_cols].values
    y = signal_panel[outcome_col].values

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=True,
                       max_iter=5000)
    model.fit(X, y)
    weights = model.coef_

    if np.abs(weights).sum() > 0:
        weights = weights / np.abs(weights).sum()
    return weights


def combine_bma(signal_panel: pd.DataFrame, signal_cols: list[str],
                outcome_col: str = "outcome",
                prior_weight: float = 1.0) -> np.ndarray:
    """
    Bayesian Model Averaging (simplified).

    Assign each signal a posterior weight based on its marginal likelihood
    (approximated by leave-one-out prediction error).

    The idea: each signal defines a "model" (just that signal alone).
    The posterior probability of each model is proportional to its
    prior probability times its likelihood. We use the in-sample
    sum of squared errors as a proxy for the negative log-likelihood.

    Parameters
    ----------
    prior_weight : float
        Prior pseudo-count for each signal. Higher = more equal weighting.

    Returns
    -------
    Weight vector (posterior model probabilities), sums to 1.
    """
    X = signal_panel[signal_cols].values
    y = signal_panel[outcome_col].values

    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X, y = X[mask], y[mask]

    n = len(signal_cols)
    log_likelihoods = np.zeros(n)

    for i in range(n):
        # Each "model" is univariate regression on signal i
        xi = X[:, i]
        slope = np.cov(xi, y)[0, 1] / max(np.var(xi), 1e-10)
        intercept = y.mean() - slope * xi.mean()
        pred = slope * xi + intercept
        sse = np.sum((y - pred) ** 2)
        # Approximate log-likelihood (Gaussian)
        sigma2 = sse / len(y)
        log_likelihoods[i] = -0.5 * len(y) * np.log(max(sigma2, 1e-10))

    # Add log prior (uniform = equal for all)
    log_posterior = log_likelihoods + np.log(prior_weight)

    # Normalize in log space for numerical stability
    log_posterior -= log_posterior.max()
    weights = np.exp(log_posterior)
    weights = weights / weights.sum()

    return weights


def apply_weights(signal_panel: pd.DataFrame, signal_cols: list[str],
                  weights: np.ndarray) -> pd.Series:
    """
    Apply combination weights to produce composite signal.

    Returns
    -------
    Series of composite signal values, same index as signal_panel.
    """
    X = signal_panel[signal_cols].values.copy()
    # Handle NaN: replace with cross-sectional median for that signal
    for j in range(X.shape[1]):
        col_median = np.nanmedian(X[:, j])
        X[np.isnan(X[:, j]), j] = col_median

    composite = X @ weights
    return pd.Series(composite, index=signal_panel.index, name="composite")
