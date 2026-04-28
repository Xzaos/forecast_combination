"""
Walk-forward evaluation engine.

The only honest way to evaluate a combination framework. At each step:
1. Use only data up to time t to estimate weights
2. Apply weights to signals at time t
3. Measure composite IC against outcome at t+1

This prevents the use of future data in weight estimation and gives
a realistic picture of out-of-sample performance.
"""

import numpy as np
import pandas as pd
from typing import Callable


def walkforward_evaluate(signal_panel: pd.DataFrame,
                         signal_cols: list[str],
                         outcome_col: str,
                         combination_fn: Callable,
                         min_train_periods: int = 20,
                         retrain_every: int = 1,
                         **combo_kwargs) -> pd.DataFrame:
    """
    Walk-forward evaluation of a combination method.

    Parameters
    ----------
    signal_panel : DataFrame with columns [ticker, date, sig1, sig2, ..., outcome]
    signal_cols : list of signal column names
    outcome_col : name of the outcome column
    combination_fn : callable(panel, signal_cols, outcome_col=...) -> weights
    min_train_periods : minimum number of unique dates before first evaluation
    retrain_every : re-estimate weights every N periods (1 = every period)

    Returns
    -------
    DataFrame with columns [date, composite_ic, weights_dict, n_stocks]
    """
    from scipy.stats import spearmanr

    dates = sorted(signal_panel["date"].unique())
    results = []
    current_weights = None

    for i, eval_date in enumerate(dates):
        if i < min_train_periods:
            continue

        # Training data: all dates strictly before eval_date
        train_dates = dates[:i]
        train = signal_panel[signal_panel["date"].isin(train_dates)]

        # Re-estimate weights?
        if current_weights is None or (i - min_train_periods) % retrain_every == 0:
            train_clean = train.dropna(subset=signal_cols + [outcome_col])
            if len(train_clean) < 50:
                n = len(signal_cols)
                current_weights = np.ones(n) / n
            else:
                try:
                    current_weights = combination_fn(
                        train_clean, signal_cols,
                        outcome_col=outcome_col, **combo_kwargs
                    )
                except Exception:
                    n = len(signal_cols)
                    current_weights = np.ones(n) / n

        # Evaluate: apply weights to eval_date signals, measure IC with outcome
        eval_data = signal_panel[signal_panel["date"] == eval_date].copy()
        eval_clean = eval_data.dropna(subset=signal_cols + [outcome_col])

        if len(eval_clean) < 10:
            continue

        # Compute composite
        X = eval_clean[signal_cols].values.copy()
        # Fill NaN with cross-sectional median
        for j in range(X.shape[1]):
            col_med = np.nanmedian(X[:, j])
            X[np.isnan(X[:, j]), j] = col_med

        composite = X @ current_weights
        outcome = eval_clean[outcome_col].values

        ic, _ = spearmanr(composite, outcome)

        results.append({
            "date": eval_date,
            "composite_ic": ic,
            "n_stocks": len(eval_clean),
            "weights": dict(zip(signal_cols, current_weights)),
        })

    return pd.DataFrame(results)


def walkforward_regime_evaluate(signal_panel: pd.DataFrame,
                                signal_cols: list[str],
                                outcome_col: str,
                                regime_labels: pd.Series,
                                combination_fn: Callable,
                                min_train_periods: int = 20,
                                **combo_kwargs) -> pd.DataFrame:
    """
    Walk-forward with regime-conditional weights.

    Same as walkforward_evaluate but estimates separate weights
    per regime and applies the weight vector corresponding to the
    current regime at evaluation time.
    """
    from scipy.stats import spearmanr
    from .regime import combine_by_regime

    dates = sorted(signal_panel["date"].unique())
    results = []

    for i, eval_date in enumerate(dates):
        if i < min_train_periods:
            continue

        # Training data
        train_dates = dates[:i]
        train = signal_panel[signal_panel["date"].isin(train_dates)]
        train_clean = train.dropna(subset=signal_cols + [outcome_col])

        if len(train_clean) < 100:
            continue

        # Get regime weights from training data only
        try:
            regime_weights = combine_by_regime(
                train_clean, signal_cols, outcome_col,
                regime_labels, combination_fn, **combo_kwargs
            )
        except Exception:
            continue

        # Current regime
        if eval_date not in regime_labels.index:
            # Find nearest prior regime
            prior = regime_labels.loc[:eval_date]
            if len(prior) == 0:
                continue
            current_regime = int(prior.iloc[-1])
        else:
            current_regime = int(regime_labels.loc[eval_date])

        # Get weights for current regime
        if current_regime in regime_weights:
            weights = regime_weights[current_regime]
        else:
            weights = np.ones(len(signal_cols)) / len(signal_cols)

        # Evaluate
        eval_data = signal_panel[signal_panel["date"] == eval_date]
        eval_clean = eval_data.dropna(subset=signal_cols + [outcome_col])

        if len(eval_clean) < 10:
            continue

        X = eval_clean[signal_cols].values
        for j in range(X.shape[1]):
            col_med = np.nanmedian(X[:, j])
            X[np.isnan(X[:, j]), j] = col_med

        composite = X @ weights
        outcome = eval_clean[outcome_col].values
        ic, _ = spearmanr(composite, outcome)

        results.append({
            "date": eval_date,
            "composite_ic": ic,
            "regime": current_regime,
            "n_stocks": len(eval_clean),
            "weights": dict(zip(signal_cols, weights)),
        })

    return pd.DataFrame(results)


def compare_methods(signal_panel: pd.DataFrame,
                    signal_cols: list[str],
                    outcome_col: str,
                    methods: dict[str, tuple],
                    min_train_periods: int = 20) -> pd.DataFrame:
    """
    Run walk-forward for multiple combination methods and compare.

    Parameters
    ----------
    methods : dict mapping method_name -> (combination_fn, kwargs_dict)
        Example: {"ridge": (combine_ridge, {"alpha": 1.0})}

    Returns
    -------
    DataFrame with summary statistics per method.
    """
    summaries = []

    for name, (fn, kwargs) in methods.items():
        results = walkforward_evaluate(
            signal_panel, signal_cols, outcome_col,
            fn, min_train_periods=min_train_periods, **kwargs
        )
        if len(results) == 0:
            continue

        ic_series = results["composite_ic"]
        summaries.append({
            "method": name,
            "mean_ic": ic_series.mean(),
            "std_ic": ic_series.std(),
            "ir": ic_series.mean() / ic_series.std() if ic_series.std() > 0 else 0,
            "pct_positive": (ic_series > 0).mean(),
            "n_periods": len(ic_series),
        })

    return pd.DataFrame(summaries).sort_values("mean_ic", ascending=False)
