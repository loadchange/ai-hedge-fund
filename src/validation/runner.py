"""End-to-end signal evaluation: signal → CPCV → IS/OOS Sharpes → PBO.

Workflow::

    evaluate_signal(
        signal_name="momentum",
        ticker="AAPL",
        start="2020-01-01",
        end="2025-01-01",
        n_splits=8,
        n_test_splits=2,
    )

Pulls daily prices, runs the named signal on a rolling 60-day window
(producing a forecast each day), splits into combinatorial purged folds,
and reports IS/OOS Sharpe per split + an overall PBO. The "strategy"
under test is "go ±1 unit when the signal is bullish/bearish" — a clean
baseline for comparing signals.

For comparing *multiple* signal variants (e.g. different parameter
sweeps), pass them all through :func:`evaluate_multiple_signals` and
PBO ranks them as a group.
"""

from __future__ import annotations

import math
from datetime import datetime
from dateutil.relativedelta import relativedelta

import numpy as np
import pandas as pd

from src.signals import SIGNAL_REGISTRY
from src.tools.api import get_prices, prices_to_df
from src.validation.cpcv import CombinatorialPurgedKFold
from src.validation.pbo import compute_pbo, deflated_sharpe_ratio


def _annualised_sharpe(returns: np.ndarray, *, periods_per_year: int = 252) -> float:
    """Sharpe ratio assuming zero risk-free rate; returns nan when insufficient data."""
    returns = np.asarray(returns, dtype=float)
    returns = returns[~np.isnan(returns)]
    if len(returns) < 2:
        return float("nan")
    std = float(np.std(returns, ddof=1))
    if std <= 0:
        return float("nan")
    return float(np.mean(returns) / std * math.sqrt(periods_per_year))


def _generate_signal_path(
    signal_name: str,
    prices_df: pd.DataFrame,
    *,
    rolling_window: int = 60,
) -> pd.Series:
    """Compute a daily ``[-1, +1]`` signal trajectory by rolling the model.

    For each date we run the signal on the trailing *rolling_window* of
    prices and record the resulting ``value`` (or 0 if the signal can't
    compute on that window).
    """
    if signal_name not in SIGNAL_REGISTRY:
        raise ValueError(
            f"Unknown signal {signal_name!r}; choose from {sorted(SIGNAL_REGISTRY)}"
        )
    cls = SIGNAL_REGISTRY[signal_name]
    instance = cls()
    n = len(prices_df)
    values = np.zeros(n)
    for i in range(rolling_window, n):
        window = prices_df.iloc[i - rolling_window: i]
        try:
            result = instance.compute_from_prices(window)
            values[i] = result.value
        except NotImplementedError:
            # Fundamental signal — skip the rolling-window backtest path.
            return pd.Series(np.nan, index=prices_df.index, name=signal_name)
        except Exception:
            values[i] = 0.0
    return pd.Series(values, index=prices_df.index, name=signal_name)


def evaluate_signal(
    signal_name: str,
    ticker: str,
    *,
    start_date: str,
    end_date: str,
    n_splits: int = 8,
    n_test_splits: int = 2,
    embargo_pct: float = 0.01,
    rolling_window: int = 60,
    api_key: str | None = None,
) -> dict:
    """Run CPCV on a single signal-ticker pair and return IS/OOS stats + PBO.

    Returns a dict::

        {
            "signal": str,
            "ticker": str,
            "n_obs": int,
            "n_splits": int,
            "n_test_splits": int,
            "is_sharpes": [...]        # one per split combination
            "oos_sharpes": [...],
            "is_sharpe_mean": float,
            "oos_sharpe_mean": float,
            "pbo": {pbo, logits, median_logit, stochastic_dominance},
            "deflated_sharpe": float,  # probability the OOS Sharpe is real
        }
    """
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    if not prices:
        return {
            "signal": signal_name,
            "ticker": ticker,
            "error": "no price data",
        }

    df = prices_to_df(prices)
    if len(df) < rolling_window * 2:
        return {
            "signal": signal_name,
            "ticker": ticker,
            "error": f"insufficient data ({len(df)} bars; need ≥ {rolling_window * 2})",
        }

    forward_returns = df["close"].pct_change().shift(-1).fillna(0).to_numpy()
    signal_path = _generate_signal_path(
        signal_name, df, rolling_window=rolling_window,
    )
    if signal_path.isna().all():
        return {
            "signal": signal_name,
            "ticker": ticker,
            "error": (
                f"signal '{signal_name}' is fundamental-only — use compute(ticker, end_date)"
            ),
        }

    # Strategy returns: position = sign(signal) × |signal| × forward return.
    strategy_returns = signal_path.to_numpy() * forward_returns
    n = len(strategy_returns)

    cpcv = CombinatorialPurgedKFold(
        n_splits=n_splits,
        n_test_splits=n_test_splits,
        embargo_pct=embargo_pct,
    )

    is_sharpes: list[float] = []
    oos_sharpes: list[float] = []
    for train_idx, test_idx in cpcv.split(n):
        is_sharpes.append(_annualised_sharpe(strategy_returns[train_idx]))
        oos_sharpes.append(_annualised_sharpe(strategy_returns[test_idx]))

    # PBO needs ≥ 2 trials; for a single signal we synthesise a baseline
    # (zero-return strategy) so we get a comparison.
    is_arr = np.array([is_sharpes, [0.0] * len(is_sharpes)])
    oos_arr = np.array([oos_sharpes, [0.0] * len(oos_sharpes)])
    pbo_result = compute_pbo(is_arr, oos_arr)

    # When every Sharpe is NaN (rolling window too small for this signal)
    # nanmean fires a RuntimeWarning — silence it; downstream code already
    # handles NaN means gracefully.
    with np.errstate(invalid="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            oos_mean = float(np.nanmean(oos_sharpes))
            is_mean = float(np.nanmean(is_sharpes))

    dsr = deflated_sharpe_ratio(
        oos_mean, n_trials=cpcv.get_n_splits(), n_obs=n,
    )

    return {
        "signal": signal_name,
        "ticker": ticker,
        "n_obs": n,
        "n_splits": n_splits,
        "n_test_splits": n_test_splits,
        "is_sharpes": [float(x) for x in is_sharpes],
        "oos_sharpes": [float(x) for x in oos_sharpes],
        "is_sharpe_mean": is_mean,
        "oos_sharpe_mean": oos_mean,
        "pbo": pbo_result,
        "deflated_sharpe": dsr,
    }
