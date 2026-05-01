"""Abnormal returns (AR), cumulative abnormal returns (CAR), and CAAR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.event_study.market_model import MarketModel


def compute_abnormal_returns(
    asset_returns: pd.Series,
    benchmark_returns: pd.Series,
    market_model: MarketModel,
) -> pd.Series:
    """Per-period abnormal return: ``AR_t = r_asset_t - (α + β · r_bench_t)``.

    Indices of the two return series must align (use ``concat`` then
    ``dropna`` upstream if necessary).
    """
    aligned = pd.concat([asset_returns, benchmark_returns], axis=1).dropna()
    aligned.columns = ["asset", "bench"]
    expected = market_model.predict(aligned["bench"].to_numpy())
    ar = aligned["asset"].to_numpy() - expected
    return pd.Series(ar, index=aligned.index, name="abnormal_return")


def compute_car(
    abnormal_returns: pd.Series,
    *,
    event_window: tuple[int, int] = (-3, +3),
    event_index: int | pd.Timestamp | None = None,
) -> float:
    """Sum of abnormal returns inside an event window.

    Args:
        abnormal_returns: AR series (output of :func:`compute_abnormal_returns`).
        event_window: ``(t_start, t_end)`` inclusive bounds in trading
            days relative to the event (e.g. ``(-3, +3)`` = the 7-day
            window centred on the event day).
        event_index: position of the event. Pass an int (positional)
            or a Timestamp (label-based). Defaults to the *last*
            observation, which is convenient for streaming use.
    """
    if abnormal_returns.empty:
        return 0.0

    if event_index is None:
        idx = len(abnormal_returns) - 1
    elif isinstance(event_index, (int, np.integer)):
        idx = int(event_index)
    else:
        loc = abnormal_returns.index.get_indexer([event_index])
        if len(loc) == 0 or loc[0] == -1:
            raise KeyError(f"Event date {event_index!r} not in AR series")
        idx = int(loc[0])

    t_start, t_end = event_window
    start = max(0, idx + t_start)
    end = min(len(abnormal_returns), idx + t_end + 1)
    return float(abnormal_returns.iloc[start:end].sum())


def compute_caar(cars: list[float]) -> float:
    """Cross-sectional average of CARs across events of the same type."""
    arr = np.asarray(cars, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) == 0:
        return float("nan")
    return float(arr.mean())
