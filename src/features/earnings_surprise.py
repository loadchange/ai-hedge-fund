"""Earnings-surprise features.

Standardised Unexpected Earnings (SUE) and Post-Earnings-Announcement
Drift (PEAD). Both are well-documented anomalies — SUE > +1 historically
predicts ~1-2% positive drift over the following 60 trading days
(Bernard & Thomas 1989).
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def standardised_unexpected_earnings(
    actuals: Sequence[float],
    estimates: Sequence[float],
    *,
    history_window: int = 8,
) -> float:
    """SUE = (latest actual − latest estimate) / std(historical surprises).

    Args:
        actuals: chronological actual EPS values (oldest → newest).
        estimates: aligned consensus estimates (same length).
        history_window: number of historical surprises used to compute
            the standard deviation. Default 8 quarters.

    Returns:
        SUE score. ``> +1`` is a meaningful positive surprise; ``< -1`` negative.
        Returns ``nan`` when there's not enough history (need ≥ 4 surprises).
    """
    a = np.asarray(actuals, dtype=float)
    e = np.asarray(estimates, dtype=float)
    if len(a) != len(e) or len(a) < 4:
        return float("nan")

    surprises = a - e
    if np.isnan(surprises[-1]):
        return float("nan")

    history = surprises[-(history_window + 1):-1]  # exclude the latest
    history = history[~np.isnan(history)]
    if len(history) < 3:
        return float("nan")

    std = float(np.std(history, ddof=1))
    if std <= 0:
        return float("nan")

    return float(surprises[-1] / std)


def pead_drift(
    abnormal_returns: pd.Series,
    *,
    event_index: int | pd.Timestamp,
    drift_window: tuple[int, int] = (1, 60),
) -> float:
    """Cumulative drift after an earnings event.

    Sums abnormal returns from ``event_index + window[0]`` through
    ``event_index + window[1]`` (inclusive). Default ``(1, 60)``
    captures the classic PEAD horizon — the 60-day post-announcement
    window where surprise momentum tends to materialise.
    """
    ar = pd.Series(abnormal_returns).astype(float)
    if ar.empty:
        return 0.0

    if isinstance(event_index, (int, np.integer)):
        idx = int(event_index)
    else:
        loc = ar.index.get_indexer([event_index])
        if len(loc) == 0 or loc[0] == -1:
            raise KeyError(f"Event date {event_index!r} not in series")
        idx = int(loc[0])

    t_start, t_end = drift_window
    start = max(0, idx + t_start)
    end = min(len(ar), idx + t_end + 1)
    if end <= start:
        return 0.0
    return float(ar.iloc[start:end].sum())
