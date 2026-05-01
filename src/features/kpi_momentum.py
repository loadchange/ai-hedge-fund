"""KPI momentum: z-scored YoY changes for cross-sectional ranking.

For each headline metric (revenue, EPS, gross_margin, …), compute the
year-over-year percentage change, then z-score it against a rolling
historical window. Z-scores let you compare momentum across companies
with very different scales (a 5% revenue uplift means very different
things for a software company vs. a bank).
"""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np


def compute_kpi_momentum(
    metrics: Iterable[Mapping],
    *,
    fields: Iterable[str] = ("revenue", "earnings_per_share", "gross_margin", "operating_margin"),
    history_window: int = 8,
) -> dict[str, float | None]:
    """Z-scored YoY change for each requested KPI.

    Args:
        metrics: chronologically-ordered iterable of dict-like financial
            metrics objects (oldest → newest). Anything with ``getattr`` /
            ``__getitem__`` access works (Pydantic models, plain dicts).
        fields: KPI names to score. Each must be a numeric attribute on
            the metric objects.
        history_window: how many periods of YoY history to compute the
            z-score against. Default 8 ≈ two years of quarterly data.

    Returns:
        ``{field: z_score | None}`` — None when there's insufficient
        history or the latest YoY is undefined (e.g. division by zero).
    """
    items = list(metrics)
    if len(items) < 5:
        return {f: None for f in fields}

    out: dict[str, float | None] = {}
    for field in fields:
        values = np.array(
            [_extract(it, field) for it in items],
            dtype=float,
        )
        # Require periods spaced ~1 year apart for a YoY; assume the
        # caller passes quarterly metrics, so YoY = pct change with lag 4.
        yoy = _yoy_changes(values, lag=4)
        if yoy is None or len(yoy) < 4 or np.isnan(yoy[-1]):
            out[field] = None
            continue

        history = yoy[-(history_window + 1):-1]
        history = history[~np.isnan(history)]
        if len(history) < 3:
            out[field] = None
            continue
        mean = float(np.mean(history))
        std = float(np.std(history, ddof=1))
        if std <= 0:
            out[field] = None
            continue
        out[field] = float((yoy[-1] - mean) / std)
    return out


def _extract(obj, field: str):
    if hasattr(obj, field):
        v = getattr(obj, field)
    else:
        try:
            v = obj[field]
        except Exception:
            return float("nan")
    if v is None:
        return float("nan")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def _yoy_changes(values: np.ndarray, *, lag: int = 4) -> np.ndarray | None:
    """Percentage YoY change with the given ``lag`` (default 4 quarters)."""
    if len(values) <= lag:
        return None
    yoy = np.full_like(values, np.nan, dtype=float)
    for i in range(lag, len(values)):
        prev = values[i - lag]
        curr = values[i]
        if np.isnan(prev) or np.isnan(curr) or prev == 0:
            yoy[i] = np.nan
        else:
            yoy[i] = (curr - prev) / abs(prev)
    return yoy
