"""Correlation matrix + average-correlation adjustment factor."""

from __future__ import annotations

from typing import Mapping

import pandas as pd


def build_correlation_matrix(
    returns_by_ticker: Mapping[str, pd.Series],
    *,
    min_overlap: int = 5,
) -> pd.DataFrame | None:
    """Pearson correlation matrix from per-ticker daily-return series.

    Returns ``None`` when fewer than 2 tickers have at least *min_overlap*
    aligned observations (calling code should treat that as "no
    correlation adjustment").
    """
    if len(returns_by_ticker) < 2:
        return None
    try:
        returns_df = pd.DataFrame(returns_by_ticker).dropna(how="any")
    except Exception:
        return None
    if returns_df.shape[1] < 2 or returns_df.shape[0] < min_overlap:
        return None
    return returns_df.corr()


def calculate_correlation_multiplier(avg_correlation: float) -> float:
    """Scale a position limit based on average pairwise correlation.

    Same staircase as the original ``risk_manager.py``:

    * avg ≥ 0.80 → 0.70x
    * avg ≥ 0.60 → 0.85x
    * avg ≥ 0.40 → 1.00x
    * avg ≥ 0.20 → 1.05x
    * avg < 0.20 → 1.10x
    """
    if avg_correlation >= 0.80:
        return 0.70
    if avg_correlation >= 0.60:
        return 0.85
    if avg_correlation >= 0.40:
        return 1.00
    if avg_correlation >= 0.20:
        return 1.05
    return 1.10


def correlation_summary(
    correlation_matrix: pd.DataFrame | None,
    ticker: str,
    *,
    compare_with: list[str] | None = None,
    top_k: int = 3,
) -> dict:
    """Pull avg / max / top-K correlations for *ticker* against *compare_with*.

    Returns the same shape consumed by ``risk_manager.py``::

        {
            "avg_correlation_with_active": float | None,
            "max_correlation_with_active": float | None,
            "top_correlated_tickers": [{"ticker": str, "correlation": float}, ...],
        }

    *compare_with* defaults to every other column in the matrix.
    """
    out: dict = {
        "avg_correlation_with_active": None,
        "max_correlation_with_active": None,
        "top_correlated_tickers": [],
    }
    if correlation_matrix is None or ticker not in correlation_matrix.columns:
        return out

    if compare_with is None:
        compare_with = [t for t in correlation_matrix.columns if t != ticker]
    else:
        compare_with = [t for t in compare_with if t in correlation_matrix.columns and t != ticker]

    if not compare_with:
        return out

    series = correlation_matrix.loc[ticker, compare_with].dropna()
    if len(series) == 0:
        return out

    avg = float(series.mean())
    mx = float(series.max())
    out["avg_correlation_with_active"] = avg
    out["max_correlation_with_active"] = mx
    top = series.sort_values(ascending=False).head(top_k)
    out["top_correlated_tickers"] = [
        {"ticker": idx, "correlation": float(val)} for idx, val in top.items()
    ]
    return out
