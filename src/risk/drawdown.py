"""Drawdown analytics + threshold triggers.

Drawdown is computed against the running cumulative max of an equity
curve. Use :func:`drawdown_stats` for a one-shot summary of an entire
backtest, :func:`should_halt_trading` to gate live decisions on a
maximum-drawdown stop, and :func:`drawdown_series` if you need the raw
underwater series for plotting.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DrawdownStats:
    """Summary of an equity curve's drawdown profile.

    All percentages are *fractional* (``-0.20`` means -20%), times are
    pandas Timestamps when the input series is indexed by date.
    """

    max_drawdown: float
    max_drawdown_date: pd.Timestamp | None
    peak_date_before_max: pd.Timestamp | None
    current_drawdown: float
    underwater_days: int
    longest_underwater_days: int


def drawdown_series(equity: pd.Series) -> pd.Series:
    """Per-period drawdown (fractional). ``0`` while at a new high."""
    equity = pd.Series(equity).dropna().astype(float)
    if equity.empty:
        return equity
    running_max = equity.cummax()
    return (equity - running_max) / running_max


def drawdown_stats(equity: pd.Series) -> DrawdownStats:
    """Summary stats for an equity curve.

    ``equity`` should be portfolio NLV per period (pandas Series indexed by date).
    """
    equity = pd.Series(equity).dropna().astype(float)
    if equity.empty:
        return DrawdownStats(0.0, None, None, 0.0, 0, 0)

    running_max = equity.cummax()
    dd = (equity - running_max) / running_max

    max_dd_idx = dd.idxmin()
    max_dd = float(dd.loc[max_dd_idx])

    # Peak that preceded the trough.
    pre_trough = equity.loc[:max_dd_idx]
    peak_idx = pre_trough.idxmax() if len(pre_trough) else None

    current_dd = float(dd.iloc[-1])

    # Underwater duration: consecutive periods with dd < 0.
    underwater_mask = dd < -1e-12  # tolerate float noise at peaks
    longest = current_streak = 0
    streak = 0
    for under in underwater_mask:
        if under:
            streak += 1
            longest = max(longest, streak)
        else:
            streak = 0
    # Streak from the right edge — current consecutive run.
    for under in underwater_mask[::-1]:
        if under:
            current_streak += 1
        else:
            break

    return DrawdownStats(
        max_drawdown=max_dd,
        max_drawdown_date=max_dd_idx if isinstance(max_dd_idx, pd.Timestamp) else None,
        peak_date_before_max=peak_idx if isinstance(peak_idx, pd.Timestamp) else None,
        current_drawdown=current_dd,
        underwater_days=current_streak,
        longest_underwater_days=longest,
    )


def should_halt_trading(equity: pd.Series, *, max_drawdown_threshold: float = -0.25) -> bool:
    """Return True when the current drawdown breaches *max_drawdown_threshold*.

    *max_drawdown_threshold* is fractional and negative (default -25%).
    Use this as a circuit-breaker in the portfolio manager: when True,
    pause new entries / unwind risky positions.
    """
    if equity is None or len(equity) == 0:
        return False
    stats = drawdown_stats(equity)
    return stats.current_drawdown <= max_drawdown_threshold
