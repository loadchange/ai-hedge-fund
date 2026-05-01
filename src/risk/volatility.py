"""Volatility primitives.

Extracted from ``src/agents/risk_manager.py`` so signal / portfolio /
validation modules can use the same math without dragging the LangGraph
agent into their import graph.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def calculate_volatility_metrics(
    prices_df: pd.DataFrame, lookback_days: int = 60
) -> dict:
    """Compute daily/annualised vol and a percentile rank vs. history.

    Returns a dict with keys ``daily_volatility`` / ``annualized_volatility``
    / ``volatility_percentile`` / ``data_points``. Defaults are robust
    fallbacks (5% daily / annualised, 100th percentile = "treat as risky").
    """
    fallback = {
        "daily_volatility": 0.05,
        "annualized_volatility": 0.05 * math.sqrt(252),
        "volatility_percentile": 100.0,
        "data_points": 0,
    }

    if prices_df is None or prices_df.empty or len(prices_df) < 2:
        return {**fallback, "data_points": 0 if prices_df is None else len(prices_df)}

    daily_returns = prices_df["close"].pct_change().dropna()
    if len(daily_returns) < 2:
        return {**fallback, "data_points": len(daily_returns)}

    recent_returns = daily_returns.tail(min(lookback_days, len(daily_returns)))
    daily_vol = recent_returns.std()
    annualized_vol = daily_vol * math.sqrt(252)

    # Percentile rank of recent vol against historical 30-day rolling vol.
    if len(daily_returns) >= 30:
        rolling_vol = daily_returns.rolling(window=30).std().dropna()
        if len(rolling_vol) > 0:
            current_pct = float((rolling_vol <= daily_vol).mean() * 100)
        else:
            current_pct = 50.0
    else:
        current_pct = 50.0

    return {
        "daily_volatility": float(daily_vol) if not np.isnan(daily_vol) else 0.025,
        "annualized_volatility": float(annualized_vol) if not np.isnan(annualized_vol) else 0.25,
        "volatility_percentile": float(current_pct) if not np.isnan(current_pct) else 50.0,
        "data_points": len(recent_returns),
    }


def calculate_volatility_adjusted_limit(annualized_volatility: float) -> float:
    """Position-limit fraction (of NLV) given a stock's annualised vol.

    Curve preserved bit-for-bit from ``risk_manager.py``:

    * vol < 15%   → 25% allocation cap
    * 15-30%      → 12.5-20% (linear taper)
    * 30-50%      → 5-15%
    * vol > 50%   → 10% (floor)
    """
    base_limit = 0.20

    if annualized_volatility < 0.15:
        vol_multiplier = 1.25
    elif annualized_volatility < 0.30:
        vol_multiplier = 1.0 - (annualized_volatility - 0.15) * 0.5
    elif annualized_volatility < 0.50:
        vol_multiplier = 0.75 - (annualized_volatility - 0.30) * 0.5
    else:
        vol_multiplier = 0.50

    vol_multiplier = max(0.25, min(1.25, vol_multiplier))
    return base_limit * vol_multiplier
