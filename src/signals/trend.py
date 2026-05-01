"""Multi-timeframe trend-following signal (EMA stack + ADX strength).

Logic preserved bit-for-bit from the original
``src/agents/technicals.py:calculate_trend_signals``.
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import calculate_adx, calculate_ema, safe_float


class TrendFollowingSignal(BaseSignal):
    """Bullish when EMA(8) > EMA(21) > EMA(55); ADX gives the strength."""

    @property
    def name(self) -> str:
        return "trend"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        ema_8 = calculate_ema(prices_df, 8)
        ema_21 = calculate_ema(prices_df, 21)
        ema_55 = calculate_ema(prices_df, 55)
        adx = calculate_adx(prices_df, 14)

        short_trend = ema_8 > ema_21
        medium_trend = ema_21 > ema_55
        trend_strength = adx["adx"].iloc[-1] / 100.0

        if short_trend.iloc[-1] and medium_trend.iloc[-1]:
            direction = "bullish"
            confidence = trend_strength
        elif not short_trend.iloc[-1] and not medium_trend.iloc[-1]:
            direction = "bearish"
            confidence = trend_strength
        else:
            direction = "neutral"
            confidence = 0.5

        confidence = safe_float(confidence, 0.5)
        sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)

        return SignalResult(
            signal_name=self.name,
            value=self._normalize_to_signal(sign * confidence),
            direction=direction,
            confidence=confidence,
            components={
                "adx": safe_float(adx["adx"].iloc[-1]),
                "trend_strength": safe_float(trend_strength),
            },
        )
