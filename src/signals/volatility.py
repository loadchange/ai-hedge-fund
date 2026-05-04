"""Volatility regime signal — bullish on vol expansion from a low base.

Logic preserved bit-for-bit from the original
``src/agents/technicals.py:calculate_volatility_signals``.
"""

from __future__ import annotations

import math

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import calculate_atr, safe_float


class VolatilitySignal(BaseSignal):
    """Detect regime shifts: contraction → expansion is bullish, sustained high vol bearish."""

    @property
    def name(self) -> str:
        return "volatility"

    @property
    def kind(self) -> str:
        return "technical"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        returns = prices_df["close"].pct_change()
        hist_vol = returns.rolling(21).std() * math.sqrt(252)
        vol_ma = hist_vol.rolling(63).mean()
        vol_regime = hist_vol / vol_ma
        vol_z_score = (hist_vol - vol_ma) / hist_vol.rolling(63).std()

        atr = calculate_atr(prices_df)
        atr_ratio = atr / prices_df["close"]

        current_vol_regime = vol_regime.iloc[-1]
        vol_z = vol_z_score.iloc[-1]

        if current_vol_regime < 0.8 and vol_z < -1:
            direction = "bullish"
            confidence = min(abs(vol_z) / 3, 1.0)
        elif current_vol_regime > 1.2 and vol_z > 1:
            direction = "bearish"
            confidence = min(abs(vol_z) / 3, 1.0)
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
                "historical_volatility": safe_float(hist_vol.iloc[-1]),
                "volatility_regime": safe_float(current_vol_regime),
                "volatility_z_score": safe_float(vol_z),
                "atr_ratio": safe_float(atr_ratio.iloc[-1]),
            },
        )
