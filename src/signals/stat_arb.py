"""Statistical-arbitrage signal — Hurst exponent + return distribution skew.

Logic preserved bit-for-bit from the original
``src/agents/technicals.py:calculate_stat_arb_signals``.
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import calculate_hurst_exponent, safe_float


class StatArbSignal(BaseSignal):
    """Mean-reverting series (Hurst < 0.4) with strong skew → directional bet."""

    @property
    def name(self) -> str:
        return "stat_arb"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        returns = prices_df["close"].pct_change()
        skew = returns.rolling(63).skew()
        kurt = returns.rolling(63).kurt()

        hurst = calculate_hurst_exponent(prices_df["close"])

        if hurst < 0.4 and skew.iloc[-1] > 1:
            direction = "bullish"
            confidence = (0.5 - hurst) * 2
        elif hurst < 0.4 and skew.iloc[-1] < -1:
            direction = "bearish"
            confidence = (0.5 - hurst) * 2
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
                "hurst_exponent": safe_float(hurst),
                "skewness": safe_float(skew.iloc[-1]),
                "kurtosis": safe_float(kurt.iloc[-1]),
            },
        )
