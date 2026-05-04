"""Multi-horizon momentum signal with volume confirmation.

Logic preserved bit-for-bit from the original
``src/agents/technicals.py:calculate_momentum_signals``.
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import safe_float


class MomentumSignal(BaseSignal):
    """Weighted 1m / 3m / 6m return momentum, gated by volume."""

    @property
    def name(self) -> str:
        return "momentum"

    @property
    def kind(self) -> str:
        return "technical"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        returns = prices_df["close"].pct_change()
        mom_1m = returns.rolling(21).sum()
        mom_3m = returns.rolling(63).sum()
        mom_6m = returns.rolling(126).sum()

        volume_ma = prices_df["volume"].rolling(21).mean()
        volume_momentum = prices_df["volume"] / volume_ma

        momentum_score = (0.4 * mom_1m + 0.3 * mom_3m + 0.3 * mom_6m).iloc[-1]
        volume_confirmation = bool(volume_momentum.iloc[-1] > 1.0) if pd.notna(volume_momentum.iloc[-1]) else False

        if pd.notna(momentum_score) and momentum_score > 0.05 and volume_confirmation:
            direction = "bullish"
            confidence = min(abs(momentum_score) * 5, 1.0)
        elif pd.notna(momentum_score) and momentum_score < -0.05 and volume_confirmation:
            direction = "bearish"
            confidence = min(abs(momentum_score) * 5, 1.0)
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
                "momentum_1m": safe_float(mom_1m.iloc[-1]),
                "momentum_3m": safe_float(mom_3m.iloc[-1]),
                "momentum_6m": safe_float(mom_6m.iloc[-1]),
                "volume_momentum": safe_float(volume_momentum.iloc[-1]),
            },
        )
