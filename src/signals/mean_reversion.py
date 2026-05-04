"""Mean-reversion signal using z-score, Bollinger Bands and RSI.

Logic preserved bit-for-bit from the original
``src/agents/technicals.py:calculate_mean_reversion_signals``.
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import calculate_bollinger_bands, calculate_rsi, safe_float


class MeanReversionSignal(BaseSignal):
    """Bullish when price is far below its 50d mean and at the lower BB band."""

    @property
    def name(self) -> str:
        return "mean_reversion"

    @property
    def kind(self) -> str:
        return "technical"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        ma_50 = prices_df["close"].rolling(window=50).mean()
        std_50 = prices_df["close"].rolling(window=50).std()
        z_score = (prices_df["close"] - ma_50) / std_50

        bb_upper, bb_lower = calculate_bollinger_bands(prices_df)
        rsi_14 = calculate_rsi(prices_df, 14)
        rsi_28 = calculate_rsi(prices_df, 28)

        price_vs_bb = (prices_df["close"].iloc[-1] - bb_lower.iloc[-1]) / (
            bb_upper.iloc[-1] - bb_lower.iloc[-1]
        )

        if z_score.iloc[-1] < -2 and price_vs_bb < 0.2:
            direction = "bullish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
        elif z_score.iloc[-1] > 2 and price_vs_bb > 0.8:
            direction = "bearish"
            confidence = min(abs(z_score.iloc[-1]) / 4, 1.0)
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
                "z_score": safe_float(z_score.iloc[-1]),
                "price_vs_bb": safe_float(price_vs_bb),
                "rsi_14": safe_float(rsi_14.iloc[-1]),
                "rsi_28": safe_float(rsi_28.iloc[-1]),
            },
        )
