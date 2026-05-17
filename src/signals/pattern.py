"""Candlestick / chart pattern recognition signal (K线形态).

Detects single-candle and multi-candle patterns from OHLCV data.
Signal direction is determined by the balance of bullish vs bearish patterns.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import calculate_atr, safe_float


class PatternSignal(BaseSignal):
    """Chart pattern recognition signal."""

    @property
    def name(self) -> str:
        return "pattern"

    @property
    def kind(self) -> str:
        return "technical"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        df = prices_df.tail(120).copy()
        if len(df) < 20:
            return self._empty_result()

        df = df.tail(60).reset_index(drop=True)
        o = df["open"].values.astype(float)
        h = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        c = df["close"].values.astype(float)
        v = df["volume"].values.astype(float) if "volume" in df.columns else None
        n = len(c)

        bullish_patterns: list[str] = []
        bearish_patterns: list[str] = []

        def body(i: int) -> float:
            return abs(c[i] - o[i])

        def upper_shadow(i: int) -> float:
            return h[i] - max(c[i], o[i])

        def lower_shadow(i: int) -> float:
            return min(c[i], o[i]) - low[i]

        avg_body = max(sum(body(i) for i in range(n)) / n, 1e-8)
        atr = float(calculate_atr(df, 14).iloc[-1]) if n >= 14 else avg_body

        # Single-candle patterns (last 3 days)
        for i in range(max(0, n - 3), n):
            bd = body(i)
            us = upper_shadow(i)
            ls = lower_shadow(i)
            total_range = h[i] - low[i]

            # Doji
            if total_range > 0 and bd < 0.1 * total_range and (us + ls) > bd * 3:
                if i > 0 and c[i - 1] > o[i - 1]:
                    bearish_patterns.append("doji_after_rally")
                elif i > 0 and c[i - 1] < o[i - 1]:
                    bullish_patterns.append("doji_after_decline")

            # Hammer
            if bd > 0 and ls > 2 * bd and us < 0.5 * bd:
                bullish_patterns.append("hammer")

            # Shooting star
            if bd > 0 and us > 2 * bd and ls < 0.5 * bd:
                bearish_patterns.append("shooting_star")

            # Big bullish candle
            if bd > 1.5 * atr and c[i] > o[i]:
                bullish_patterns.append("big_bullish")

            # Big bearish candle
            if bd > 1.5 * atr and c[i] < o[i]:
                bearish_patterns.append("big_bearish")

        # Multi-candle patterns
        if n >= 3:
            i = n - 1
            # Morning Star
            if (
                c[i - 2] < o[i - 2] and body(i - 2) > avg_body * 1.5
                and body(i - 1) < avg_body * 0.4
                and c[i] > o[i] and body(i) > avg_body * 1.5
                and c[i] > (o[i - 2] + c[i - 2]) / 2
            ):
                bullish_patterns.append("morning_star")

            # Evening Star
            if (
                c[i - 2] > o[i - 2] and body(i - 2) > avg_body * 1.5
                and body(i - 1) < avg_body * 0.4
                and c[i] < o[i] and body(i) > avg_body * 1.5
                and c[i] < (o[i - 2] + c[i - 2]) / 2
            ):
                bearish_patterns.append("evening_star")

            # Bullish Engulfing
            if (
                c[i] > o[i] and c[i - 1] < o[i - 1]
                and o[i] < c[i - 1] and c[i] > o[i - 1]
            ):
                bullish_patterns.append("bullish_engulfing")

            # Bearish Engulfing
            if (
                c[i] < o[i] and c[i - 1] > o[i - 1]
                and o[i] > c[i - 1] and c[i] < o[i - 1]
            ):
                bearish_patterns.append("bearish_engulfing")

        # Chart patterns
        if n >= 21:
            # 20-day breakout with volume confirmation
            high_20d = max(h[n - 21: n - 1])
            if c[-1] > high_20d and v is not None and v[-1] > sum(v[n - 6:n - 1]) / 5 * 1.5:
                bullish_patterns.append("breakout_20d_high")

        if n >= 10:
            # Box consolidation
            recent_high = max(h[n - 10:])
            recent_low = min(low[n - 10:])
            box_range_pct = (recent_high - recent_low) / recent_low * 100 if recent_low > 0 else 0
            if box_range_pct < 8:
                if v is not None and len(v) >= 2:
                    pass  # neutral, don't add

            # Double bottom (simplified)
            recent_lows = sorted(range(n), key=lambda idx: low[idx])[:5]
            if len(recent_lows) >= 2:
                lo1, lo2 = sorted(recent_lows[:2])
                if lo2 - lo1 >= 5 and abs(low[lo1] - low[lo2]) / max(low[lo1], low[lo2]) < 0.03:
                    mid_high = max(h[lo1:lo2 + 1])
                    if mid_high > low[lo1] * 1.03:
                        bullish_patterns.append("double_bottom")

        # Determine signal
        bull_count = len(bullish_patterns)
        bear_count = len(bearish_patterns)

        if bull_count >= 2 and bear_count == 0:
            direction = "bullish"
            confidence = min(0.9, bull_count / 4)
        elif bear_count >= 2 and bull_count == 0:
            direction = "bearish"
            confidence = min(0.9, bear_count / 4)
        elif bull_count > bear_count:
            direction = "bullish"
            confidence = (bull_count - bear_count) / 6
        elif bear_count > bull_count:
            direction = "bearish"
            confidence = (bear_count - bull_count) / 6
        else:
            direction = "neutral"
            confidence = 0.3

        confidence = safe_float(confidence, 0.3)
        sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)

        return SignalResult(
            signal_name=self.name,
            value=self._normalize_to_signal(sign * confidence),
            direction=direction,
            confidence=confidence,
            components={
                "bullish_count": bull_count,
                "bearish_count": bear_count,
            },
            metadata={
                "bullish_patterns": bullish_patterns,
                "bearish_patterns": bearish_patterns,
            },
        )
