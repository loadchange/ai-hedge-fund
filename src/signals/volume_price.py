"""Volume-price analysis signal (量价分析).

Detects volume-price alignment vs divergence:
- Bullish: price up + volume expanding, or accumulation phase
- Bearish: price up + volume shrinking (bearish divergence), or distribution
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult
from src.signals.utils import safe_float


class VolumePriceSignal(BaseSignal):
    """Volume-price relationship signal."""

    @property
    def name(self) -> str:
        return "volume_price"

    @property
    def kind(self) -> str:
        return "technical"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:
        df = prices_df.tail(60).copy()
        if len(df) < 20:
            return self._empty_result()

        close = df["close"]
        volume = df["volume"].astype(float)

        latest_vol = float(volume.iloc[-1])
        avg_vol_5 = float(volume.tail(5).mean())
        avg_vol_20 = float(volume.tail(20).mean())

        vol_ratio_5d = safe_float(latest_vol / avg_vol_5, 1.0) if avg_vol_5 > 0 else 1.0
        vol_ratio_20d = safe_float(latest_vol / avg_vol_20, 1.0) if avg_vol_20 > 0 else 1.0

        # Up vs down day volume
        price_up = close.diff() > 0
        up_days = df[price_up]
        down_days = df[~price_up]
        up_vol_avg = float(up_days["volume"].astype(float).mean()) if len(up_days) > 0 else 0
        down_vol_avg = float(down_days["volume"].astype(float).mean()) if len(down_days) > 0 else 0

        # Volume trend (last 5d avg vs prior 5d avg)
        if len(volume) >= 10:
            recent_5 = float(volume.tail(5).mean())
            prior_5 = float(volume.iloc[-10:-5].mean())
            vol_trend_pct = safe_float((recent_5 - prior_5) / prior_5 * 100, 0) if prior_5 > 0 else 0
            vol_trend = "expanding" if vol_trend_pct > 20 else "shrinking" if vol_trend_pct < -20 else "stable"
        else:
            vol_trend_pct = 0
            vol_trend = "insufficient_data"

        # Volume-price correlation (20d)
        try:
            vp_corr = float(volume.tail(20).corr(close.tail(20).pct_change()))
            vp_corr = safe_float(vp_corr, 0)
        except Exception:
            vp_corr = 0.0

        # Pattern classification
        price_up_latest = close.iloc[-1] > close.iloc[-2] if len(close) >= 2 else False
        pattern = "neutral"
        if price_up_latest and vol_ratio_5d > 1.2:
            pattern = "volume_price_alignment_bullish"
        elif not price_up_latest and vol_ratio_5d > 1.2:
            pattern = "volume_price_alignment_bearish"
        elif price_up_latest and vol_ratio_5d < 0.7:
            pattern = "bearish_divergence"  # price up but volume shrinking
        elif not price_up_latest and vol_ratio_5d < 0.7:
            pattern = "potential_reversal"  # selling exhaustion

        # Signal determination
        direction = "neutral"
        confidence = 0.5

        if pattern == "volume_price_alignment_bullish" and up_vol_avg > down_vol_avg:
            direction = "bullish"
            confidence = min(0.9, 0.5 + vol_ratio_5d * 0.1)
        elif pattern == "volume_price_alignment_bearish" and down_vol_avg > up_vol_avg:
            direction = "bearish"
            confidence = min(0.9, 0.5 + vol_ratio_5d * 0.1)
        elif pattern == "bearish_divergence":
            direction = "bearish"
            confidence = 0.4 + (1.0 - vol_ratio_5d) * 0.2
        elif pattern == "potential_reversal":
            direction = "bullish"
            confidence = 0.35

        confidence = safe_float(confidence, 0.5)
        sign = 1.0 if direction == "bullish" else (-1.0 if direction == "bearish" else 0.0)

        return SignalResult(
            signal_name=self.name,
            value=self._normalize_to_signal(sign * confidence),
            direction=direction,
            confidence=confidence,
            components={
                "volume_ratio_5d": safe_float(vol_ratio_5d),
                "volume_ratio_20d": safe_float(vol_ratio_20d),
                "up_volume_avg": safe_float(up_vol_avg),
                "down_volume_avg": safe_float(down_vol_avg),
                "volume_trend_pct": safe_float(vol_trend_pct),
                "volume_price_corr": safe_float(vp_corr),
            },
            metadata={
                "volume_trend": vol_trend,
                "pattern": pattern,
            },
        )
