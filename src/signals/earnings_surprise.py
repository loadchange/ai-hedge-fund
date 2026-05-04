"""Earnings-surprise signal — score the latest EPS beat/miss vs. estimate.

Pulls quarterly earnings dates + actuals + estimates from yfinance
(``Ticker.earnings_dates``) and computes a SUE-like score: positive
when actual EPS beats the consensus estimate, negative on a miss,
scaled by the magnitude of the surprise relative to estimate. The
signal degrades to neutral for tickers yfinance can't provide
earnings for (most A-shares, some HK names).
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult


class EarningsSurpriseSignal(BaseSignal):
    """SUE-style score from the latest quarterly earnings surprise."""

    @property
    def name(self) -> str:
        return "earnings_surprise"

    @property
    def kind(self) -> str:
        return "fundamental"

    def compute_from_prices(self, prices_df):  # noqa: ARG002
        raise NotImplementedError(
            "EarningsSurpriseSignal requires earnings data — use compute(ticker, end_date)."
        )

    def compute(self, ticker: str, end_date: str, **kwargs) -> SignalResult:  # noqa: ARG002
        try:
            import yfinance as yf
        except ImportError:
            return self._neutral("yfinance not installed")

        try:
            df = yf.Ticker(ticker).earnings_dates
        except Exception as e:
            return self._neutral(f"yfinance earnings_dates error: {e}")

        if df is None or df.empty:
            return self._neutral("no earnings data available")

        # Keep only past earnings on/before end_date that have actual EPS reported.
        end_ts = pd.to_datetime(end_date)
        df = df.copy()
        # yfinance index is a tz-aware DatetimeIndex; strip tz for comparison.
        try:
            df.index = df.index.tz_localize(None)
        except (TypeError, AttributeError):
            pass
        df = df[df.index <= end_ts]
        eps_actual_col = "Reported EPS" if "Reported EPS" in df.columns else "EPS Actual"
        eps_estimate_col = "EPS Estimate"
        if eps_actual_col not in df.columns or eps_estimate_col not in df.columns:
            return self._neutral("yfinance schema missing EPS columns")
        df = df[df[eps_actual_col].notna() & df[eps_estimate_col].notna()]
        if df.empty:
            return self._neutral("no actual-vs-estimate EPS rows")

        latest = df.sort_index(ascending=False).iloc[0]
        actual = self._safe_float(latest[eps_actual_col])
        estimate = self._safe_float(latest[eps_estimate_col])
        if not estimate:
            return self._neutral("estimate is zero/missing")

        relative = (actual - estimate) / abs(estimate)
        composite = self._normalize_to_signal(self._sigmoid(relative, scale=10.0))

        components = {
            "eps_actual": actual,
            "eps_estimate": estimate,
            "eps_surprise_relative": relative,
        }

        if composite > 0.2:
            direction, confidence = "bullish", abs(composite)
        elif composite < -0.2:
            direction, confidence = "bearish", abs(composite)
        else:
            direction, confidence = "neutral", 0.5

        return SignalResult(
            signal_name=self.name,
            value=composite,
            direction=direction,
            confidence=confidence,
            components=components,
            metadata={"report_period": df.sort_index(ascending=False).index[0].strftime("%Y-%m-%d")},
        )

    def _neutral(self, note: str) -> SignalResult:
        return SignalResult(
            signal_name=self.name,
            value=0.0,
            direction="neutral",
            confidence=0.5,
            components={},
            metadata={"note": note},
        )
