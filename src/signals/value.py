"""Value signal — composite score from P/E, P/B, P/S, EV/EBITDA percentiles.

A ticker scores +1 when it is "cheap" relative to the cross-sectional
distribution implied by the latest financial metrics. Lower multiples
imply a higher (more positive) signal.
"""

from __future__ import annotations

import pandas as pd

from src.signals.base import BaseSignal
from src.signals.types import SignalResult


class ValueSignal(BaseSignal):
    """Composite cheapness score from valuation multiples."""

    @property
    def name(self) -> str:
        return "value"

    def compute_from_prices(self, prices_df: pd.DataFrame) -> SignalResult:  # noqa: ARG002
        # Value signal needs financial metrics, not prices.
        raise NotImplementedError(
            "ValueSignal requires financial metrics — use compute(ticker, end_date)."
        )

    def compute(self, ticker: str, end_date: str, **kwargs) -> SignalResult:
        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=1, api_key=kwargs.get("api_key"))
        if not metrics:
            return self._empty_result()
        m = metrics[0]

        components: dict[str, float] = {}
        scores: list[float] = []

        # Lower P/E is better; use sigmoid centred at 15.
        pe = self._safe_float(getattr(m, "price_to_earnings_ratio", None), float("nan"))
        if pe and pe > 0 and pe < 200:
            components["price_to_earnings_ratio"] = pe
            scores.append(self._sigmoid((15 - pe) / 10, scale=1.0))

        pb = self._safe_float(getattr(m, "price_to_book_ratio", None), float("nan"))
        if pb and pb > 0 and pb < 50:
            components["price_to_book_ratio"] = pb
            scores.append(self._sigmoid((2.5 - pb) / 1.5, scale=1.0))

        ps = self._safe_float(getattr(m, "price_to_sales_ratio", None), float("nan"))
        if ps and ps > 0 and ps < 50:
            components["price_to_sales_ratio"] = ps
            scores.append(self._sigmoid((3 - ps) / 2, scale=1.0))

        ev_ebitda = self._safe_float(getattr(m, "enterprise_value_to_ebitda_ratio", None), float("nan"))
        if ev_ebitda and ev_ebitda > 0 and ev_ebitda < 100:
            components["enterprise_value_to_ebitda_ratio"] = ev_ebitda
            scores.append(self._sigmoid((10 - ev_ebitda) / 6, scale=1.0))

        fcf_yield = self._safe_float(getattr(m, "free_cash_flow_yield", None), float("nan"))
        if fcf_yield is not None and not pd.isna(fcf_yield):
            components["free_cash_flow_yield"] = fcf_yield
            scores.append(self._sigmoid(fcf_yield * 10, scale=1.0))

        if not scores:
            return SignalResult(
                signal_name=self.name,
                value=0.0,
                direction="neutral",
                confidence=0.5,
                components=components,
                metadata={"note": "no usable valuation metrics"},
            )

        composite = sum(scores) / len(scores)
        composite = self._normalize_to_signal(composite)

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
            metadata={"n_components": len(scores)},
        )
