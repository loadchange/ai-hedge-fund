"""Earnings-surprise signal — react to BEAT / MISS / MEET on revenue & EPS.

Reads the latest :class:`Earnings` payload from
``FinancialDatasetsSource.get_earnings`` and produces a signed score
biased to recent quarters. ``BEAT`` on both revenue and EPS scores +1;
``MISS`` on both scores −1; mixed or ``MEET`` produces a smaller score.

Falls back to neutral when the API does not return earnings data (e.g.
A-shares, where this signal is not yet supported).
"""

from __future__ import annotations

from src.signals.base import BaseSignal
from src.signals.types import SignalResult


_SURPRISE_SCORE = {"BEAT": 1.0, "MEET": 0.0, "MISS": -1.0}


class EarningsSurpriseSignal(BaseSignal):
    """SUE-style score from the latest quarterly earnings surprise."""

    @property
    def name(self) -> str:
        return "earnings_surprise"

    def compute_from_prices(self, prices_df):  # noqa: ARG002
        raise NotImplementedError(
            "EarningsSurpriseSignal requires earnings data — use compute(ticker, end_date)."
        )

    def compute(self, ticker: str, end_date: str, **kwargs) -> SignalResult:
        from src.data.sources.financialsets import FinancialDatasetsSource

        api_key = kwargs.get("api_key")
        with FinancialDatasetsSource(api_key=api_key) as fd:
            earnings = fd.get_earnings(ticker)

        if earnings is None or earnings.quarterly is None:
            return SignalResult(
                signal_name=self.name,
                value=0.0,
                direction="neutral",
                confidence=0.5,
                components={},
                metadata={"note": "no earnings data available"},
            )

        q = earnings.quarterly
        components: dict[str, float] = {}
        scores: list[float] = []

        rev_surprise = (q.revenue_surprise or "").upper()
        if rev_surprise in _SURPRISE_SCORE:
            components["revenue_surprise_score"] = _SURPRISE_SCORE[rev_surprise]
            scores.append(_SURPRISE_SCORE[rev_surprise])

        eps_surprise = (q.eps_surprise or "").upper()
        if eps_surprise in _SURPRISE_SCORE:
            components["eps_surprise_score"] = _SURPRISE_SCORE[eps_surprise]
            scores.append(_SURPRISE_SCORE[eps_surprise])

        # Magnitude — surprise size in pct.
        if q.revenue is not None and q.estimated_revenue and q.estimated_revenue != 0:
            mag = (q.revenue - q.estimated_revenue) / abs(q.estimated_revenue)
            components["revenue_surprise_magnitude"] = self._safe_float(mag)
            scores.append(self._sigmoid(self._safe_float(mag), scale=20.0))

        if q.earnings_per_share is not None and q.estimated_earnings_per_share and q.estimated_earnings_per_share != 0:
            mag = (q.earnings_per_share - q.estimated_earnings_per_share) / abs(
                q.estimated_earnings_per_share
            )
            components["eps_surprise_magnitude"] = self._safe_float(mag)
            scores.append(self._sigmoid(self._safe_float(mag), scale=10.0))

        if not scores:
            return SignalResult(
                signal_name=self.name,
                value=0.0,
                direction="neutral",
                confidence=0.5,
                components=components,
                metadata={"note": "no surprise data on latest quarter"},
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
            metadata={
                "report_period": earnings.report_period,
                "fiscal_period": earnings.fiscal_period,
            },
        )
