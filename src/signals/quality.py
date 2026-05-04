"""Quality signal — high & stable ROE, healthy margins, conservative leverage.

Buffett-flavoured fundamental quality: profitability + balance-sheet
strength + margin durability across multiple historical periods.
"""

from __future__ import annotations

import statistics

from src.signals.base import BaseSignal
from src.signals.types import SignalResult


class QualitySignal(BaseSignal):
    """Composite quality score across ROE consistency, margins and leverage."""

    @property
    def name(self) -> str:
        return "quality"

    @property
    def kind(self) -> str:
        return "fundamental"

    def compute_from_prices(self, prices_df):  # noqa: ARG002
        raise NotImplementedError(
            "QualitySignal requires financial metrics — use compute(ticker, end_date)."
        )

    def compute(self, ticker: str, end_date: str, **kwargs) -> SignalResult:
        from src.tools.api import get_financial_metrics

        metrics = get_financial_metrics(
            ticker, end_date, period="ttm", limit=8, api_key=kwargs.get("api_key")
        )
        if not metrics:
            return self._empty_result()

        # Latest period — point-in-time view.
        latest = metrics[0]

        components: dict[str, float] = {}
        scores: list[float] = []

        # ROE consistency: how often is ROE > 15%?
        roes = [self._safe_float(getattr(m, "return_on_equity", None), float("nan")) for m in metrics]
        roes = [r for r in roes if r and not _is_nan(r)]
        if roes:
            components["roe_latest"] = roes[0]
            components["roe_mean"] = sum(roes) / len(roes)
            high_roe_pct = sum(1 for r in roes if r > 0.15) / len(roes)
            components["roe_high_pct"] = high_roe_pct
            # high_roe_pct in [0, 1]; map to [-1, +1] centred at 0.5.
            scores.append((high_roe_pct - 0.5) * 2)

        # Operating margin level + stability.
        op_margins = [
            self._safe_float(getattr(m, "operating_margin", None), float("nan")) for m in metrics
        ]
        op_margins = [m for m in op_margins if m is not None and not _is_nan(m)]
        if op_margins:
            avg_om = sum(op_margins) / len(op_margins)
            components["operating_margin_avg"] = avg_om
            scores.append(self._sigmoid(avg_om, scale=5.0))
            if len(op_margins) > 2:
                std_om = statistics.pstdev(op_margins)
                stability = 1.0 - min(std_om / max(abs(avg_om), 1e-6), 1.0)
                components["operating_margin_stability"] = stability
                scores.append((stability - 0.5) * 2)

        # Leverage — lower D/E is better.
        de = self._safe_float(getattr(latest, "debt_to_equity", None), float("nan"))
        if de is not None and not _is_nan(de) and de >= 0:
            components["debt_to_equity"] = de
            scores.append(self._sigmoid((1.0 - de) / 0.8, scale=1.0))

        # Current ratio (liquidity).
        cr = self._safe_float(getattr(latest, "current_ratio", None), float("nan"))
        if cr is not None and not _is_nan(cr) and cr > 0:
            components["current_ratio"] = cr
            scores.append(self._sigmoid((cr - 1.5) / 0.8, scale=1.0))

        if not scores:
            return SignalResult(
                signal_name=self.name,
                value=0.0,
                direction="neutral",
                confidence=0.5,
                components=components,
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
            metadata={"n_periods": len(metrics), "n_components": len(scores)},
        )


def _is_nan(x: float) -> bool:
    """Return True for NaN / inf safely (no math import needed at top)."""
    return x != x or x == float("inf") or x == float("-inf")
