"""Quantitative signal pipeline.

Each signal subclass of :class:`BaseSignal` produces a numeric score in
``[-1, +1]`` (bearish to bullish) with a structured :class:`SignalResult`
payload (z-score, percentile rank, components for diagnostics).

The classic technical signals (trend / mean_reversion / momentum /
volatility / stat_arb) live here as ``BaseSignal`` subclasses; the
``src/agents/technicals.py`` LangGraph node delegates to them so existing
backtest behaviour is preserved while quant modules (validation,
event_study, portfolio optimizer) get a clean per-signal API.

Fundamental signals (value / quality / earnings_surprise) are
``BaseSignal`` subclasses that pull data via the free providers wired
through ``src/tools/api.py`` and ``src/data/sources/``.
"""

from __future__ import annotations

from src.signals.base import BaseSignal
from src.signals.types import (
    ExecutionResult,
    PortfolioTarget,
    QuantSignals,
    SignalResult,
    TradeOrder,
)
from src.signals.composite import weighted_signal_combination

# Concrete signal classes — re-exported for convenience.
from src.signals.trend import TrendFollowingSignal
from src.signals.mean_reversion import MeanReversionSignal
from src.signals.momentum import MomentumSignal
from src.signals.volatility import VolatilitySignal
from src.signals.stat_arb import StatArbSignal
from src.signals.value import ValueSignal
from src.signals.quality import QualitySignal
from src.signals.earnings_surprise import EarningsSurpriseSignal

# Registry maps name → class. Useful for CLI / validation runners that
# look up signals by string identifier.
SIGNAL_REGISTRY: dict[str, type[BaseSignal]] = {
    "trend": TrendFollowingSignal,
    "mean_reversion": MeanReversionSignal,
    "momentum": MomentumSignal,
    "volatility": VolatilitySignal,
    "stat_arb": StatArbSignal,
    "value": ValueSignal,
    "quality": QualitySignal,
    "earnings_surprise": EarningsSurpriseSignal,
}


def signals_by_kind(kind: str) -> list[str]:
    """Return signal keys whose class declares ``kind``.

    Single source of truth so the issue templates, parse_issue.py, and
    runner.py never go out of sync about which signals support which
    evaluation modes.
    """
    return [
        key for key, cls in SIGNAL_REGISTRY.items() if cls().kind == kind
    ]


TECHNICAL_SIGNALS: tuple[str, ...] = tuple(signals_by_kind("technical"))
FUNDAMENTAL_SIGNALS: tuple[str, ...] = tuple(signals_by_kind("fundamental"))


__all__ = [
    "BaseSignal",
    "SignalResult",
    "QuantSignals",
    "PortfolioTarget",
    "TradeOrder",
    "ExecutionResult",
    "SIGNAL_REGISTRY",
    "TECHNICAL_SIGNALS",
    "FUNDAMENTAL_SIGNALS",
    "signals_by_kind",
    "weighted_signal_combination",
    "TrendFollowingSignal",
    "MeanReversionSignal",
    "MomentumSignal",
    "VolatilitySignal",
    "StatArbSignal",
    "ValueSignal",
    "QualitySignal",
    "EarningsSurpriseSignal",
]
