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
``BaseSignal`` subclasses that pull data directly via ``DataClient``.
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

__all__ = [
    "BaseSignal",
    "SignalResult",
    "QuantSignals",
    "PortfolioTarget",
    "TradeOrder",
    "ExecutionResult",
    "SIGNAL_REGISTRY",
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
