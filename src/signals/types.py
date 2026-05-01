"""Pydantic models for the quantitative signal pipeline.

Ported from ``v2/models.py``. These are the data contracts that flow
between signals → portfolio optimizer → execution.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SignalResult(BaseModel):
    """Output of a single quantitative signal.

    ``value`` is the canonical signed score in ``[-1, +1]`` consumed by
    portfolio optimizers / validation runners. ``direction`` and
    ``confidence`` carry the legacy categorical / magnitude breakdown so
    the technical-analyst agent can reproduce its historical dict format
    exactly (regression-safe rewrite from ``src/agents/technicals.py``).
    """

    signal_name: str = Field(description="e.g. 'momentum', 'earnings_surprise'")
    value: float = Field(description="Signed score in [-1, +1] (bearish to bullish)")
    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    confidence: float = Field(default=0.0, description="Magnitude in [0, 1] — legacy semantic")
    z_score: float | None = None
    percentile: float | None = None  # 0-100
    components: dict[str, float] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class QuantSignals(BaseModel):
    """All signals for a single ticker on a single date."""

    ticker: str
    date: str
    signals: dict[str, SignalResult] = Field(default_factory=dict)
    composite_score: float | None = None


class PortfolioTarget(BaseModel):
    """Output of the portfolio optimizer — target weights."""

    weights: dict[str, float] = Field(
        default_factory=dict,
        description="ticker -> target weight (-1 to +1)",
    )
    expected_return: float | None = None
    expected_risk: float | None = None


class TradeOrder(BaseModel):
    """A single trade to execute."""

    ticker: str
    action: Literal["buy", "sell", "short", "cover"]
    shares: int = 0
    price: float = 0.0
    estimated_cost: float = 0.0
    reason: str = ""


class ExecutionResult(BaseModel):
    """Output of the execution layer."""

    orders: list[TradeOrder] = Field(default_factory=list)
    total_cost: float = 0.0
