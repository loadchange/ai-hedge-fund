"""Pydantic models for strategy definitions."""

from __future__ import annotations

from pydantic import BaseModel
from typing import Literal


class AgentWeight(BaseModel):
    """Weight configuration for a single agent within a strategy."""

    agent_key: str
    weight: float = 1.0
    enabled: bool = True


class ScoringAdjustment(BaseModel):
    """Scoring parameters that modify signal processing and risk management."""

    signal_threshold_bullish: float = 0.2
    signal_threshold_bearish: float = -0.2
    confidence_cap: float | None = None
    risk_multiplier: float = 1.0


class StrategyDefinition(BaseModel):
    """Complete strategy definition loaded from YAML."""

    name: str
    display_name: str
    description: str
    category: Literal["technical", "fundamental", "hybrid", "sentiment"]
    agents: list[AgentWeight] = []
    signal_weights: dict[str, float] = {}
    scoring: ScoringAdjustment = ScoringAdjustment()
    instructions: str = ""
