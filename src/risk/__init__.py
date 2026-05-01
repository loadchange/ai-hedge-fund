"""Risk management primitives — volatility, correlation, drawdown, stress, sizing.

Used by:
* :mod:`src.agents.risk_manager` (LangGraph node) — keeps its public
  contract unchanged but delegates the math here.
* The future portfolio optimizer (:mod:`src.portfolio`) — risk-parity
  weighting and tracking-error constraints share this code.
* The validation pipeline (:mod:`src.validation`) — out-of-sample
  drawdown / vol stats per CPCV split.
"""

from src.risk.correlation import (
    build_correlation_matrix,
    calculate_correlation_multiplier,
    correlation_summary,
)
from src.risk.drawdown import (
    DrawdownStats,
    drawdown_series,
    drawdown_stats,
)
from src.risk.sizing import kelly_fraction, vol_targeted_size
from src.risk.stress import StressScenario, apply_scenario, default_scenarios
from src.risk.volatility import (
    calculate_volatility_adjusted_limit,
    calculate_volatility_metrics,
)

__all__ = [
    "build_correlation_matrix",
    "calculate_correlation_multiplier",
    "calculate_volatility_adjusted_limit",
    "calculate_volatility_metrics",
    "correlation_summary",
    "DrawdownStats",
    "drawdown_series",
    "drawdown_stats",
    "kelly_fraction",
    "vol_targeted_size",
    "StressScenario",
    "apply_scenario",
    "default_scenarios",
]
