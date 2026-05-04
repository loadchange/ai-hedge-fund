"""LangGraph technical analyst node.

Thin orchestration layer over ``src/signals/`` — the actual indicator
math lives in :mod:`src.signals` so it can be reused by the validation /
event-study / portfolio modules without dragging the LangGraph plumbing
along.

The agent's external contract (output dict shape, weighting, final
signal/confidence) is unchanged from the pre-refactor version so all
existing backtests reproduce bit-for-bit.
"""

from __future__ import annotations

import json

from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.signals import (
    MeanReversionSignal,
    MomentumSignal,
    StatArbSignal,
    TrendFollowingSignal,
    VolatilitySignal,
    weighted_signal_combination,
)
from src.signals.composite import signal_result_to_legacy
from src.signals.utils import normalize_pandas, safe_float
from src.tools.api import get_prices, prices_to_df
from src.utils.progress import progress


# Strategy → BaseSignal class mapping (instantiated per call).
_TECHNICAL_SIGNALS: dict[str, type] = {
    "trend": TrendFollowingSignal,
    "mean_reversion": MeanReversionSignal,
    "momentum": MomentumSignal,
    "volatility": VolatilitySignal,
    "stat_arb": StatArbSignal,
}

# Strategy weights — preserved from the historical implementation so the
# weighted combination output is unchanged.
_STRATEGY_WEIGHTS = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "momentum": 0.25,
    "volatility": 0.15,
    "stat_arb": 0.15,
}


def technical_analyst_agent(state: AgentState, agent_id: str = "technical_analyst_agent"):
    """LangGraph node: compute multi-strategy technical signals per ticker."""
    data = state["data"]
    start_date = data["start_date"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    technical_analysis: dict[str, dict] = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Analyzing price data")

        prices = get_prices(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
        )
        if not prices:
            progress.update_status(agent_id, ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)

        # Run each registered technical signal. Each returns a SignalResult
        # carrying both the canonical signed `value` and the legacy
        # direction/confidence/metrics breakdown.
        legacy_signals: dict[str, dict] = {}
        for strategy, signal_cls in _TECHNICAL_SIGNALS.items():
            progress.update_status(
                agent_id,
                ticker,
                {
                    "trend": "Calculating trend signals",
                    "mean_reversion": "Calculating mean reversion",
                    "momentum": "Calculating momentum",
                    "volatility": "Analyzing volatility",
                    "stat_arb": "Statistical analysis",
                }.get(strategy, f"Computing {strategy}"),
            )
            result = signal_cls().compute_from_prices(prices_df)
            legacy_signals[strategy] = signal_result_to_legacy(result)

        progress.update_status(agent_id, ticker, "Combining signals")
        combined_signal = weighted_signal_combination(legacy_signals, _STRATEGY_WEIGHTS)

        technical_analysis[ticker] = {
            "signal": combined_signal["signal"],
            "confidence": round(safe_float(combined_signal["confidence"], 0.5) * 100),
            "reasoning": {
                strategy: {
                    "signal": payload["signal"],
                    "confidence": round(safe_float(payload["confidence"], 0.5) * 100),
                    "metrics": normalize_pandas(payload["metrics"]),
                }
                for strategy, payload in legacy_signals.items()
            },
        }

        progress.update_status(
            agent_id,
            ticker,
            "Done",
            analysis=json.dumps(technical_analysis, indent=4, default=str),
        )

    message = HumanMessage(
        content=json.dumps(technical_analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(technical_analysis, "Technical Analyst")

    state["data"]["analyst_signals"][agent_id] = technical_analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }


# Re-exports for backward compatibility — anything that imported these
# helpers from ``src.agents.technicals`` keeps working.
__all__ = [
    "technical_analyst_agent",
    "weighted_signal_combination",
    "normalize_pandas",
    "safe_float",
]
