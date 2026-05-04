"""LangGraph risk-management node.

Volatility- and correlation-adjusted position limits per ticker. Math
lives in :mod:`src.risk`; this module is just the orchestration glue
(price fetching, portfolio NLV, status updates).

External contract is unchanged from the pre-refactor version so existing
backtests and the portfolio_manager downstream see the same payload.
"""

from __future__ import annotations

import json
import math

import pandas as pd
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.risk.correlation import (
    build_correlation_matrix,
    calculate_correlation_multiplier,
    correlation_summary,
)
from src.risk.volatility import (
    calculate_volatility_adjusted_limit,
    calculate_volatility_metrics,
)
from src.tools.api import get_prices, prices_to_df
from src.utils.progress import progress


_DEFAULT_VOL_FALLBACK = {
    "daily_volatility": 0.05,
    "annualized_volatility": 0.05 * math.sqrt(252),
    "volatility_percentile": 100.0,
    "data_points": 0,
}


def risk_management_agent(state: AgentState, agent_id: str = "risk_management_agent"):
    """Compute volatility- and correlation-adjusted position limits per ticker."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    risk_analysis: dict[str, dict] = {}
    current_prices: dict[str, float] = {}
    volatility_data: dict[str, dict] = {}
    returns_by_ticker: dict[str, pd.Series] = {}

    all_tickers = set(tickers) | set(portfolio.get("positions", {}).keys())

    # ------------------------------------------------------------------
    # Phase 1: pull prices, compute volatility + return series.
    # ------------------------------------------------------------------
    for ticker in all_tickers:
        progress.update_status(agent_id, ticker, "Fetching price data and calculating volatility")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status(agent_id, ticker, "Warning: No price data found")
            volatility_data[ticker] = dict(_DEFAULT_VOL_FALLBACK)
            continue

        prices_df = prices_to_df(prices)
        if prices_df.empty or len(prices_df) <= 1:
            progress.update_status(agent_id, ticker, "Warning: Insufficient price data")
            current_prices[ticker] = 0
            fallback = dict(_DEFAULT_VOL_FALLBACK)
            fallback["data_points"] = len(prices_df) if not prices_df.empty else 0
            volatility_data[ticker] = fallback
            continue

        current_price = float(prices_df["close"].iloc[-1])
        current_prices[ticker] = current_price

        vol_metrics = calculate_volatility_metrics(prices_df)
        volatility_data[ticker] = vol_metrics

        daily_returns = prices_df["close"].pct_change().dropna()
        if len(daily_returns) > 0:
            returns_by_ticker[ticker] = daily_returns

        progress.update_status(
            agent_id,
            ticker,
            f"Price: {current_price:.2f}, Ann. Vol: {vol_metrics['annualized_volatility']:.1%}",
        )

    # ------------------------------------------------------------------
    # Phase 2: cross-ticker correlation matrix.
    # ------------------------------------------------------------------
    correlation_matrix = build_correlation_matrix(returns_by_ticker)

    # Tickers currently held (long − short ≠ 0).
    active_positions = {
        t for t, pos in portfolio.get("positions", {}).items()
        if abs(pos.get("long", 0) - pos.get("short", 0)) > 0
    }

    # ------------------------------------------------------------------
    # Phase 3: total NLV.
    # ------------------------------------------------------------------
    total_portfolio_value = portfolio.get("cash", 0.0)
    for ticker, position in portfolio.get("positions", {}).items():
        if ticker in current_prices:
            total_portfolio_value += position.get("long", 0) * current_prices[ticker]
            total_portfolio_value -= position.get("short", 0) * current_prices[ticker]

    progress.update_status(agent_id, None, f"Total portfolio value: {total_portfolio_value:.2f}")

    # ------------------------------------------------------------------
    # Phase 4: per-ticker risk limit.
    # ------------------------------------------------------------------
    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Calculating volatility- and correlation-adjusted limits")

        if ticker not in current_prices or current_prices[ticker] <= 0:
            progress.update_status(agent_id, ticker, "Failed: No valid price data")
            risk_analysis[ticker] = {
                "remaining_position_limit": 0.0,
                "current_price": 0.0,
                "reasoning": {"error": "Missing price data for risk calculation"},
            }
            continue

        current_price = current_prices[ticker]
        vol_data = volatility_data.get(ticker, {})

        position = portfolio.get("positions", {}).get(ticker, {})
        long_value = position.get("long", 0) * current_price
        short_value = position.get("short", 0) * current_price
        current_position_value = abs(long_value - short_value)

        vol_adjusted_limit_pct = calculate_volatility_adjusted_limit(
            vol_data.get("annualized_volatility", 0.25)
        )

        # Correlation-aware multiplier — compare with currently active
        # positions, falling back to all other tickers when nothing's
        # active yet.
        compare_with = (
            list(active_positions) if active_positions else None
        )
        corr_metrics = correlation_summary(correlation_matrix, ticker, compare_with=compare_with)
        avg_corr = corr_metrics["avg_correlation_with_active"]
        corr_multiplier = (
            calculate_correlation_multiplier(avg_corr) if avg_corr is not None else 1.0
        )

        combined_limit_pct = vol_adjusted_limit_pct * corr_multiplier
        position_limit = total_portfolio_value * combined_limit_pct

        remaining_position_limit = position_limit - current_position_value
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(current_price),
            "volatility_metrics": {
                "daily_volatility": float(vol_data.get("daily_volatility", 0.05)),
                "annualized_volatility": float(vol_data.get("annualized_volatility", 0.25)),
                "volatility_percentile": float(vol_data.get("volatility_percentile", 100)),
                "data_points": int(vol_data.get("data_points", 0)),
            },
            "correlation_metrics": corr_metrics,
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position_value": float(current_position_value),
                "base_position_limit_pct": float(vol_adjusted_limit_pct),
                "correlation_multiplier": float(corr_multiplier),
                "combined_position_limit_pct": float(combined_limit_pct),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
                "risk_adjustment": (
                    f"Volatility x Correlation adjusted: {combined_limit_pct:.1%} "
                    f"(base {vol_adjusted_limit_pct:.1%})"
                ),
            },
        }

        progress.update_status(
            agent_id,
            ticker,
            f"Adj. limit: {combined_limit_pct:.1%}, Available: ${max_position_size:.0f}",
        )

    progress.update_status(agent_id, None, "Done")

    message = HumanMessage(content=json.dumps(risk_analysis), name=agent_id)
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Volatility-Adjusted Risk Management Agent")

    state["data"]["analyst_signals"][agent_id] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
