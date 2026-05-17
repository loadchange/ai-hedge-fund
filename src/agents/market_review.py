"""LangGraph market review node.

Rule-based agent that analyzes major market indices to provide macro-level
context for trading decisions. No LLM calls — pure computation.

Computes daily/weekly change percentages, recent volatility (20-day ATR),
and trend direction for major indices in US/CN/HK markets.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import numpy as np
from langchain_core.messages import HumanMessage

from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_prices, prices_to_df
from src.utils.progress import progress

MARKET_INDICES: dict[str, list[str]] = {
    "us": ["SPY", "QQQ"],
    "cn": ["000300.SS"],
    "hk": ["^HSI"],
}


def _compute_index_metrics(prices_df) -> dict:
    """Compute change %, volatility, and trend from a prices DataFrame."""
    close = prices_df["close"]
    if len(close) < 2:
        return {"change_pct": 0.0, "weekly_change_pct": 0.0, "volatility": 0.0, "trend": "neutral"}

    daily_returns = close.pct_change().dropna()

    # Daily change
    change_pct = float((close.iloc[-1] / close.iloc[-2] - 1) * 100) if len(close) >= 2 else 0.0

    # Weekly change (5 trading days)
    weekly_idx = max(0, len(close) - 6)
    weekly_change_pct = float((close.iloc[-1] / close.iloc[weekly_idx] - 1) * 100)

    # Volatility (annualized, 20-day rolling std)
    if len(daily_returns) >= 5:
        volatility = float(daily_returns.tail(20).std() * np.sqrt(252) * 100)
    else:
        volatility = float(daily_returns.std() * np.sqrt(252) * 100) if len(daily_returns) > 0 else 0.0

    # Trend: based on 10-day and 20-day moving average crossover
    if len(close) >= 20:
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        if ma10.iloc[-1] > ma20.iloc[-1]:
            trend = "bullish"
        elif ma10.iloc[-1] < ma20.iloc[-1]:
            trend = "bearish"
        else:
            trend = "neutral"
    elif len(close) >= 5:
        # Short-term trend
        short_ma = close.tail(5).mean()
        prev_ma = close.iloc[:-5].mean() if len(close) > 5 else close.iloc[0]
        if short_ma > prev_ma:
            trend = "bullish"
        elif short_ma < prev_ma:
            trend = "bearish"
        else:
            trend = "neutral"
    else:
        trend = "neutral"

    return {
        "change_pct": round(change_pct, 2),
        "weekly_change_pct": round(weekly_change_pct, 2),
        "volatility": round(volatility, 2),
        "trend": trend,
        "latest_price": round(float(close.iloc[-1]), 2),
    }


def market_review_agent(state: AgentState, agent_id: str = "market_review_agent"):
    """LangGraph node: analyze major market indices for macro context."""
    data = state["data"]
    end_date = data["end_date"]
    start_date = data.get("start_date")

    # Default lookback: 3 months
    if not start_date:
        try:
            end_dt = datetime.strptime(end_date, "%Y-%m-%d")
            start_date = (end_dt - timedelta(days=90)).strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    market_overview: dict[str, dict] = {}
    all_metrics: dict[str, dict] = {}

    for market, indices in MARKET_INDICES.items():
        market_metrics: dict[str, dict] = {}
        for idx_ticker in indices:
            progress.update_status(agent_id, idx_ticker, "Fetching index data")

            prices = get_prices(
                ticker=idx_ticker,
                start_date=start_date,
                end_date=end_date,
            )
            if not prices:
                progress.update_status(agent_id, idx_ticker, "Failed: No data")
                continue

            prices_df = prices_to_df(prices)
            metrics = _compute_index_metrics(prices_df)
            market_metrics[idx_ticker] = metrics
            all_metrics[idx_ticker] = metrics

            progress.update_status(agent_id, idx_ticker, "Done")

        if market_metrics:
            market_overview[market] = {
                "indices": market_metrics,
            }

    # Aggregate signal: average across all indices
    if all_metrics:
        avg_change = np.mean([m["change_pct"] for m in all_metrics.values()])
        avg_weekly = np.mean([m["weekly_change_pct"] for m in all_metrics.values()])
        trends = [m["trend"] for m in all_metrics.values()]
        bullish_count = trends.count("bullish")
        bearish_count = trends.count("bearish")

        if bullish_count > bearish_count and avg_change > -0.5:
            signal = "bullish"
            confidence = min(95, 50 + bullish_count * 15 + max(0, int(avg_change * 5)))
        elif bearish_count > bullish_count and avg_change < 0.5:
            signal = "bearish"
            confidence = min(95, 50 + bearish_count * 15 + max(0, int(abs(avg_change) * 5)))
        else:
            signal = "neutral"
            confidence = 50

        reasoning_parts = []
        for ticker, m in all_metrics.items():
            direction = "+" if m["change_pct"] >= 0 else ""
            reasoning_parts.append(
                f"{ticker}: {direction}{m['change_pct']}% (weekly: {direction}{m['weekly_change_pct']}%), "
                f"vol={m['volatility']}%, trend={m['trend']}"
            )

        reasoning = (
            f"Market overview: {signal.upper()} (confidence: {confidence}%).\n"
            f"Avg daily change: {avg_change:+.2f}%, avg weekly: {avg_weekly:+.2f}%.\n"
            + "; ".join(reasoning_parts)
        )
    else:
        signal = "neutral"
        confidence = 0
        reasoning = "No index data available."
        market_overview = {}

    analysis = {
        "market_overview": market_overview,
        "signal": signal,
        "confidence": confidence,
        "reasoning": reasoning,
    }

    message = HumanMessage(
        content=json.dumps(analysis),
        name=agent_id,
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(analysis, "Market Review")

    state["data"]["analyst_signals"][agent_id] = analysis
    progress.update_status(agent_id, None, "Done")

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
