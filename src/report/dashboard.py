"""Dashboard report generator.

Produces structured decision dashboards in Markdown, Rich, or JSON format
from the run_hedge_fund result dictionary.
"""

from __future__ import annotations

import json
from datetime import datetime

from src.i18n import get_text, set_lang


class DashboardReport:
    """Generate a structured decision dashboard from analysis results."""

    def __init__(self, result: dict, lang: str = "en"):
        self.result = result
        self.lang = lang
        set_lang(lang)
        self.zh = lang == "zhCN"

    def render_markdown(self) -> str:
        """Render as GitHub-flavored Markdown."""
        lines: list[str] = []

        # Header
        now = datetime.now()
        title = "决策仪表盘" if self.zh else "Decision Dashboard"
        lines.append(f"# {title}\n")
        lines.append(f"**{now.strftime('%Y-%m-%d %H:%M')}**\n")

        # Market context
        market_md = self._build_market_context_md()
        if market_md:
            lines.append(market_md)

        # Signal matrix
        lines.append(self._build_signal_matrix_md())

        # Decisions
        lines.append(self._build_decisions_md())

        return "\n".join(lines) + "\n"

    def render_rich(self) -> str:
        """Render using Rich library (terminal output)."""
        return self.render_markdown()

    def render_json(self) -> dict:
        """Return structured JSON dashboard data."""
        decisions = self.result.get("decisions", {})
        signals = self.result.get("analyst_signals", {})

        dashboard = {
            "generated_at": datetime.now().isoformat(),
            "decisions": {},
            "signals": {},
        }

        for ticker, decision in decisions.items():
            dashboard["decisions"][ticker] = {
                "action": (decision or {}).get("action", ""),
                "quantity": (decision or {}).get("quantity", 0),
                "confidence": (decision or {}).get("confidence", 0),
                "reasoning": (decision or {}).get("reasoning", ""),
            }

        for agent_id, by_ticker in signals.items():
            if agent_id == "risk_management_agent":
                continue
            dashboard["signals"][agent_id] = by_ticker

        return dashboard

    def _build_market_context_md(self) -> str:
        """Build market context section from market_review_agent signals."""
        signals = self.result.get("analyst_signals", {})
        market_review = signals.get("market_review_agent", {})
        overview = market_review.get("market_overview", {})
        if not overview:
            return ""

        header = "市场概况" if self.zh else "Market Context"
        lines = [f"## {header}\n"]

        for market, data in overview.items():
            for idx_ticker, metrics in data.get("indices", {}).items():
                change = metrics.get("change_pct", 0)
                vol = metrics.get("volatility", 0)
                trend = metrics.get("trend", "neutral")
                direction = "+" if change >= 0 else ""
                lines.append(f"- **{idx_ticker}**: {direction}{change}% | Vol: {vol}% | {trend.upper()}")

        lines.append("")
        return "\n".join(lines)

    def _build_signal_matrix_md(self) -> str:
        """Build agent x ticker signal matrix."""
        signals = self.result.get("analyst_signals", {})
        decisions = self.result.get("decisions", {})
        tickers = list(decisions.keys())

        if not tickers:
            return ""

        header = "信号矩阵" if self.zh else "Signal Matrix"
        lines = [f"## {header}\n"]

        # Collect agent rows
        agents: list[tuple[str, str]] = []
        for agent_id, by_ticker in signals.items():
            if agent_id == "risk_management_agent":
                continue
            display = agent_id.replace("_agent", "").replace("_", " ").title()
            agents.append((agent_id, display))

        # Table header
        cols = [get_text("agent") if self.zh else "Agent"] + tickers
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        # Table rows
        for agent_id, display in agents:
            cells = [f"**{display}**"]
            by_ticker = signals.get(agent_id, {})
            for ticker in tickers:
                sig = (by_ticker or {}).get(ticker, {})
                if not sig:
                    cells.append("-")
                    continue
                signal = (sig.get("signal") or "").upper()
                conf = sig.get("confidence", 0)
                if isinstance(conf, (int, float)) and conf > 0:
                    cells.append(f"{signal} {conf:.0f}%")
                else:
                    cells.append(signal)
            lines.append("| " + " | ".join(cells) + " |")

        lines.append("")
        return "\n".join(lines)

    def _build_decisions_md(self) -> str:
        """Build final trading decisions section."""
        decisions = self.result.get("decisions", {})
        if not decisions:
            return ""

        header = "交易决策" if self.zh else "Trading Decisions"
        lines = [f"## {header}\n"]

        for ticker, decision in decisions.items():
            action = (decision or {}).get("action", "").upper()
            qty = (decision or {}).get("quantity", 0)
            conf = (decision or {}).get("confidence", 0)
            reasoning = (decision or {}).get("reasoning", "")

            action_emoji = {"BUY": "🟢", "SELL": "🔴", "SHORT": "🔴", "HOLD": "🟡"}.get(action, "⚪")
            lines.append(f"### {action_emoji} {ticker}: {action}")
            if qty:
                lines.append(f"- **Quantity**: {qty}")
            lines.append(f"- **Confidence**: {conf:.0f}%")
            if reasoning:
                lines.append(f"- **Reasoning**: {reasoning[:500]}")
            lines.append("")

        return "\n".join(lines)
