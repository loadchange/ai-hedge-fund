"""Run hedge fund / backtester from parsed.json and emit a Markdown reply.

Both modes call the underlying engine programmatically so we have the raw
structured result and can render proper GitHub-flavoured Markdown tables —
Rich's box-drawing tables look broken inside a code block when CJK and
ASCII mix. The full structured output is also written to disk so the
workflow can attach it as an Action artifact for download.

Inputs (env / files):
  parsed.json           — output of parse_issue.py (must have ok=true)
  MODE                  — "ticker" | "backtester"

Outputs:
  comment.md            — body for `gh issue comment --body-file`
  result.json           — structured ticker result (ticker mode)
  backtest_metrics.json — final performance metrics (backtester mode)
  portfolio_values.json — full equity curve (backtester mode)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv

load_dotenv()

# Repo root must be on sys.path so `from src.*` works.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.i18n import (  # noqa: E402
    get_text,
    set_lang,
    summarize_json_reasoning,
    translate_action,
    translate_agent_name,
    translate_signal,
)
from src.utils.analysts import ANALYST_CONFIG, ANALYST_ORDER  # noqa: E402

_FLAG_TRUE = {"--analysts-all", "--show-reasoning", "--ollama"}
_REASONING_CAP = 800


def _parse_flags(args: list[str]) -> dict:
    out: dict = {}
    i = 0
    while i < len(args):
        a = args[i]
        if a in _FLAG_TRUE:
            out[a[2:]] = True
            i += 1
        elif a.startswith("--") and i + 1 < len(args):
            out[a[2:]] = args[i + 1]
            i += 2
        else:
            i += 1
    return out


def _md_cell(s: str | None, cap: int | None = None) -> str:
    """Escape a string for a GitHub Markdown table cell."""
    if s is None:
        return ""
    text = str(s).strip()
    if cap and len(text) > cap:
        text = text[: cap - 1].rstrip() + "…"
    # Pipe and newline both break table rows.
    return text.replace("|", "\\|").replace("\r\n", "\n").replace("\n", "<br>")


def _resolve_provider(model_name: str) -> str:
    if not model_name:
        return "OpenAI"
    try:
        from src.llm.models import find_model_by_name

        model = find_model_by_name(model_name)
        return model.provider.value if model else "OpenAI"
    except Exception:
        return "OpenAI"


def _build_portfolio(tickers: list[str]) -> dict:
    return {
        "cash": 100_000.0,
        "margin_requirement": 0.0,
        "margin_used": 0.0,
        "positions": {
            t: {
                "long": 0,
                "short": 0,
                "long_cost_basis": 0.0,
                "short_cost_basis": 0.0,
                "short_margin_used": 0.0,
            }
            for t in tickers
        },
        "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
    }


# ── Ticker mode ──────────────────────────────────────────────────────────────


def _agent_display_name(agent_id: str) -> str:
    raw = agent_id.replace("_agent", "").replace("_", " ").title()
    return translate_agent_name(raw)


def _agent_order_index(agent_id: str) -> int:
    """Sort key matching ANALYST_ORDER, with risk_management at the end."""
    if agent_id == "risk_management_agent":
        return 999
    key = agent_id.replace("_agent", "")
    for idx, (_, k) in enumerate(ANALYST_ORDER):
        if k == key:
            return idx
    return 998


def _render_ticker_markdown(parsed_summary: str, result: dict, lang: str) -> str:
    set_lang(lang)
    decisions = result.get("decisions") or {}
    signals = result.get("analyst_signals") or {}

    lines: list[str] = []
    title = "✅ 分析完成" if lang == "zhCN" else "✅ Analysis complete"
    params_label = "调用参数" if lang == "zhCN" else "Parameters"
    lines.append(f"## {title}\n")
    lines.append(f"**{params_label}**: `{parsed_summary}`\n")

    agent_label = get_text("agent")
    signal_label = get_text("signal")
    conf_label = get_text("confidence")
    reasoning_label = get_text("reasoning")
    action_label = get_text("action")
    qty_label = get_text("quantity")
    decision_title = get_text("trading_decision")
    analysis_title = get_text("agent_analysis")

    for ticker, decision in decisions.items():
        lines.append(f"\n### {ticker}\n")

        # Trading decision
        action = (decision or {}).get("action", "").upper()
        action_disp = translate_action(action) if action else "-"
        qty = (decision or {}).get("quantity", 0)
        conf = (decision or {}).get("confidence", 0)
        reasoning = (decision or {}).get("reasoning", "")

        lines.append(f"**{decision_title}**\n")
        lines.append(f"| {action_label} | {qty_label} | {conf_label} |")
        lines.append("|---|---|---|")
        lines.append(f"| {_md_cell(action_disp)} | {qty} | {conf:.0f}% |")
        lines.append("")
        lines.append(f"**{reasoning_label}**: {_md_cell(reasoning, _REASONING_CAP)}")

        # Agent signals
        rows = []
        for agent_id, by_ticker in signals.items():
            if agent_id == "risk_management_agent":
                continue
            sig = (by_ticker or {}).get(ticker)
            if not sig:
                continue
            rows.append(
                (
                    _agent_order_index(agent_id),
                    _agent_display_name(agent_id),
                    translate_signal((sig.get("signal") or "").upper()),
                    sig.get("confidence", 0),
                    summarize_json_reasoning(sig.get("reasoning", "")),
                )
            )
        rows.sort()

        if rows:
            lines.append(f"\n**{analysis_title}**\n")
            lines.append(
                f"| {agent_label} | {signal_label} | {conf_label} | {reasoning_label} |"
            )
            lines.append("|---|---|---:|---|")
            for _, name, sig, c, reason in rows:
                conf_str = f"{c:.0f}%" if isinstance(c, (int, float)) else str(c)
                lines.append(
                    f"| {_md_cell(name)} | {_md_cell(sig)} | {conf_str} | {_md_cell(reason, _REASONING_CAP)} |"
                )

    # Portfolio summary across all tickers
    if len(decisions) > 1:
        lines.append(f"\n**{get_text('portfolio_summary')}**\n")
        lines.append(
            f"| {get_text('ticker')} | {action_label} | {qty_label} | {conf_label} | {get_text('bullish')} | {get_text('bearish')} | {get_text('neutral')} |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        for ticker, decision in decisions.items():
            counts = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0}
            for agent_id, by_ticker in signals.items():
                if agent_id == "risk_management_agent":
                    continue
                sig = (by_ticker or {}).get(ticker)
                if not sig:
                    continue
                key = (sig.get("signal") or "").upper()
                if key in counts:
                    counts[key] += 1
            action = (decision or {}).get("action", "").upper()
            lines.append(
                f"| {ticker} | {_md_cell(translate_action(action) if action else '-')} | "
                f"{(decision or {}).get('quantity', 0)} | "
                f"{(decision or {}).get('confidence', 0):.0f}% | "
                f"{counts['BULLISH']} | {counts['BEARISH']} | {counts['NEUTRAL']} |"
            )

    return "\n".join(lines) + "\n"


def _run_ticker(parsed: dict, flags: dict) -> None:
    lang = flags.get("lang", "zhCN")
    set_lang(lang)

    tickers = flags["tickers"].split(",")
    end_date = flags.get("end-date") or datetime.now().strftime("%Y-%m-%d")
    start_date = flags.get("start-date") or (
        datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=3)
    ).strftime("%Y-%m-%d")
    model_name = flags.get("model", "")

    if flags.get("analysts-all"):
        selected_analysts = list(ANALYST_CONFIG.keys())
    elif flags.get("analysts"):
        selected_analysts = flags["analysts"].split(",")
    else:
        selected_analysts = list(ANALYST_CONFIG.keys())

    portfolio = _build_portfolio(tickers)

    # Imported here to avoid pulling in colorama / questionary side-effects on import.
    from src.main import run_hedge_fund
    from src.utils.display import print_trading_output

    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio=portfolio,
        show_reasoning=flags.get("show-reasoning", False),
        selected_analysts=selected_analysts,
        model_name=model_name,
        model_provider=_resolve_provider(model_name),
    )

    # Save raw result for audit / future debugging.
    Path("result.json").write_text(
        json.dumps(result, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )

    # Mirror Rich output to CI logs.
    print_trading_output(result)

    # Generate the Markdown comment.
    md = _render_ticker_markdown(parsed.get("summary", ""), result, lang)
    Path("comment.md").write_text(md, encoding="utf-8")


# ── Backtester mode ──────────────────────────────────────────────────────────


def _serialise_point(p: dict) -> dict:
    """Convert a PortfolioValuePoint to a JSON-safe dict (Date → ISO string)."""
    out: dict = {}
    for k, v in p.items():
        if hasattr(v, "isoformat"):
            out[k] = v.isoformat()
        else:
            out[k] = v
    return out


def _render_backtester_markdown(parsed_summary: str, metrics: dict, points: list, lang: str) -> str:
    set_lang(lang)
    zh = lang == "zhCN"

    lines: list[str] = []
    title = "📈 回测完成" if zh else "📈 Backtest complete"
    params_label = "调用参数" if zh else "Parameters"
    lines.append(f"## {title}\n")
    lines.append(f"**{params_label}**: `{parsed_summary}`\n")

    # Performance summary — always rendered, never truncated.
    lines.append(f"### {'绩效摘要' if zh else 'Performance Summary'}\n")
    lines.append(f"| {'指标' if zh else 'Metric'} | {'值' if zh else 'Value'} |")
    lines.append("|---|---:|")

    if points and len(points) >= 2:
        first_value = points[0].get("Portfolio Value") or 0.0
        last_value = points[-1].get("Portfolio Value") or 0.0
        total_return = ((last_value - first_value) / first_value * 100) if first_value else 0.0
        lines.append(f"| {get_text('initial_value')} | ${first_value:,.2f} |")
        lines.append(f"| {get_text('final_value')} | ${last_value:,.2f} |")
        lines.append(f"| {get_text('total_return')} | {total_return:+.2f}% |")
    else:
        lines.append(f"| {('数据点不足' if zh else 'Not enough data points')} | – |")

    if isinstance(metrics, dict):
        for key, formatter in (
            ("sharpe_ratio", lambda v: f"{v:.2f}"),
            ("sortino_ratio", lambda v: f"{v:.2f}"),
            ("max_drawdown", lambda v: f"{v:+.2f}%"),
        ):
            v = metrics.get(key)
            if v is None:
                continue
            label = get_text(key)
            lines.append(f"| {label} | {formatter(v)} |")

        total_costs = metrics.get("total_transaction_costs")
        if isinstance(total_costs, (int, float)) and total_costs > 0:
            label = "总交易成本" if zh else "Total Transaction Costs"
            lines.append(f"| {label} | ${total_costs:,.2f} |")

    lines.append("")

    # Equity curve sample — last N trading days, full series in artifact.
    if points:
        N = 20
        sample = points[-N:] if len(points) > N else points
        header = "净值走势" if zh else "Equity curve"
        recent = "最近" if zh else "last"
        days = "个交易日" if zh else "trading days"
        lines.append(f"### {header}（{recent} {len(sample)} {days}）\n" if zh else f"### {header} (last {len(sample)} trading days)\n")
        lines.append(
            f"| {'日期' if zh else 'Date'} | {'组合净值' if zh else 'Portfolio Value'} | "
            f"{'多头敞口' if zh else 'Long'} | {'空头敞口' if zh else 'Short'} | "
            f"{'净敞口' if zh else 'Net'} |"
        )
        lines.append("|---|---:|---:|---:|---:|")
        for p in sample:
            d = p.get("Date")
            d_str = d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            pv = p.get("Portfolio Value", 0) or 0
            le = p.get("Long Exposure", 0) or 0
            se = p.get("Short Exposure", 0) or 0
            ne = p.get("Net Exposure", 0) or 0
            lines.append(f"| {d_str} | ${pv:,.2f} | ${le:,.2f} | ${se:,.2f} | ${ne:,.2f} |")

        if len(points) > N:
            note = (
                f"\n_（共 {len(points)} 个交易日；完整序列见 Action 工件 `portfolio_values.json`）_"
                if zh else
                f"\n_(Total {len(points)} trading days; the full series is in workflow artifact `portfolio_values.json`.)_"
            )
            lines.append(note)

    # Footer pointing to artifacts.
    artifact_note = (
        "\n---\n📎 **完整数据**：本次运行的 `portfolio_values.json`、`backtest_metrics.json`、`parsed.json` 已作为 Action 工件上传，可在本次 run 页面顶部 *Artifacts* 区域下载。"
        if zh else
        "\n---\n📎 **Full data**: `portfolio_values.json`, `backtest_metrics.json`, and `parsed.json` are uploaded as workflow artifacts — download them from the *Artifacts* section at the top of this run page."
    )
    lines.append(artifact_note)

    return "\n".join(lines) + "\n"


def _run_backtester(parsed: dict, flags: dict) -> None:
    lang = flags.get("lang", "zhCN")
    set_lang(lang)

    tickers = flags["tickers"].split(",")
    end_date = flags.get("end-date") or datetime.now().strftime("%Y-%m-%d")
    start_date = flags.get("start-date") or (
        datetime.strptime(end_date, "%Y-%m-%d") - relativedelta(months=1)
    ).strftime("%Y-%m-%d")
    model_name = flags.get("model", "")
    initial_cash = float(flags.get("initial-cash") or 100_000.0)
    margin_req = float(flags.get("margin-requirement") or 0.0)
    cost_bps = float(flags.get("cost-bps") or 10.0)
    cost_model_name = flags.get("cost-model") or "fixed"

    if flags.get("analysts-all"):
        selected_analysts = list(ANALYST_CONFIG.keys())
    elif flags.get("analysts"):
        selected_analysts = flags["analysts"].split(",")
    else:
        selected_analysts = list(ANALYST_CONFIG.keys())

    from src.backtester import run_backtest
    from src.backtesting.costs import build_cost_model
    from src.backtesting.engine import BacktestEngine
    from src.main import run_hedge_fund

    cost_model = build_cost_model(cost_model_name, bps=cost_bps)

    engine = BacktestEngine(
        agent=run_hedge_fund,
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        initial_capital=initial_cash,
        model_name=model_name,
        model_provider=_resolve_provider(model_name),
        selected_analysts=selected_analysts,
        initial_margin_requirement=margin_req,
        cost_model=cost_model,
    )

    metrics = run_backtest(engine) or {}
    points = list(engine.get_portfolio_values())
    metrics["total_transaction_costs"] = engine.get_total_transaction_costs()
    metrics["costs_by_action"] = engine.get_costs_by_action()

    # Persist structured outputs for the workflow's artifact step.
    Path("backtest_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )
    Path("portfolio_values.json").write_text(
        json.dumps([_serialise_point(p) for p in points], ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )

    md = _render_backtester_markdown(parsed.get("summary", ""), metrics, points, lang)
    Path("comment.md").write_text(md, encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    mode = os.environ.get("MODE", "ticker")
    parsed = json.loads(Path("parsed.json").read_text(encoding="utf-8"))
    if not parsed.get("ok"):
        # Should not reach here — workflow checks ok before calling us.
        sys.stderr.write("[bot] parsed.json has ok=false; aborting.\n")
        sys.exit(1)

    flags = _parse_flags(parsed["args"])

    if mode == "backtester":
        _run_backtester(parsed, flags)
    else:
        _run_ticker(parsed, flags)


if __name__ == "__main__":
    main()
