"""Run hedge fund / backtester from parsed.json and emit a Markdown reply.

For the [ticker] mode we call run_hedge_fund() directly so we have the raw
structured result (decisions + analyst_signals) and can render proper
GitHub-flavoured Markdown tables — Rich's box-drawing tables look broken
inside a code block when CJK and ASCII mix.

For the [backtester] mode we still subprocess the script and post the
captured terminal output as a fenced code block (stripping ANSI). The
backtester's per-day output is too verbose to reformat usefully here.

Inputs (env / files):
  parsed.json           — output of parse_issue.py (must have ok=true)
  MODE                  — "ticker" | "backtester"

Outputs:
  comment.md            — body for `gh issue comment --body-file`
  result.json           — structured run result (ticker mode only, for audit)
  output.txt            — raw stdout/stderr (backtester mode only)
"""

from __future__ import annotations

import json
import os
import re
import subprocess
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
_ANSI_RE = re.compile(r"\x1b\[[0-9;?]*[a-zA-Z]")
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


def _run_backtester(parsed: dict) -> None:
    cmd = ["uv", "run", "python", "src/backtester.py"] + parsed["args"]
    print("[bot] Running:", " ".join(cmd), flush=True)
    captured = bytearray()
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as p:
        assert p.stdout is not None
        for chunk in iter(lambda: p.stdout.read(1024), b""):
            sys.stdout.buffer.write(chunk)
            sys.stdout.flush()
            captured.extend(chunk)
        ret = p.wait()

    raw = captured.decode("utf-8", errors="replace")
    Path("output.txt").write_text(raw, encoding="utf-8")

    cleaned = _ANSI_RE.sub("", raw)
    MAX = 50_000
    truncated = len(cleaned) > MAX
    shown = cleaned[:MAX].rstrip()

    md = ["## 📈 回测结果\n"]
    md.append(f"**调用参数**: `{parsed.get('summary', '')}`\n")
    md.append("```")
    md.append(shown)
    if truncated:
        md.append("\n... (输出已截断，完整结果见 Action 日志)")
    md.append("```")
    Path("comment.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    if ret != 0:
        sys.exit(ret)


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
        _run_backtester(parsed)
    else:
        _run_ticker(parsed, flags)


if __name__ == "__main__":
    main()
