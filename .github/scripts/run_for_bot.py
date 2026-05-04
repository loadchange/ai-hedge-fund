"""Run a bot mode from parsed.json and emit a Markdown reply.

Modes call the underlying engine programmatically so we have the raw
structured result and can render proper GitHub-flavoured Markdown tables —
Rich's box-drawing tables look broken inside a code block when CJK and
ASCII mix. The full structured output is also written to disk so the
workflow can attach it as an Action artifact for download.

Inputs (env / files):
  parsed.json           — output of parse_issue.py (must have ok=true)
  MODE                  — "ticker" | "backtester" | "validate" | "event_study"

Outputs:
  comment.md                — body for `gh issue comment --body-file`
  result.json               — structured ticker result (ticker mode)
  backtest_metrics.json     — final performance metrics (backtester mode)
  portfolio_values.json     — full equity curve (backtester mode)
  validation_results.json   — IS/OOS Sharpe + PBO per signal-ticker (validate)
  event_study_result.json   — α/β + AR/CAR + significance (event_study)
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


# ── Validate mode (CPCV + PBO) ───────────────────────────────────────────────


def _build_validate_recovery_tip(error_rows: list[dict], zh: bool) -> str | None:
    """Suggest concrete fixes when every signal hits insufficient_data.

    Looks at the smallest available bar count and the requested rolling
    window across the error rows, then computes the largest rolling
    window that *would* have fit. When the shortfall lines up with the
    yfinance ~1-year cap that GitHub Actions runners hit, recommend
    switching to A-share tickers (akshare/baostock pull multi-year
    history without that cap).
    """
    insuff = [r for r in error_rows if r.get("error_kind") == "insufficient_data"]
    if not insuff:
        return None

    min_bars = min(int(r.get("n_bars_available", 0) or 0) for r in insuff)
    requested_rw = max(int(r.get("rolling_window", 60) or 60) for r in insuff)
    suggested_rw = max(20, min_bars // 2)
    looks_capped = 240 <= min_bars <= 260  # yfinance unauth cap ≈ 1 year ≈ 252 bars

    if zh:
        bullets = [
            f"把 `rolling-window` 改成 ≤ **{suggested_rw}**（当前 {requested_rw}，需要 ≥ {requested_rw * 2} bars，但只拿到 {min_bars}）",
            "把 `开始日期` 推前 1–2 年，给 rolling window 留够历史",
        ]
        if looks_capped:
            bullets.append(
                "数据看起来被掐在 ~1 年（≈252 bars），这是 **yfinance 在 GitHub Actions runner IP 上的硬限制**。"
                "改用 A 股代码（如 `600519.SS` / `000001.SZ`）走 baostock/akshare 路径即可拿到多年历史。"
            )
        return "**建议**：\n" + "\n".join(f"- {b}" for b in bullets)

    bullets = [
        f"Drop `rolling-window` to ≤ **{suggested_rw}** (you used {requested_rw}, which needs ≥ {requested_rw * 2} bars but only {min_bars} were returned)",
        "Push `start date` 1–2 years earlier so the rolling window has enough history",
    ]
    if looks_capped:
        bullets.append(
            "The fetch is capped at ~1 year (≈252 bars) — this is the **yfinance limit when called from a GitHub Actions runner IP**. "
            "Switch to A-share tickers (e.g. `600519.SS`, `000001.SZ`) to use baostock/akshare and get multi-year history."
        )
    return "**Suggestions**:\n" + "\n".join(f"- {b}" for b in bullets)


def _render_validate_markdown(parsed_summary: str, results: list[dict], lang: str) -> str:
    set_lang(lang)
    zh = lang == "zhCN"

    lines: list[str] = []
    title = "🧪 信号验证完成" if zh else "🧪 Signal validation complete"
    params_label = "调用参数" if zh else "Parameters"
    lines.append(f"## {title}\n")
    lines.append(f"**{params_label}**: `{parsed_summary}`\n")

    error_rows = [r for r in results if "error" in r]
    success_rows = [r for r in results if "error" not in r]
    all_failed = bool(results) and not success_rows

    def _fmt(v: float | None) -> str:
        if v is None:
            return "–"
        try:
            f = float(v)
        except (TypeError, ValueError):
            return "–"
        if f != f:  # NaN
            return "–"
        return f"{f:+.2f}"

    if all_failed:
        # Skip the empty results table — every cell would be a dash. Lead
        # with a clear "all failed" block so the user reads the why first.
        block_title = "❌ 全部组合都没有结果" if zh else "❌ No usable results for any combination"
        lines.append(f"### {block_title}\n")
        lines.append(
            f"{'下面列出每个 signal × ticker 的失败原因：' if zh else 'Per-row reasons below:'}\n"
        )
        for r in error_rows:
            lines.append(
                f"- **{r.get('signal', '-')} / {r.get('ticker', '-')}**: "
                f"{_md_cell(r['error'], 200)}"
            )
        tip = _build_validate_recovery_tip(error_rows, zh)
        if tip:
            lines.append("")
            lines.append(tip)
    else:
        headline = "结果汇总" if zh else "Results"
        lines.append(f"### {headline}\n")
        lines.append(
            f"| {'信号' if zh else 'Signal'} | {'股票' if zh else 'Ticker'} | "
            f"IS Sharpe | OOS Sharpe | PBO | DSR | "
            f"{'样本数' if zh else 'N obs'} |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")

        for r in results:
            if "error" in r:
                lines.append(
                    f"| {_md_cell(r.get('signal', '-'))} | {_md_cell(r.get('ticker', '-'))} | "
                    f"– | – | – | – | – |"
                )
                continue
            is_s = r.get("is_sharpe_mean")
            oos_s = r.get("oos_sharpe_mean")
            pbo = (r.get("pbo") or {}).get("pbo")
            dsr = r.get("deflated_sharpe")
            n_obs = r.get("n_obs", "–")
            lines.append(
                f"| {_md_cell(r['signal'])} | {_md_cell(r['ticker'])} | "
                f"{_fmt(is_s)} | {_fmt(oos_s)} | "
                f"{_fmt(pbo) if pbo is not None else '–'} | "
                f"{_fmt(dsr) if dsr is not None else '–'} | {n_obs} |"
            )

        if error_rows:
            lines.append("")
            lines.append(f"### {'失败行' if zh else 'Errors'}\n")
            for r in error_rows:
                lines.append(
                    f"- **{r.get('signal', '-')} / {r.get('ticker', '-')}**: "
                    f"{_md_cell(r['error'], 200)}"
                )
            tip = _build_validate_recovery_tip(error_rows, zh)
            if tip:
                lines.append("")
                lines.append(tip)

        # Interpretation hint
        lines.append("")
        if zh:
            lines.append(
                "> **解读**：`OOS Sharpe` 是关键指标；`PBO` 是过拟合概率（接近 0 越好，"
                "接近 1 表示样本外几乎没有 alpha）；`DSR` 是 Deflated Sharpe Ratio "
                "（多重检验校正后 Sharpe 显著的概率）。"
            )
        else:
            lines.append(
                "> **Interpretation**: `OOS Sharpe` is the headline number; `PBO` is the "
                "Probability of Backtest Overfitting (closer to 0 is better, ≈1 means the "
                "alpha vanishes out-of-sample); `DSR` is the Deflated Sharpe Ratio "
                "(probability the OOS Sharpe is real after multiple-testing correction)."
            )

    artifact_note = (
        "\n---\n📎 **完整数据**：本次运行的 `validation_results.json` 已作为 Action 工件上传，"
        "可在本次 run 页面顶部 *Artifacts* 区域下载。"
        if zh else
        "\n---\n📎 **Full data**: `validation_results.json` is uploaded as a workflow "
        "artifact — download it from the *Artifacts* section at the top of this run page."
    )
    lines.append(artifact_note)

    return "\n".join(lines) + "\n"


def _run_validate(parsed: dict) -> None:
    """Subprocess into ``python -m src.validation.cli evaluate ...`` so the
    CLI keeps a single source of truth, then read the JSON it writes for
    rendering."""
    import subprocess

    lang = parsed.get("lang", "zhCN")
    if lang not in ("en", "zhCN"):
        lang = "zhCN"
    set_lang(lang)

    args = list(parsed["args"])
    # Force the CLI to write to a known path regardless of issue body.
    if "--out" not in args:
        args += ["--out", "validation_results.json"]

    cmd = [sys.executable, "-m", "src.validation.cli", *args]
    sys.stdout.write(f"[bot] running: {' '.join(cmd)}\n")
    completed = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"validation CLI exited {completed.returncode}; see logs above."
        )

    results_path = Path("validation_results.json")
    results = json.loads(results_path.read_text(encoding="utf-8"))

    md = _render_validate_markdown(parsed.get("summary", ""), results, lang)
    Path("comment.md").write_text(md, encoding="utf-8")


# ── Event-study mode (market model + CAR) ────────────────────────────────────


def _default_benchmark(ticker: str) -> str:
    """SPY for US-style tickers; 000300.SS for A-share & .SS/.SZ formats."""
    upper = ticker.upper()
    if upper.endswith(".SS") or upper.endswith(".SZ") or (
        len(upper) == 6 and upper.isdigit()
    ):
        return "000300.SS"
    if upper.endswith(".HK"):
        return "^HSI"
    return "SPY"


def _render_event_study_markdown(parsed_summary: str, payload: dict, lang: str) -> str:
    set_lang(lang)
    zh = lang == "zhCN"

    lines: list[str] = []
    title = "📊 事件研究完成" if zh else "📊 Event study complete"
    params_label = "调用参数" if zh else "Parameters"
    lines.append(f"## {title}\n")
    lines.append(f"**{params_label}**: `{parsed_summary}`\n")

    # Market model fit
    fit_title = "市场模型拟合" if zh else "Market-model fit"
    lines.append(f"### {fit_title}\n")
    lines.append(f"| {'指标' if zh else 'Metric'} | {'值' if zh else 'Value'} |")
    lines.append("|---|---:|")
    lines.append(f"| {'股票' if zh else 'Ticker'} | {payload['ticker']} |")
    lines.append(f"| {'基准' if zh else 'Benchmark'} | {payload['benchmark']} |")
    lines.append(f"| {'事件日期' if zh else 'Event date'} | {payload['event_date']} |")
    lines.append(
        f"| α ({'日均超额' if zh else 'daily intercept'}) | "
        f"{payload['alpha'] * 100:+.4f}% |"
    )
    lines.append(f"| β | {payload['beta']:+.3f} |")
    lines.append(
        f"| {'残差标准差' if zh else 'Residual std'} σₑ | "
        f"{payload['residual_std'] * 100:.4f}% |"
    )
    lines.append(
        f"| {'估计窗口样本' if zh else 'Estimation obs'} | {payload['n_obs']} |"
    )
    lines.append(
        f"| {'事件日 AR' if zh else 'Event-day AR'} | "
        f"{payload['ar_event_day'] * 100:+.4f}% |"
    )
    lines.append("")

    # CAR table
    car_title = "累计超额收益（CAR）与显著性" if zh else "Cumulative abnormal returns (CAR) & significance"
    lines.append(f"### {car_title}\n")
    lines.append(
        f"| {'窗口' if zh else 'Window'} | {'天数' if zh else 'Days'} | "
        f"CAR | t-stat | p-value | {'结论' if zh else 'Verdict'} |"
    )
    lines.append("|---|---:|---:|---:|---:|---|")
    for row in payload["windows"]:
        w = row["window"]
        wlabel = f"({w[0]:+d}, {w[1]:+d})"
        car_pct = row["car"] * 100
        t_stat = row["t_stat"]
        p_val = row["p_value"]
        if t_stat != t_stat:  # NaN
            t_str = "–"
        else:
            t_str = f"{t_stat:+.2f}"
        if p_val != p_val:
            p_str = "–"
            verdict = "–"
        else:
            p_str = f"{p_val:.3f}"
            if p_val < 0.01:
                verdict = "★★★ p<0.01"
            elif p_val < 0.05:
                verdict = "★★ p<0.05"
            elif p_val < 0.10:
                verdict = "★ p<0.10"
            else:
                verdict = "ns" if not zh else "不显著"
        lines.append(
            f"| {wlabel} | {row['L']} | {car_pct:+.3f}% | {t_str} | {p_str} | {verdict} |"
        )
    lines.append("")

    # Interpretation
    if zh:
        lines.append(
            "> **解读**：α/β 在事件前 252 个交易日（与事件保留 30 天间隔以避免污染）"
            "估计；CAR 是事件窗口内超额收益累积。t-stat = CAR / (σₑ · √L)，"
            "L 为窗口天数。p<0.05 即认为该事件在统计上显著推动了超额收益。"
        )
    else:
        lines.append(
            "> **Interpretation**: α/β are estimated over the 252 trading days "
            "preceding the event (with a 30-day gap to avoid contamination). CAR is "
            "the cumulative abnormal return inside the event window. "
            "t-stat = CAR / (σₑ · √L), where L is window length. "
            "p<0.05 means the event statistically moved the stock vs. benchmark."
        )

    artifact_note = (
        "\n---\n📎 **完整数据**：本次运行的 `event_study_result.json` 已作为 Action 工件上传，"
        "可在本次 run 页面顶部 *Artifacts* 区域下载。"
        if zh else
        "\n---\n📎 **Full data**: `event_study_result.json` is uploaded as a workflow "
        "artifact — download it from the *Artifacts* section at the top of this run page."
    )
    lines.append(artifact_note)
    return "\n".join(lines) + "\n"


def _run_event_study(parsed: dict) -> None:
    """Programmatic event-study: fit market model on (event-gap-est..event-gap),
    compute AR over the event window, and run a per-window t-test using
    σₑ from the estimation window."""
    import math

    import numpy as np  # local: scipy/numpy aren't needed for ticker mode
    import pandas as pd
    from scipy.stats import t as t_dist

    from src.event_study import (
        compute_abnormal_returns,
        compute_car,
        fit_market_model,
    )
    from src.tools.api import get_prices, prices_to_df

    es = parsed["event_study"]
    lang = es.get("lang", "zhCN")
    if lang not in ("en", "zhCN"):
        lang = "zhCN"
    set_lang(lang)

    ticker = es["ticker"]
    event_date = es["event_date"]
    window_before = int(es["window_before"])
    window_after = int(es["window_after"])
    estimation_window = int(es["estimation_window"])
    gap = int(es["gap"])
    benchmark = es.get("benchmark") or _default_benchmark(ticker)

    # Pull ~1.7× calendar days to cover trading-day requirements.
    event_dt = datetime.strptime(event_date, "%Y-%m-%d")
    pre_trading_days = estimation_window + gap + window_before + 30
    post_trading_days = window_after + 10
    start_date = (event_dt - relativedelta(days=int(pre_trading_days * 1.7))).strftime(
        "%Y-%m-%d"
    )
    end_date = (event_dt + relativedelta(days=int(post_trading_days * 1.7))).strftime(
        "%Y-%m-%d"
    )

    asset_prices = get_prices(ticker, start_date, end_date)
    bench_prices = get_prices(benchmark, start_date, end_date)
    if not asset_prices:
        raise RuntimeError(f"No price data for ticker {ticker!r}.")
    if not bench_prices:
        raise RuntimeError(f"No price data for benchmark {benchmark!r}.")

    asset_df = prices_to_df(asset_prices)
    bench_df = prices_to_df(bench_prices)
    asset_returns = asset_df["close"].pct_change().rename("asset")
    bench_returns = bench_df["close"].pct_change().rename("bench")

    aligned = pd.concat([asset_returns, bench_returns], axis=1).dropna()
    aligned.columns = ["asset", "bench"]
    if aligned.empty:
        raise RuntimeError("Aligned ticker / benchmark return series is empty.")

    event_ts = pd.Timestamp(event_date)
    pos = int(aligned.index.searchsorted(event_ts))
    if pos >= len(aligned):
        raise RuntimeError(
            f"Event date {event_date} is past the last available trading day "
            f"{aligned.index[-1].strftime('%Y-%m-%d')}."
        )

    mm = fit_market_model(
        aligned["asset"],
        aligned["bench"],
        estimation_window=estimation_window,
        gap=gap,
        event_idx=pos,
    )
    ar = compute_abnormal_returns(aligned["asset"], aligned["bench"], mm)

    user_window = (-window_before, window_after)
    candidates = [(-1, 1), (-3, 3), (-5, 5), (-1, 5), user_window]
    seen: set[tuple[int, int]] = set()
    windows: list[tuple[int, int]] = []
    for w in candidates:
        if w not in seen:
            seen.add(w)
            windows.append(w)

    sigma_e = mm.residual_std
    df_for_t = max(2, mm.n_obs - 2)
    car_rows: list[dict] = []
    for w in windows:
        car = compute_car(ar, event_window=w, event_index=pos)
        L = w[1] - w[0] + 1
        if sigma_e and not math.isnan(sigma_e) and sigma_e > 0 and L > 0:
            t_stat = car / (sigma_e * math.sqrt(L))
            p_val = 2.0 * (1.0 - float(t_dist.cdf(abs(t_stat), df=df_for_t)))
        else:
            t_stat = float("nan")
            p_val = float("nan")
        car_rows.append(
            {"window": list(w), "L": L, "car": float(car), "t_stat": float(t_stat), "p_value": float(p_val)}
        )

    ar_event = float(ar.iloc[pos]) if pos < len(ar) else float("nan")

    payload = {
        "ticker": ticker,
        "benchmark": benchmark,
        "event_date": event_date,
        "alpha": float(mm.alpha),
        "beta": float(mm.beta),
        "residual_std": float(mm.residual_std) if mm.residual_std == mm.residual_std else None,
        "n_obs": int(mm.n_obs),
        "ar_event_day": ar_event,
        "windows": car_rows,
    }
    Path("event_study_result.json").write_text(
        json.dumps(payload, ensure_ascii=False, default=str, indent=2),
        encoding="utf-8",
    )

    md = _render_event_study_markdown(parsed.get("summary", ""), payload, lang)
    Path("comment.md").write_text(md, encoding="utf-8")


# ── Entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    mode = os.environ.get("MODE", "ticker")
    parsed = json.loads(Path("parsed.json").read_text(encoding="utf-8"))
    if not parsed.get("ok"):
        # Should not reach here — workflow checks ok before calling us.
        sys.stderr.write("[bot] parsed.json has ok=false; aborting.\n")
        sys.exit(1)

    if mode == "validate":
        _run_validate(parsed)
        return
    if mode == "event_study":
        _run_event_study(parsed)
        return

    flags = _parse_flags(parsed["args"])
    if mode == "backtester":
        _run_backtester(parsed, flags)
    else:
        _run_ticker(parsed, flags)


if __name__ == "__main__":
    main()
