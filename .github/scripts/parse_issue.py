"""Parse a GitHub issue title+body via an OpenAI-compatible LLM into bot args.

Reads from env:
  ISSUE_TITLE, ISSUE_BODY, MODE, AI_BASE_URL, AI_API_KEY, AI_MODEL.

MODE selects which schema to extract:
  ticker        — single-day multi-agent analysis  (src/main.py)
  backtester    — multi-day backtest               (src/backtester.py)
  validate      — CPCV + PBO signal evaluation     (src.validation.cli; no LLM)
  event_study   — market model + CAR around event  (no LLM)

Writes a JSON object to stdout (always exit 0):
  on success: {"ok": true, "args": [...], "summary": "..."}
  on failure: {"ok": false, "reason": "..."}

Run via ``uv run python .github/scripts/parse_issue.py`` so src.* is on
sys.path — we read SIGNAL_REGISTRY directly to keep the validate-mode
signal taxonomy in lockstep with the codebase.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

# Make src.* importable when the script is invoked from anywhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.signals import FUNDAMENTAL_SIGNALS, TECHNICAL_SIGNALS  # noqa: E402


_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,20}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ANALYST_RE = re.compile(r"^[a-z_]{2,40}$")
_SIGNAL_RE = re.compile(r"^[a-z_]{2,40}$")

# Validate mode runs CPCV which is daily-rolling, so it only accepts
# technical signals. Fundamentals (value / quality / earnings_surprise)
# update on report dates and need a different evaluator (event-time
# CPCV, not yet implemented) — reject them up front with guidance.
_KNOWN_TECHNICAL_SIGNALS = set(TECHNICAL_SIGNALS)
_KNOWN_FUNDAMENTAL_SIGNALS = set(FUNDAMENTAL_SIGNALS)
_KNOWN_SIGNALS = _KNOWN_TECHNICAL_SIGNALS | _KNOWN_FUNDAMENTAL_SIGNALS

# LLM-call budget for the workflow's 60-minute timeout. Only enforced
# for ticker/backtester modes (validate / event_study don't call LLMs).
# Each LLM round-trip averages ~9s wall time, so 60 min × 60 / 9 ≈ 400.
_ESTIMATED_CALL_LIMIT = 400
_AVG_LLM_SECONDS = 9
_CI_TIMEOUT_MINUTES = 60
_TOTAL_ANALYSTS_WHEN_ALL = 20


# ── helpers ──────────────────────────────────────────────────────────────────


def _trading_days_between(start: str | None, end: str | None) -> int:
    if not start or not end:
        return 1
    try:
        s = _dt.date.fromisoformat(start)
        e = _dt.date.fromisoformat(end)
    except ValueError:
        return 1
    delta_days = (e - s).days
    if delta_days <= 0:
        return 1
    return max(1, int(round(delta_days * 5 / 7)))


def _emit(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


def _today() -> str:
    return _dt.date.today().isoformat()


def _safe_list(value, validator) -> list[str] | None:
    if not value or not isinstance(value, str):
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items or not all(validator.match(v) for v in items):
        return None
    return items


def _call_ai(messages: list[dict]) -> str:
    base = os.environ["AI_BASE_URL"].rstrip("/")
    key = os.environ["AI_API_KEY"]
    model = os.environ["AI_MODEL"]
    payload = json.dumps(
        {
            "model": model,
            "messages": messages,
            "response_format": {"type": "json_object"},
            "temperature": 0.1,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/chat/completions",
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = json.loads(resp.read())
    return body["choices"][0]["message"]["content"]


# ── per-mode prompts ─────────────────────────────────────────────────────────


_SCHEMA_TICKER = (
    "You extract hedge-fund analysis arguments from a GitHub issue. "
    "Return ONLY a JSON object:\n"
    '{"ok": bool, "tickers": string|null, "start_date": string|null, '
    '"end_date": string|null, "analysts": string|null, '
    '"show_reasoning": bool, "lang": "en"|"zhCN", "reason": string|null}\n\n'
    "Rules:\n"
    "- tickers: comma-separated, e.g. 'AAPL,MSFT' (US), '600519.SS' (Shanghai),\n"
    "  '002594.SZ' (Shenzhen), '9988.HK' (HK). Bare 6-digit numbers are auto-classified.\n"
    "- start_date / end_date: YYYY-MM-DD only or null.\n"
    "- analysts: comma-separated lowercase keys: aswath_damodaran, ben_graham,\n"
    "  bill_ackman, cathie_wood, charlie_munger, duan_yongping, michael_burry,\n"
    "  mohnish_pabrai, nassim_taleb, peter_lynch, phil_fisher, rakesh_jhunjhunwala,\n"
    "  stanley_druckenmiller, warren_buffett, technical_analyst, fundamentals_analyst,\n"
    "  growth_analyst, news_sentiment_analyst, sentiment_analyst, valuation_analyst.\n"
    "  Use 'all' for everything. Null means default (= all).\n"
    "- lang: 'zhCN' if the issue is mostly Chinese, else 'en'.\n"
    "- ok=false ONLY when tickers cannot be extracted; put a short reason.\n"
    "- Never invent tickers."
)

_SCHEMA_VALIDATE = (
    "You extract signal-validation arguments from a GitHub issue. "
    "Return ONLY a JSON object:\n"
    '{"ok": bool, "signals": string|null, "tickers": string|null, '
    '"start_date": string|null, "end_date": string|null, '
    '"n_splits": int|null, "n_test_splits": int|null, '
    '"rolling_window": int|null, "lang": "en"|"zhCN", "reason": string|null}\n\n'
    "Rules:\n"
    "- signals: comma-separated keys from this set ONLY: "
    + ", ".join(sorted(TECHNICAL_SIGNALS))
    + ".\n"
    "  These are the technical signals that CPCV can evaluate.\n"
    "  Fundamental signals (" + ", ".join(sorted(FUNDAMENTAL_SIGNALS)) + ")\n"
    "  do NOT belong here — pass them through verbatim if the user mentions\n"
    "  them and the validator below will reject with guidance.\n"
    "- tickers: comma-separated US/HK/CN tickers (e.g. AAPL, 9988.HK, 600519.SS).\n"
    "- start_date / end_date: YYYY-MM-DD or null.\n"
    "- n_splits (default 8), n_test_splits (default 2), rolling_window (default 60).\n"
    "  Bump rolling_window to 180 if any momentum/volatility-style signal is requested.\n"
    "- lang: 'zhCN' if the issue is mostly Chinese, else 'en'.\n"
    "- ok=false if signals or tickers cannot be extracted."
)

_SCHEMA_EVENT_STUDY = (
    "You extract event-study arguments from a GitHub issue. "
    "Return ONLY a JSON object:\n"
    '{"ok": bool, "ticker": string|null, "event_date": string|null, '
    '"window_before": int|null, "window_after": int|null, '
    '"estimation_window": int|null, "gap": int|null, '
    '"benchmark": string|null, "lang": "en"|"zhCN", "reason": string|null}\n\n'
    "Rules:\n"
    "- ticker: a single ticker (AAPL, 600519.SS, 9988.HK).\n"
    "- event_date: YYYY-MM-DD, the day of the event.\n"
    "- window_before / window_after: trading days relative to event_date,\n"
    "  e.g. for window (-3, +3) → window_before=3, window_after=3 (BOTH POSITIVE INTEGERS).\n"
    "- estimation_window (default 252), gap (default 30): both positive integers.\n"
    "- benchmark: ticker or null. Defaults: 'SPY' for US, '000300.SS' for CN.\n"
    "- lang: 'zhCN' if mostly Chinese, else 'en'.\n"
    "- ok=false if ticker or event_date can't be extracted."
)


# ── per-mode validators ──────────────────────────────────────────────────────


def _validate_ticker(parsed: dict, mode: str) -> tuple[list[str] | None, str]:
    """ticker / backtester modes share the same args (mode just changes which CLI runs)."""
    if not parsed.get("ok"):
        return None, parsed.get("reason") or "AI 无法从标题/正文中提取分析参数。"

    tickers_raw = parsed.get("tickers")
    tickers = _safe_list(tickers_raw, _TICKER_RE)
    if not tickers:
        return None, "未能识别有效的股票代码。请在标题或正文中明确写出 ticker，例如 AAPL、600519.SS、9988.HK。"

    args: list[str] = ["--tickers", ",".join(tickers)]
    bits = [f"tickers={','.join(tickers)}"]

    start = parsed.get("start_date")
    if isinstance(start, str) and _DATE_RE.match(start):
        args += ["--start-date", start]
        bits.append(f"start={start}")

    end = parsed.get("end_date")
    if isinstance(end, str) and _DATE_RE.match(end):
        args += ["--end-date", end]
        bits.append(f"end={end}")

    analysts_raw = parsed.get("analysts")
    if analysts_raw == "all" or analysts_raw is None:
        args += ["--analysts-all"]
        bits.append("analysts=all")
    else:
        analysts = _safe_list(analysts_raw, _ANALYST_RE)
        if not analysts:
            return None, f"分析师列表无效：{analysts_raw!r}。请使用 README 中列出的 key。"
        args += ["--analysts", ",".join(analysts)]
        bits.append(f"analysts={','.join(analysts)}")

    lang = parsed.get("lang") if parsed.get("lang") in ("en", "zhCN") else "zhCN"
    args += ["--lang", lang]
    bits.append(f"lang={lang}")

    if parsed.get("show_reasoning") is True and mode == "ticker":
        args += ["--show-reasoning"]
        bits.append("show_reasoning=true")

    model = os.environ.get("AI_MODEL", "").strip()
    if model:
        args += ["--model", model]
        bits.append(f"model={model}")

    # Capacity guard.
    n_analysts = _TOTAL_ANALYSTS_WHEN_ALL if analysts_raw in (None, "all") else len(_safe_list(analysts_raw, _ANALYST_RE) or [])
    days = _trading_days_between(
        start if isinstance(start, str) else None,
        end if isinstance(end, str) else None,
    )
    per_day = max(1, len(tickers)) * (max(1, n_analysts) + 2)
    estimated = per_day * (days if mode == "backtester" else 1)
    if estimated > _ESTIMATED_CALL_LIMIT:
        return None, _CAPACITY_PREFIX + _format_capacity_error(
            mode=mode,
            lang=lang,
            tickers=tickers,
            n_analysts=n_analysts,
            analysts_raw=analysts_raw,
            days=days,
            per_day=per_day,
            estimated=estimated,
        )
    return args, "; ".join(bits)


# Sentinel prefix on capacity-error reasons so the workflow comment
# renderer can recognise the message is self-contained (don't double-
# wrap with a generic header / example body).
_CAPACITY_PREFIX = "<!-- self-contained -->\n"


def _format_capacity_error(
    *,
    mode: str,
    lang: str,
    tickers: list[str],
    n_analysts: int,
    analysts_raw: str | None,
    days: int,
    per_day: int,
    estimated: int,
) -> str:
    """Build the user-visible message when a ticker/backtester request exceeds the LLM-call budget."""
    zh = lang == "zhCN"
    n_tickers = max(1, len(tickers))
    estimated_minutes = max(1, round(estimated * _AVG_LLM_SECONDS / 60))

    # Suggest concrete parameter combinations that *would* fit in the budget.
    suggestions: list[tuple[str, int, int]] = []  # (analysts_label, n_analysts_choice, days_max)
    if mode == "backtester":
        for label, n in (("warren_buffett, duan_yongping", 2),
                         ("warren_buffett, charlie_munger, duan_yongping", 3),
                         ("all (20 analysts)", _TOTAL_ANALYSTS_WHEN_ALL)):
            d = max(1, _ESTIMATED_CALL_LIMIT // (n_tickers * (n + 2)))
            suggestions.append((label, n, d))

    if zh:
        lines: list[str] = []
        lines.append("⚠️ **任务规模超过单次执行上限，机器人提前拒绝**\n")
        lines.append(f"**预估 LLM 调用次数：{estimated} 次**（上限 {_ESTIMATED_CALL_LIMIT}）\n")
        lines.append("**算法拆解**：")
        lines.append(f"- tickers = {n_tickers}")
        analysts_disp = "all" if analysts_raw in (None, "all") else (analysts_raw or str(n_analysts))
        lines.append(f"- analysts = {n_analysts}（输入：{analysts_disp}；+2 = risk_management + portfolio_manager）")
        if mode == "backtester":
            lines.append(f"- trading days ≈ {days}（按 5/7 折算自然日 → 交易日）")
            lines.append(f"- 每日调用 = {n_tickers} × ({n_analysts} + 2) = {per_day}")
            lines.append(f"- 总数 = {per_day} × {days} = {estimated}")
        else:
            lines.append(f"- 单次 = {n_tickers} × ({n_analysts} + 2) = {per_day}")
        lines.append("")
        lines.append("**为什么不能执行**")
        lines.append(
            f"GitHub Actions workflow timeout 是 {_CI_TIMEOUT_MINUTES} 分钟，"
            f"每次 LLM 调用平均约 {_AVG_LLM_SECONDS} 秒，"
            f"上限 ≈ {_CI_TIMEOUT_MINUTES} × 60 / {_AVG_LLM_SECONDS} ≈ {_ESTIMATED_CALL_LIMIT} 次。"
            f"本任务预估耗时 ≈ {estimated_minutes} 分钟，会被 timeout 中止。"
        )
        if suggestions:
            lines.append("")
            lines.append("**可行的参数组合**（任选其一，编辑本 issue 即重新触发）：\n")
            lines.append("| analysts | 最长交易日 | 对应日历窗口 |")
            lines.append("|---|---:|---|")
            for label, _, d in suggestions:
                cal = max(1, round(d * 7 / 5))
                lines.append(f"| `{label}` | ~{d} 天 | ~{cal} 天 |")
        else:
            lines.append("")
            lines.append("**建议**：减少 ticker 数、把 `all` 换成 2–3 个具体 analyst key。")
        return "\n".join(lines)

    # English
    lines = []
    lines.append("⚠️ **Request exceeds the per-run LLM-call budget — rejected up front**\n")
    lines.append(f"**Estimated LLM calls: {estimated}** (limit {_ESTIMATED_CALL_LIMIT})\n")
    lines.append("**Breakdown**:")
    lines.append(f"- tickers = {n_tickers}")
    analysts_disp = "all" if analysts_raw in (None, "all") else (analysts_raw or str(n_analysts))
    lines.append(f"- analysts = {n_analysts} (input: {analysts_disp}; +2 = risk_management + portfolio_manager)")
    if mode == "backtester":
        lines.append(f"- trading days ≈ {days} (calendar days × 5/7)")
        lines.append(f"- per-day calls = {n_tickers} × ({n_analysts} + 2) = {per_day}")
        lines.append(f"- total = {per_day} × {days} = {estimated}")
    else:
        lines.append(f"- single run = {n_tickers} × ({n_analysts} + 2) = {per_day}")
    lines.append("")
    lines.append("**Why this can't run**")
    lines.append(
        f"The GitHub Actions workflow timeout is {_CI_TIMEOUT_MINUTES} minutes; each LLM "
        f"call takes ~{_AVG_LLM_SECONDS}s, so the practical ceiling is "
        f"~{_CI_TIMEOUT_MINUTES} × 60 / {_AVG_LLM_SECONDS} ≈ {_ESTIMATED_CALL_LIMIT} calls. "
        f"This task would take ~{estimated_minutes} min and hit the timeout."
    )
    if suggestions:
        lines.append("")
        lines.append("**Parameter combos that fit** (pick one, edit the issue to retrigger):\n")
        lines.append("| analysts | max trading days | calendar window |")
        lines.append("|---|---:|---|")
        for label, _, d in suggestions:
            cal = max(1, round(d * 7 / 5))
            lines.append(f"| `{label}` | ~{d} days | ~{cal} days |")
    else:
        lines.append("")
        lines.append("**Suggestion**: drop ticker count, replace `all` with 2–3 specific analyst keys.")
    return "\n".join(lines)


def _validate_validate_mode(parsed: dict) -> tuple[list[str] | None, str]:
    """CPCV signal evaluation — no LLM cost."""
    if not parsed.get("ok"):
        return None, parsed.get("reason") or "AI 无法从标题/正文中提取信号验证参数。"

    signals = _safe_list(parsed.get("signals"), _SIGNAL_RE)
    if not signals or any(s not in _KNOWN_SIGNALS for s in signals):
        return None, (
            "未能识别有效的 signal key。请使用以下之一: "
            f"{', '.join(sorted(_KNOWN_SIGNALS))}"
        )

    # CPCV is a daily-rolling evaluator, so fundamental signals
    # (which only update on report dates) can't be measured by it.
    # Reject up front with the list of valid technical signals.
    fundamentals = [s for s in signals if s in _KNOWN_FUNDAMENTAL_SIGNALS]
    if fundamentals:
        return None, (
            f"以下 signal 是基本面（fundamental）信号，CPCV 只能评估技术（technical）信号："
            f"{', '.join(fundamentals)}。请改用以下技术信号之一："
            f"{', '.join(sorted(_KNOWN_TECHNICAL_SIGNALS))}。\n\n"
            f"(CPCV 按日滚动重算信号；fundamental 信号只在财报时点更新，"
            f"用日频回测无意义。)"
        )

    tickers = _safe_list(parsed.get("tickers"), _TICKER_RE)
    if not tickers:
        return None, "未能识别 ticker。请在正文中给出，例如 AAPL,MSFT。"

    start = parsed.get("start_date")
    end = parsed.get("end_date")
    if not (isinstance(start, str) and _DATE_RE.match(start)):
        return None, "请给出有效的开始日期 (YYYY-MM-DD)。"
    if not (isinstance(end, str) and _DATE_RE.match(end)):
        return None, "请给出有效的结束日期 (YYYY-MM-DD)。"

    args = ["evaluate", "--signal", ",".join(signals), "--ticker", ",".join(tickers),
            "--start", start, "--end", end]
    bits = [f"signals={','.join(signals)}", f"tickers={','.join(tickers)}",
            f"window={start}→{end}"]

    for key, flag, default in (
        ("n_splits", "--n-splits", 8),
        ("n_test_splits", "--n-test-splits", 2),
        ("rolling_window", "--rolling-window", 60),
    ):
        v = parsed.get(key)
        if isinstance(v, int) and 0 < v < 1000:
            args += [flag, str(v)]
            bits.append(f"{key}={v}")
        else:
            bits.append(f"{key}={default}")

    return args, "; ".join(bits)


def _validate_event_study(parsed: dict) -> tuple[dict | None, str]:
    """Event study — returns a dict of fields (not a CLI arg list).

    Event study has no CLI; run_for_bot.py reads the structured payload
    directly from parsed.json's ``event_study`` field.
    """
    if not parsed.get("ok"):
        return None, parsed.get("reason") or "AI 无法从标题/正文中提取事件研究参数。"

    ticker = parsed.get("ticker")
    if not isinstance(ticker, str) or not _TICKER_RE.match(ticker):
        return None, "请给出有效的单个 ticker（例如 AAPL、600519.SS）。"

    event_date = parsed.get("event_date")
    if not (isinstance(event_date, str) and _DATE_RE.match(event_date)):
        return None, "请给出有效的事件日期 (YYYY-MM-DD)。"

    def _pos_int(value, default: int, lo: int = 1, hi: int = 1000) -> int:
        try:
            v = int(value)
        except (TypeError, ValueError):
            return default
        return v if lo <= v <= hi else default

    payload = {
        "ticker": ticker,
        "event_date": event_date,
        "window_before": _pos_int(parsed.get("window_before"), 3, lo=0, hi=120),
        "window_after": _pos_int(parsed.get("window_after"), 3, lo=0, hi=120),
        "estimation_window": _pos_int(parsed.get("estimation_window"), 252, lo=30, hi=1000),
        "gap": _pos_int(parsed.get("gap"), 30, lo=0, hi=120),
    }
    benchmark = parsed.get("benchmark")
    if isinstance(benchmark, str) and _TICKER_RE.match(benchmark):
        payload["benchmark"] = benchmark
    payload["lang"] = parsed.get("lang") if parsed.get("lang") in ("en", "zhCN") else "zhCN"

    bits = [
        f"ticker={ticker}", f"event={event_date}",
        f"window=(-{payload['window_before']},+{payload['window_after']})",
        f"est={payload['estimation_window']}d", f"gap={payload['gap']}d",
        f"lang={payload['lang']}",
    ]
    if "benchmark" in payload:
        bits.append(f"benchmark={payload['benchmark']}")
    return payload, "; ".join(bits)


# ── entry point ──────────────────────────────────────────────────────────────


def main() -> None:
    title = (os.environ.get("ISSUE_TITLE", "") or "").strip()
    body = os.environ.get("ISSUE_BODY", "") or ""
    mode = os.environ.get("MODE", "ticker")

    if not title:
        _emit({"ok": False, "reason": "Issue 标题为空。"})
        return

    schema = {
        "ticker": _SCHEMA_TICKER,
        "backtester": _SCHEMA_TICKER,
        "validate": _SCHEMA_VALIDATE,
        "event_study": _SCHEMA_EVENT_STUDY,
    }.get(mode, _SCHEMA_TICKER)

    user_msg = (
        f"Mode: {mode}\n"
        f"Today's date: {_today()}\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body[:4000]}"
    )

    try:
        raw = _call_ai(
            [
                {"role": "system", "content": schema},
                {"role": "user", "content": user_msg},
            ]
        )
    except urllib.error.HTTPError as e:
        _emit({"ok": False, "reason": f"调用解析 AI 失败：HTTP {e.code} {e.reason}。"})
        return
    except Exception as e:  # noqa: BLE001
        _emit({"ok": False, "reason": f"调用解析 AI 失败：{e}"})
        return

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _emit({"ok": False, "reason": "解析 AI 返回了非 JSON 内容。"})
        return

    if mode in ("ticker", "backtester"):
        args, summary = _validate_ticker(parsed, mode)
        if args is None:
            _emit({"ok": False, "reason": summary})
            return
        _emit({"ok": True, "args": args, "summary": summary})

    elif mode == "validate":
        args, summary = _validate_validate_mode(parsed)
        if args is None:
            _emit({"ok": False, "reason": summary})
            return
        lang = parsed.get("lang") if parsed.get("lang") in ("en", "zhCN") else "zhCN"
        _emit({"ok": True, "args": args, "summary": summary, "lang": lang})

    elif mode == "event_study":
        payload, summary = _validate_event_study(parsed)
        if payload is None:
            _emit({"ok": False, "reason": summary})
            return
        _emit({"ok": True, "event_study": payload, "summary": summary})

    else:
        _emit({"ok": False, "reason": f"未知模式: {mode!r}"})


if __name__ == "__main__":
    main()
