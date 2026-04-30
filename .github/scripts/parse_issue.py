"""Parse a GitHub issue title+body via an OpenAI-compatible LLM into hedge-fund CLI args.

Reads from env:
  ISSUE_TITLE, ISSUE_BODY, MODE (ticker|backtester),
  AI_BASE_URL, AI_API_KEY, AI_MODEL.

Writes a JSON object to stdout (always exit 0):
  on success: {"ok": true, "args": [...], "tickers": "...", "summary": "..."}
  on failure: {"ok": false, "reason": "..."}

Designed for stdlib-only execution so it can run before `uv sync` if needed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys
import urllib.error
import urllib.request


_TICKER_RE = re.compile(r"^[A-Za-z0-9.\-]{1,20}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_ANALYST_RE = re.compile(r"^[a-z_]{2,40}$")

# Approximate hedge-fund cost: each "agent run" makes ~1 LLM call. The graph
# fans out (analysts + risk + portfolio) per ticker per day. Backtester loops
# over many days, so the multiplier is large. We cap the projected call count
# so the workflow's 30-minute window has a realistic chance of finishing.
#
# 200 calls × ~8s/call ≈ 27 minutes — leaves a small buffer for parsing,
# data fetching, and rendering.
_ESTIMATED_CALL_LIMIT = 200
_TOTAL_ANALYSTS_WHEN_ALL = 20  # 13 personas + 6 generic + growth


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
    # ~5 trading days per 7 calendar days.
    return max(1, int(round(delta_days * 5 / 7)))


def _estimate_llm_calls(mode: str, n_tickers: int, n_analysts: int, start: str | None, end: str | None) -> int:
    per_day = max(1, n_tickers) * (max(1, n_analysts) + 2)  # +2 = risk + portfolio
    if mode == "backtester":
        return per_day * _trading_days_between(start, end)
    return per_day


def _emit(payload: dict) -> None:
    json.dump(payload, sys.stdout, ensure_ascii=False)
    sys.stdout.write("\n")


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


def _today() -> str:
    return _dt.date.today().isoformat()


def _safe_list(value, validator) -> list[str] | None:
    """Validate a comma-separated string field. Returns the cleaned list or None."""
    if not value or not isinstance(value, str):
        return None
    items = [v.strip() for v in value.split(",") if v.strip()]
    if not items:
        return None
    if not all(validator.match(v) for v in items):
        return None
    return items


def _build_system_prompt() -> str:
    return (
        "You extract hedge-fund CLI arguments from a GitHub issue. "
        "Return ONLY a JSON object with this exact schema:\n"
        '{"ok": bool, "tickers": string|null, "start_date": string|null, '
        '"end_date": string|null, "analysts": string|null, '
        '"show_reasoning": bool, "lang": "en"|"zhCN", "reason": string|null}\n\n'
        "Rules:\n"
        "- tickers: comma-separated, examples: 'AAPL,MSFT' (US), '600519.SS' (Shanghai),\n"
        "  '002594.SZ' (Shenzhen), '9988.HK' (Hong Kong). 6-digit pure numbers should be\n"
        "  classified to Shanghai/Shenzhen automatically by downstream code, so just pass\n"
        "  the bare 6-digit form if unsure.\n"
        "- start_date / end_date: YYYY-MM-DD format only, or null if not specified.\n"
        "- analysts: comma-separated lowercase keys from this list ONLY:\n"
        "  aswath_damodaran, ben_graham, bill_ackman, cathie_wood, charlie_munger,\n"
        "  duan_yongping, michael_burry, mohnish_pabrai, nassim_taleb, peter_lynch,\n"
        "  phil_fisher, rakesh_jhunjhunwala, stanley_druckenmiller, warren_buffett,\n"
        "  technical_analyst, fundamentals_analyst, growth_analyst, news_sentiment_analyst,\n"
        "  sentiment_analyst, valuation_analyst.\n"
        "  Use 'all' to mean every analyst. Null means default (use 'all').\n"
        "- lang: prefer 'zhCN' if the issue is mostly Chinese, else 'en'.\n"
        "- show_reasoning: true if the user explicitly asks to see reasoning details.\n"
        "- ok=false ONLY when tickers cannot be reasonably extracted from title or body;\n"
        "  put a short Chinese explanation in 'reason' so the bot can quote it back.\n"
        "- Never invent tickers. If the user only gave a company name, set ok=false and ask\n"
        "  for the explicit ticker symbol in 'reason'."
    )


def _validate_and_build_args(parsed: dict, mode: str) -> tuple[list[str] | None, str]:
    """Sanity-check the AI output. Returns (args, summary) or (None, reason)."""
    if not parsed.get("ok"):
        return None, parsed.get("reason") or "AI 无法从标题/正文中提取分析参数。"

    tickers_raw = parsed.get("tickers")
    tickers = _safe_list(tickers_raw, _TICKER_RE)
    if not tickers:
        return None, "未能识别有效的股票代码。请在标题或正文中明确写出 ticker，例如 AAPL、600519.SS、9988.HK。"

    args: list[str] = ["--tickers", ",".join(tickers)]
    summary_bits = [f"tickers={','.join(tickers)}"]

    start = parsed.get("start_date")
    if isinstance(start, str) and _DATE_RE.match(start):
        args += ["--start-date", start]
        summary_bits.append(f"start={start}")

    end = parsed.get("end_date")
    if isinstance(end, str) and _DATE_RE.match(end):
        args += ["--end-date", end]
        summary_bits.append(f"end={end}")

    analysts_raw = parsed.get("analysts")
    if analysts_raw == "all" or analysts_raw is None:
        args += ["--analysts-all"]
        summary_bits.append("analysts=all")
    else:
        analysts = _safe_list(analysts_raw, _ANALYST_RE)
        if not analysts:
            return None, f"分析师列表无效：{analysts_raw!r}。请使用 README 中列出的 key。"
        args += ["--analysts", ",".join(analysts)]
        summary_bits.append(f"analysts={','.join(analysts)}")

    lang = parsed.get("lang") if parsed.get("lang") in ("en", "zhCN") else "zhCN"
    args += ["--lang", lang]
    summary_bits.append(f"lang={lang}")

    if parsed.get("show_reasoning") is True and mode == "ticker":
        args += ["--show-reasoning"]
        summary_bits.append("show_reasoning=true")

    model = os.environ.get("AI_MODEL", "").strip()
    if model:
        args += ["--model", model]
        summary_bits.append(f"model={model}")

    # Capacity guard — refuse jobs that have no realistic chance of finishing
    # inside the 30-minute workflow timeout.
    n_analysts = _TOTAL_ANALYSTS_WHEN_ALL if analysts_raw in (None, "all") else len(_safe_list(analysts_raw, _ANALYST_RE) or [])
    estimated = _estimate_llm_calls(mode, len(tickers), n_analysts, start if isinstance(start, str) else None, end if isinstance(end, str) else None)
    if estimated > _ESTIMATED_CALL_LIMIT:
        days = _trading_days_between(start if isinstance(start, str) else None, end if isinstance(end, str) else None)
        reason = (
            f"任务规模超过单次执行上限（预估 {estimated} 次 LLM 调用，上限 {_ESTIMATED_CALL_LIMIT}）。"
            f"当前: tickers={len(tickers)} × analysts={n_analysts}"
            f"{f' × 交易日≈{days}' if mode == 'backtester' else ''}。\n"
            "请减少 ticker 数、缩短日期范围、或把 `all` 改成 2–3 个具体的 analyst key。"
        )
        return None, reason

    return args, "; ".join(summary_bits)


def main() -> None:
    title = os.environ.get("ISSUE_TITLE", "") or ""
    body = os.environ.get("ISSUE_BODY", "") or ""
    mode = os.environ.get("MODE", "ticker")

    if not title.strip():
        _emit({"ok": False, "reason": "Issue 标题为空。"})
        return

    user_msg = (
        f"Mode: {mode}\n"
        f"Today's date: {_today()}\n"
        f"Issue title: {title}\n"
        f"Issue body:\n{body[:4000]}"
    )

    try:
        raw = _call_ai(
            [
                {"role": "system", "content": _build_system_prompt()},
                {"role": "user", "content": user_msg},
            ]
        )
    except urllib.error.HTTPError as e:
        _emit({"ok": False, "reason": f"调用解析 AI 失败：HTTP {e.code} {e.reason}。"})
        return
    except Exception as e:  # noqa: BLE001 - report as bot reply
        _emit({"ok": False, "reason": f"调用解析 AI 失败：{e}"})
        return

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        _emit({"ok": False, "reason": "解析 AI 返回了非 JSON 内容。"})
        return

    args, summary_or_reason = _validate_and_build_args(parsed, mode)
    if args is None:
        _emit({"ok": False, "reason": summary_or_reason})
        return

    _emit(
        {
            "ok": True,
            "args": args,
            "tickers": ",".join(_safe_list(parsed["tickers"], _TICKER_RE) or []),
            "summary": summary_or_reason,
        }
    )


if __name__ == "__main__":
    main()
