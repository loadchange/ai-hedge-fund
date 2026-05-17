#!/usr/bin/env python3
"""Market report generator for GitHub Actions.

Supports daily, weekly, and monthly report types.
Runs market_review + technical_analyst on major indices (all rule-based, no LLM),
generates a detailed report, and writes comment.md for the bot to post.

Environment variables:
  REPORT_MARKET  — "cn", "hk", or "us"
  REPORT_TYPE    — "daily", "weekly", or "monthly" (default: daily)
  REPORT_DATE    — optional override (YYYY-MM-DD), defaults to today
  NOTIFY_URLS    — Apprise URL for notification (optional)
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date, datetime, timedelta

# ── Market configs ──────────────────────────────────────────────────────────

MARKET_CONFIG = {
    "cn": {
        "name_zh": "中国A股",
        "name_en": "China A-Share",
        "tickers": [
            # 上证系列
            "000001.SS",  # 上证综指
            "000016.SS",  # 上证50
            "000688.SS",  # 科创50
            # 深证系列
            "399001.SZ",  # 深证成指
            "399006.SZ",  # 创业板指
            # 中证系列
            "000300.SS",  # 沪深300
            "000905.SS",  # 中证500
            "000852.SS",  # 中证1000
            # 港美参考
            "^HSI", "SPY", "QQQ",
        ],
        "lookback_days": 90,
    },
    "hk": {
        "name_zh": "港股",
        "name_en": "Hong Kong",
        "tickers": [
            # 香港指数
            "^HSI",  # 恒生指数
            # A股参考
            "000300.SS", "000001.SS",
            # 美股参考
            "SPY", "QQQ",
        ],
        "lookback_days": 90,
    },
    "us": {
        "name_zh": "美股",
        "name_en": "US Market",
        "tickers": [
            # 三大旗舰
            "^DJI",   # 道琼斯工业平均
            "^GSPC",   # 标普500
            "^IXIC",   # 纳斯达克综合
            "^NDX",    # 纳斯达克100
            # 宽基延伸
            "^RUT",    # 罗素2000 (小盘)
            "^W5000",  # 威尔逊5000 (全市场)
            # 行业指数
            "^SOX",    # 费城半导体
            "^NBI",    # 纳斯达克生物科技
            # A股/港股参考
            "000300.SS", "^HSI",
        ],
        "lookback_days": 90,
    },
}

TICKER_NAMES = {
    # 上证系列
    "000001.SS": "上证综指",
    "000016.SS": "上证50",
    "000688.SS": "科创50",
    "000010.SS": "上证180",
    # 深证系列
    "399001.SZ": "深证成指",
    "399006.SZ": "创业板指",
    "399330.SZ": "深证100",
    # 中证系列
    "000300.SS": "沪深300",
    "000905.SS": "中证500",
    "000852.SS": "中证1000",
    # 港股
    "^HSI": "恒生指数",
    # 美股旗舰
    "^DJI": "道琼斯工业",
    "^GSPC": "标普500",
    "^IXIC": "纳斯达克综指",
    "^NDX": "纳斯达克100",
    # 美股宽基
    "^RUT": "罗素2000",
    "^W5000": "威尔逊5000",
    # 美股行业
    "^SOX": "费城半导体",
    "^NBI": "纳斯达克生技",
    # ETF
    "SPY": "S&P 500 ETF",
    "QQQ": "纳斯达克100 ETF",
    "DIA": "道琼斯ETF",
    "IWM": "罗素2000 ETF",
}

TREND_EMOJI = {"bullish": "📈", "bearish": "📉", "neutral": "➡️"}
SIGNAL_EMOJI = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}

REPORT_TYPE_LABEL = {
    "daily": ("日报", "Daily Report"),
    "weekly": ("周报", "Weekly Report"),
    "monthly": ("月报", "Monthly Report"),
}


def _fmt_change(pct: float) -> str:
    if pct >= 0:
        return f"+{pct:.2f}%"
    return f"{pct:.2f}%"


def build_report(cfg: dict, report_date: str, analyst_signals: dict, report_type: str = "daily") -> str:
    """Build the full Markdown report from raw analyst signals."""
    lines: list[str] = []
    name_zh = cfg["name_zh"]
    name_en = cfg["name_en"]
    tickers = cfg["tickers"]
    type_zh, type_en = REPORT_TYPE_LABEL.get(report_type, ("日报", "Daily Report"))

    # ── 1. Market Overview (from market_review_agent) ─────────────────────
    mr = analyst_signals.get("market_review_agent", {})
    overview = mr.get("market_overview", {})
    mr_signal = mr.get("signal", "neutral")
    mr_conf = mr.get("confidence", 0)

    lines.append(f"## 📊 {name_zh} {type_zh} / {name_en} {type_en} — {report_date}\n")

    # Overall signal
    lines.append(f"**大盘信号**: {SIGNAL_EMOJI.get(mr_signal, '⚪')} {mr_signal.upper()} (置信度 {mr_conf}%)\n")

    # Index table
    lines.append("### 🌍 全球指数概览\n")
    lines.append("| 指数 | 最新价 | 日涨跌 | 周涨跌 | 波动率 | 趋势 |")
    lines.append("|:-----|-------:|-------:|-------:|-------:|:-----|")

    for market_key in ["us", "cn", "hk"]:
        market_data = overview.get(market_key, {})
        for idx_ticker, m in market_data.get("indices", {}).items():
            label = TICKER_NAMES.get(idx_ticker, idx_ticker)
            daily = _fmt_change(m.get("change_pct", 0))
            weekly = _fmt_change(m.get("weekly_change_pct", 0))
            vol = f"{m.get('volatility', 0):.1f}%"
            price = f"{m.get('latest_price', 0):.2f}"
            trend = m.get("trend", "neutral")
            emoji = TREND_EMOJI.get(trend, "")
            lines.append(f"| {label} ({idx_ticker}) | {price} | {daily} | {weekly} | {vol} | {emoji} {trend.upper()} |")
    lines.append("")

    # ── 2. Technical Analysis (from technical_analyst_agent) ──────────────
    ta_data = analyst_signals.get("technical_analyst_agent", {})

    lines.append("### 📈 技术分析详情\n")

    for ticker in tickers:
        ta = ta_data.get(ticker, {})
        if not ta:
            continue

        label = TICKER_NAMES.get(ticker, ticker)
        signal = ta.get("signal", "neutral")
        confidence = ta.get("confidence", 0)
        reasoning = ta.get("reasoning", {})

        lines.append(f"#### {SIGNAL_EMOJI.get(signal, '⚪')} {label} ({ticker}) — {signal.upper()} {confidence}%\n")

        # Trend
        trend = reasoning.get("trend", {})
        if trend:
            ts = trend.get("signal", "neutral")
            tc = trend.get("confidence", 0)
            metrics = trend.get("metrics", {})
            adx = metrics.get("adx", 0)
            lines.append(f"- **趋势**: {ts.upper()} ({tc}%) — ADX: {adx:.1f}")

        # Mean reversion
        mr_section = reasoning.get("mean_reversion", {})
        if mr_section:
            metrics = mr_section.get("metrics", {})
            rsi14 = metrics.get("rsi_14", 0)
            zscore = metrics.get("z_score", 0)
            bb = metrics.get("price_vs_bb", 0)
            lines.append(f"- **均值回归**: RSI(14): {rsi14:.1f}, Z-Score: {zscore:.2f}, BB位置: {bb:.1%}")

        # Momentum
        mom = reasoning.get("momentum", {})
        if mom:
            metrics = mom.get("metrics", {})
            m1m = metrics.get("momentum_1m", 0)
            vm = metrics.get("volume_momentum", 0)
            lines.append(f"- **动量**: 1月动量: {_fmt_change(m1m * 100)}, 成交量动量: {vm:.2f}")

        # Volatility
        vol_section = reasoning.get("volatility", {})
        if vol_section:
            metrics = vol_section.get("metrics", {})
            hv = metrics.get("historical_volatility", 0)
            atr = metrics.get("atr_ratio", 0)
            lines.append(f"- **波动率**: 年化: {hv:.1%}, ATR比率: {atr:.2%}")

        lines.append("")

    # ── 3. Summary ────────────────────────────────────────────────────────
    lines.append("### 📝 总结\n")

    # Aggregate signals
    bullish = []
    bearish = []
    neutral = []
    for ticker in tickers:
        ta = ta_data.get(ticker, {})
        if not ta:
            continue
        s = ta.get("signal", "neutral")
        label = TICKER_NAMES.get(ticker, ticker)
        if s == "bullish":
            bullish.append(label)
        elif s == "bearish":
            bearish.append(label)
        else:
            neutral.append(label)

    if bullish:
        lines.append(f"- 🟢 **看多**: {', '.join(bullish)}")
    if bearish:
        lines.append(f"- 🔴 **看空**: {', '.join(bearish)}")
    if neutral:
        lines.append(f"- 🟡 **中性**: {', '.join(neutral)}")
    lines.append("")
    lines.append("> ⚠️ 以上分析仅供参考，不构成投资建议。")

    return "\n".join(lines)


def main() -> None:
    market = os.environ.get("REPORT_MARKET", "cn")
    report_type = os.environ.get("REPORT_TYPE", "daily")
    if market not in MARKET_CONFIG:
        print(f"ERROR: Unknown market '{market}'", file=sys.stderr)
        sys.exit(1)
    if report_type not in ("daily", "weekly", "monthly"):
        print(f"ERROR: Unknown report type '{report_type}'", file=sys.stderr)
        sys.exit(1)

    cfg = MARKET_CONFIG[market]

    # Lookback based on report type
    lookback = cfg["lookback_days"]
    if report_type == "weekly":
        lookback = max(lookback, 180)
    elif report_type == "monthly":
        lookback = max(lookback, 365)

    # Date range
    report_date_str = os.environ.get("REPORT_DATE")
    if report_date_str:
        end_dt = datetime.strptime(report_date_str, "%Y-%m-%d")
    else:
        end_dt = datetime.now()
    end_date = end_dt.strftime("%Y-%m-%d")
    start_date = (end_dt - timedelta(days=lookback)).strftime("%Y-%m-%d")

    # Check trading day (only skip for daily reports)
    from src.data.trading_calendar import get_trading_calendar

    cal = get_trading_calendar()
    today = end_dt.date()
    if report_type == "daily" and not cal.is_trading_day(today, market):
        print(f"SKIP: {today} is not a {market} trading day")
        with open("comment.md", "w") as f:
            f.write("<!-- skip -->\n")
        sys.exit(0)

    # Build agent state
    from langchain_core.messages import HumanMessage

    state = {
        "messages": [HumanMessage(content="Analyze major market indices.")],
        "data": {
            "tickers": cfg["tickers"],
            "portfolio": {
                "cash": 100000,
                "margin_requirement": 0.0,
                "margin_used": 0.0,
                "positions": {},
                "realized_gains": {},
            },
            "start_date": start_date,
            "end_date": end_date,
            "analyst_signals": {},
        },
        "metadata": {
            "show_reasoning": True,
            "model_name": "rule-based",
            "model_provider": "rule-based",
        },
    }

    # Run rule-based agents directly (no LLM needed)
    from src.agents.market_review import market_review_agent
    from src.agents.technicals import technical_analyst_agent

    print("Running market_review_agent...")
    result_mr = market_review_agent(state)
    state["messages"] = result_mr.get("messages", state["messages"])
    state["data"] = result_mr.get("data", state["data"])

    print("Running technical_analyst_agent...")
    result_ta = technical_analyst_agent(state)
    state["messages"] = result_ta.get("messages", state["messages"])
    state["data"] = result_ta.get("data", state["data"])

    analyst_signals = state["data"]["analyst_signals"]

    # Build report
    report_date = today.strftime("%Y-%m-%d")
    comment = build_report(cfg, report_date, analyst_signals, report_type)

    # Write outputs
    with open("result.json", "w") as f:
        json.dump({"analyst_signals": analyst_signals}, f, ensure_ascii=False, indent=2, default=str)
    with open("comment.md", "w") as f:
        f.write(comment)

    # Send notification
    type_zh = REPORT_TYPE_LABEL.get(report_type, ("日报",))[0]
    notify_urls = os.environ.get("NOTIFY_URLS", "").strip()
    if notify_urls:
        from src.notifications import get_notification_manager, NotificationMessage

        nm = get_notification_manager()
        if nm.url_count > 0:
            msg = NotificationMessage(
                title=f"{cfg['name_zh']} {type_zh} {report_date}",
                body=comment,
            )
            nm.send(msg)
            print(f"Notification sent to {nm.url_count} channel(s)")

    print(f"Report generated: {report_date} / {cfg['name_en']}")


if __name__ == "__main__":
    main()
