#!/usr/bin/env python3
"""Daily market report generator for GitHub Actions.

Determines which market triggered the run, checks if today is a trading day,
runs market_review + technical_analyst on major indices, generates a
dashboard report, and writes comment.md for the bot to post.

Environment variables:
  REPORT_MARKET  — "cn", "hk", or "us"
  REPORT_DATE    — optional override (YYYY-MM-DD), defaults to today
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
        "tickers": ["000300.SS", "000001.SS", "^HSI", "SPY", "QQQ"],
        "lookback_days": 90,
    },
    "hk": {
        "name_zh": "港股",
        "name_en": "Hong Kong",
        "tickers": ["^HSI", "000300.SS", "SPY", "QQQ"],
        "lookback_days": 90,
    },
    "us": {
        "name_zh": "美股",
        "name_en": "US Market",
        "tickers": ["SPY", "QQQ", "DIA", "IWM", "000300.SS", "^HSI"],
        "lookback_days": 90,
    },
}


def main() -> None:
    market = os.environ.get("REPORT_MARKET", "cn")
    if market not in MARKET_CONFIG:
        print(f"ERROR: Unknown market '{market}'", file=sys.stderr)
        sys.exit(1)

    cfg = MARKET_CONFIG[market]

    # Date range
    report_date_str = os.environ.get("REPORT_DATE")
    if report_date_str:
        end_dt = datetime.strptime(report_date_str, "%Y-%m-%d")
    else:
        end_dt = datetime.now()
    end_date = end_dt.strftime("%Y-%m-%d")
    start_date = (end_dt - timedelta(days=cfg["lookback_days"])).strftime("%Y-%m-%d")

    # Check trading day
    from src.data.trading_calendar import get_trading_calendar

    cal = get_trading_calendar()
    today = end_dt.date()
    if not cal.is_trading_day(today, market):
        print(f"SKIP: {today} is not a {market} trading day")
        # Write empty comment so the workflow can skip
        with open("comment.md", "w") as f:
            f.write("<!-- skip -->\n")
        sys.exit(0)

    # Run analysis
    tickers = cfg["tickers"]
    from src.main import run_hedge_fund

    result = run_hedge_fund(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        portfolio={
            "cash": 100000,
            "margin_requirement": 0.0,
            "margin_used": 0.0,
            "positions": {t: {"long": 0, "short": 0, "long_cost_basis": 0.0, "short_cost_basis": 0.0, "short_margin_used": 0.0} for t in tickers},
            "realized_gains": {t: {"long": 0.0, "short": 0.0} for t in tickers},
        },
        show_reasoning=True,
        selected_analysts=["market_review", "technical_analyst"],
        model_name="deepseek-v4-flash",
        model_provider="DeepSeek",
    )

    # Generate dashboard
    from src.report.dashboard import DashboardReport

    report = DashboardReport(result)
    md = report.render_markdown()

    # Build full comment with header
    today_str = today.strftime("%Y-%m-%d")
    header = f"## 📊 {cfg['name_zh']} 日报 / {cfg['name_en']} Daily Report — {today_str}\n\n"
    comment = header + md

    # Also write result.json for artifact upload
    with open("result.json", "w") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)

    # Write comment.md
    with open("comment.md", "w") as f:
        f.write(comment)

    # Send notification if NOTIFY_URLS is set
    notify_urls = os.environ.get("NOTIFY_URLS", "").strip()
    if notify_urls:
        from src.notifications import get_notification_manager, NotificationMessage

        nm = get_notification_manager()
        if nm.url_count > 0:
            msg = NotificationMessage(
                title=f"{cfg['name_zh']} 日报 {today_str}",
                body=comment,
            )
            nm.send(msg)
            print(f"Notification sent to {nm.url_count} channel(s)")

    print(f"Report generated: {today_str} / {cfg['name_en']}")


if __name__ == "__main__":
    main()
