---
name: 📈 Backtester (English)
about: Run a historical backtest on a stock (invokes src/backtester.py)
title: "Backtest: "
labels: ["bot-backtester", "lang-en"]
---

> This template is tagged with `bot-backtester`. The bot parses arguments from the body and invokes `uv run python src/backtester.py`. The reply will match the language of this issue.

**Tickers (required)**

For example `AAPL,MSFT`, `600519.SS`, `9988.HK`.

**Backtest date range (recommended)**

- Start date (YYYY-MM-DD):
- End date (YYYY-MM-DD):

**Analysts**

Write `all` to enable every analyst, or list a subset (comma-separated), e.g.:
`warren_buffett, duan_yongping, charlie_munger`

**Notes**

Longer windows and more analysts increase token cost and runtime — set the range thoughtfully.
