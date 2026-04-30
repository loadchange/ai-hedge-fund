---
name: 🤖 Ticker analysis (English)
about: Run a full multi-agent analysis on a stock (invokes src/main.py)
title: "Ticker analysis: "
labels: ["bot-ticker", "lang-en"]
---

> This template is tagged with `bot-ticker`. The bot parses arguments from the body and invokes `uv run python src/main.py`. The reply will match the language of this issue (English issue → English reply, Chinese issue → Chinese reply).

**Tickers (required)**

For example `AAPL,MSFT`, `600519.SS`, `9988.HK`. Bare 6-digit codes are auto-classified to Shanghai/Shenzhen.

**Date range (optional)**

- Start date (YYYY-MM-DD):
- End date (YYYY-MM-DD):

**Analysts**

Write `all` to enable every analyst, or list a subset (comma-separated), e.g.:
`warren_buffett, duan_yongping, charlie_munger`

The full key list is in the [README](../../#agent-roster) or `src/utils/analysts.py`.

**Notes**

Anything you'd like the bot to pay special attention to.
