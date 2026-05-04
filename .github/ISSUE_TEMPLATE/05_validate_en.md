---
name: 🧪 Signal validation (CPCV / PBO)
about: Test whether a quant signal generalises out-of-sample (no LLM cost, runs in seconds)
title: "Signal validation: "
labels: ["bot-validate", "lang-en"]
---

> This template carries the `bot-validate` label. The bot runs
> `python -m src.validation.cli evaluate` against the requested signals
> and tickers. **No LLM calls** — pure-math validation, finishes in
> seconds to tens of seconds.

**Signal keys (required)**

Comma-separated. CPCV is a daily-rolling evaluator, so it only
supports the **technical** signals: `trend`, `mean_reversion`,
`momentum`, `volatility`, `stat_arb`.

Fundamental signals (`value`, `quality`, `earnings_surprise`) update
on report dates rather than every trading day and need a different
evaluator — they're rejected here with a friendly message.

Example: `momentum, mean_reversion, trend`

**Tickers (required)**

Comma-separated, e.g. `AAPL,MSFT,NVDA`.

**Date range (recommended)**

- Start date (YYYY-MM-DD):
- End date (YYYY-MM-DD):

At least 18 months recommended so the 126-day momentum window has
enough history.

**CPCV parameters (optional)**

- n-splits (default 8):
- n-test-splits (default 2):
- rolling-window (default 60; bump to 180 for momentum-style signals):

**Notes**

Anything specific to focus on. The reply will include IS Sharpe / OOS
Sharpe / PBO (Probability of Backtest Overfitting) / Deflated Sharpe
Ratio per signal-ticker pair.
