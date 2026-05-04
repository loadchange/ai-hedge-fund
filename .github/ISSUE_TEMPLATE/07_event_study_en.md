---
name: 📊 Event study (market model + CAR)
about: Quantify abnormal returns around a specific event date (no LLM cost)
title: "Event study: "
labels: ["bot-event-study", "lang-en"]
---

> This template carries the `bot-event-study` label. The bot pulls the
> ticker + benchmark (defaults: `SPY` for US, `000300.SS` for CN), fits
> a 252-day market-model α/β with a 30-day pre-event gap, then reports
> abnormal returns and CARs over the requested window plus significance
> tests. **No LLM calls.**

**Ticker (required)**

A single ticker, e.g. `AAPL` or `600519.SS`.

**Event date (required)**

Trading day of the event, YYYY-MM-DD. Example: `2025-02-13` (NVDA earnings).

**Event window (optional)**

`(t_before, t_after)` in trading days relative to the event. Default
`(-3, +3)` (3 days each side).

Examples: `(-1, +1)`, `(-5, +20)`.

**Estimation window (optional)**

- estimation-window (default 252 days):
- gap (days between estimation end and event, default 30):

**Benchmark (optional)**

Defaults: `SPY` for US, `000300.SS` for CN. Override with any ticker
(e.g. `QQQ`, `^HSI`).

**Notes**

The reply will include the fitted α/β, event-day AR, several CAR
windows, plus parametric (t-test) and non-parametric (Wilcoxon)
significance.
