# AI Hedge Fund

An educational, multi-agent trading system: thirteen LLM "persona"
investors (Buffett, Munger, Cathie Wood, Duan Yongping, Druckenmiller,
…) plus a quantitative stack decide BUY / SELL / HOLD / SHORT per
ticker.

Markets: **US, Hong Kong, China A-share** (Shanghai + Shenzhen).
Output: **English / Simplified Chinese**.

> **Disclaimer.** Educational / research only. No real trades, no
> investment advice, past performance ≠ future results.

## Architecture

```
   CLI · Issue bot · Daily Report (cron)
        │
   DataSourceManager  (US: yfinance→akshare · HK: tencent/yfinance/akshare
        │             · CN: baostock/akshare/tencent)
        ├──────────────────────────────┐
        ▼                              ▼
   LLM persona agents            Quant signals (BaseSignal)
   (LangGraph; Buffett /         trend · mean_reversion · momentum
   Munger / Wood / …)            volatility · stat_arb · value
        │                        quality · earnings_surprise
        └────────────┬───────────┘
                     ▼
          Market Review (rule-based)
          SPY · QQQ · 000300.SS · ^HSI
                     ▼
          Risk manager → Portfolio manager
          (vol / corr / drawdown caps; optional
          cvxpy MVO · risk parity · Black-Litt.)
                     ▼
          BUY / SELL / HOLD / SHORT  (10 bps default cost)
                     ▼
          Dashboard Report → Notification (Lark/Telegram/Discord/Slack/…)
                     ▼
          Backtester · Validation (CPCV + PBO + Deflated Sharpe)
```

## Table of contents

- [Install](#install) · [Run](#run) · [Issue bot](#issue-bot) ·
  [Daily market report](#daily-market-report) ·
  [Agent roster](#agent-roster) · [Quantitative modules](#quantitative-modules)
- [Markets & data sources](#markets--data-sources) ·
  [Strategies](#strategies) ·
  [Notifications](#notifications) ·
  [Internationalisation](#internationalisation) ·
  [Examples](#examples)
- [Acknowledgements](#acknowledgements) · [License](#license)

## Install

```bash
git clone https://github.com/loadchange/ai-hedge-fund.git
cd ai-hedge-fund
curl -LsSf https://astral.sh/uv/install.sh | sh   # if you don't have uv
uv sync
cp .env.example .env                              # set ONE LLM key
```

Market data is **all free** (yfinance / akshare / baostock / tencent) —
no paid provider required.

## Run

```bash
# 1. Hedge fund — single-day multi-agent decision
uv run python src/main.py --tickers AAPL,MSFT --model deepseek-v4-flash \
  --analysts warren_buffett,duan_yongping --lang zhCN  # or --ollama

# 2. Backtester — re-runs the workflow per business day
uv run python src/backtester.py --tickers AAPL --model deepseek-v4-flash \
  --start-date 2025-01-01 --end-date 2025-02-01 \
  --analysts warren_buffett,duan_yongping
# --cost-model {fixed,spread}  --cost-bps N  (default fixed 10 bps)

# 3. Validation CLI — CPCV + PBO + Deflated Sharpe (no LLM)
uv run python -m src.validation.cli evaluate \
  --signal momentum,trend --ticker AAPL,MSFT \
  --start 2023-01-01 --end 2025-04-01 --rolling-window 180
```

> ⚠️ Backtester cost scales as `analysts × tickers × days`. Prefer
> 1 ticker × 2–3 analysts × 1–2 weeks for experimentation.

| Command | LLM? |
|---|---|
| `src/main.py` | yes — one call per persona × ticker + portfolio mgr |
| `src/main.py --analysts market_review,technical_analyst` | no — rule-based only |
| `src/backtester.py` | yes per business day |
| `python -m src.validation.cli evaluate` | no — pure CPCV/PBO |
| `from src.{signals,risk,portfolio,validation,event_study,features}` | no — quant modules are LLM-free |

## Issue bot

The [`Hedge Fund Issue Bot`](.github/workflows/issue-bot.yml) workflow
turns issues into runnable jobs: open from a template → bot replies
with Markdown → issue auto-closes. Subscribers get email notifications
via GitHub's native flow.

> **Restricted to repo owner only.** Non-owners receive a polite reply
> pointing them to fork the repo and self-deploy.

| Mode | Label | LLM | Output |
|---|---|---|---|
| Ticker analysis | `bot-ticker` | yes | Single-day BUY/SELL/HOLD/SHORT per ticker |
| Backtester | `bot-backtester` | yes/day | Multi-day equity curve + Sharpe + costs |
| Signal validation | `bot-validate` | no | CPCV IS/OOS Sharpe + PBO + DSR |
| Event study | `bot-event-study` | no | Market-model α/β + AR/CAR + t-stat |

**Required repo secrets**: `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`
(must exist in [`src/llm/api_models.json`](src/llm/api_models.json)).

## Daily market report

The [`Daily Market Report`](.github/workflows/daily-report.yml) workflow
runs automatically on each trading day, creates a GitHub issue, runs
index analysis for CN/HK/US markets, and pushes a notification.

| Market | Trigger time | Indices |
|---|---|---|
| China A-share | 3:45 PM Beijing | 000300.SS, 000001.SS, ^HSI, SPY, QQQ |
| Hong Kong | 4:47 PM Hong Kong | ^HSI, 000300.SS, SPY, QQQ |
| US | ~4:53 PM ET | SPY, QQQ, DIA, IWM, 000300.SS, ^HSI |

The workflow uses `market_review` + `technical_analyst` (both rule-based,
no LLM cost) and sends results via the configured notification channel
(Apprise).

**Required repo secrets**: `NOTIFY_URLS` (Apprise URL, e.g.
`lark://webhook_token`).

## Agent roster

CLI: `--analysts warren_buffett,duan_yongping,…` or `all`. Issue body:
free text — the LLM extracts.

**Investor personas** (LLM): `aswath_damodaran` · `ben_graham` ·
`bill_ackman` · `cathie_wood` · `charlie_munger` · `duan_yongping` ·
`michael_burry` · `mohnish_pabrai` · `nassim_taleb` · `peter_lynch` ·
`phil_fisher` · `rakesh_jhunjhunwala` · `stanley_druckenmiller` ·
`warren_buffett`.

**Generic analysts**: `valuation_analyst` · `sentiment_analyst` ·
`news_sentiment_analyst` · `fundamentals_analyst` · `growth_analyst` ·
`technical_analyst` (delegates to `src/signals/`, **no LLM**).

**Market overview**: `market_review` (rule-based, **no LLM** — analyzes
SPY, QQQ, 000300.SS, ^HSI for macro context).

**Decision layer** (always on): `risk_management_agent` (no LLM, sets
position limits from vol/correlation), `portfolio_manager` (LLM, final
BUY/SELL/HOLD/SHORT/COVER).

## Quantitative modules

Six standalone packages, importable without LangGraph:

| Module | Purpose |
|---|---|
| [`src/signals/`](src/signals/) | `BaseSignal` ABC + 8 signals (5 technical / 3 fundamental); `SignalResult` ∈ `[-1, +1]` |
| [`src/risk/`](src/risk/) | Vol / correlation, drawdown, scenario stress (2008/2020/2022/2025), Kelly + vol-targeted sizing |
| [`src/portfolio/`](src/portfolio/) | cvxpy optimizers (MVO / risk parity / Black-Litterman), Ledoit-Wolf shrinkage, MP eigenvalue cleaning |
| [`src/validation/`](src/validation/) | CPCV + PBO + Deflated Sharpe + CLI |
| [`src/event_study/`](src/event_study/) | Market-model α/β fit, AR / CAR / CAAR, t-test + Wilcoxon |
| [`src/features/`](src/features/) | SUE, PEAD drift, KPI YoY z-scores, lead-lag, Granger causality |

## Markets & data sources

[`classify_ticker`](src/data/sources/base.py) routes by suffix: US
(default), HK (`.HK`), CN Shanghai (`.SS` / `6xxxxx`), CN Shenzhen
(`.SZ` / `0xxxxx` / `3xxxxx`).

| Source | Coverage |
|---|---|
| `yfinance` | US / HK / CN prices + financials + news + line items + earnings dates |
| `akshare` | US / HK / CN prices + financials (Sina / Eastmoney upstream) |
| `baostock` | CN adjusted OHLCV + structured quarterly fundamentals |
| `tencent` | CN / HK realtime quotes (market-cap, PE, PB) |

## Strategies

Six built-in strategy presets in [`src/strategies/defaults/`](src/strategies/defaults/):

| Strategy | Key | Focus |
|---|---|---|
| Momentum / Trend | `momentum_trend` | Technical trend-following (default) |
| Deep Value | `value_deep` | Buffett / Graham / Fisher, margin of safety |
| Contrarian | `contrarian` | Burry / Taleb, mean reversion |
| Growth / Disruption | `growth_disrupt` | Cathie Wood, momentum-heavy |
| Macro-Aware | `macro_aware` | Market review + balanced |
| A-Share (CN) | `cn_a_share` | Duan Yongping, baostock/akshare optimized |

```bash
uv run python src/main.py --tickers 600519.SS --strategy cn_a_share --lang zhCN
```

## Notifications

Multi-channel notifications via [Apprise](https://github.com/caronc/apprise).
Set `NOTIFY_URLS` in `.env` (one URL per line):

```bash
NOTIFY_URLS=lark://webhook_token
# NOTIFY_URLS=feishu://webhook_token
# NOTIFY_URLS=tgram://bottoken/chat_id
# NOTIFY_URLS=discord://webhook_id/webhook_token
# NOTIFY_URLS=slack://tokenA/tokenB/tokenC
# NOTIFY_URLS=mailto://user:pass@smtp.example.com/to@example.com
```

```bash
uv run python src/main.py --tickers AAPL --notify   # send after analysis
```

## Internationalisation

`--lang zhCN` switches every CLI / bot output to Simplified Chinese.
Implemented in [`src/i18n.py`](src/i18n.py): translates table headers /
signals / agent names, injects a language instruction into every LLM
prompt, and post-sanitises model output so JSON blobs become natural
language.

## Examples

Five LLM-free demos in [`examples/`](examples/):

```bash
uv run bash   examples/01_signal_validation.sh    # CPCV/PBO via CLI
uv run python examples/02_signals_to_portfolio.py # signals → BL → MVO
uv run python examples/03_risk_analytics.py       # drawdown + stress
uv run python examples/04_event_study.py          # market model + CAR
uv run python examples/05_features.py             # SUE + KPI + Granger
```

## Acknowledgements

Built on the original [virattt/ai-hedge-fund](https://github.com/virattt/ai-hedge-fund) —
the multi-agent persona idea is theirs, credit to the original author.

That said, this codebase (and the upstream) is an **AI vibe-coded
toy**. It can be a fun lens for thinking about a stock, but it has not
been validated against real capital, leans on free APIs with
incomplete fundamentals, and runs non-deterministic LLMs whose
outputs shift with prompt phrasing. Treat every output as a
conversation starter, not a recommendation. **Do your own research
(DYOR).** Past performance ≠ future. Risk only what you can afford to
lose entirely.

## License

MIT — see [LICENSE](LICENSE).
