# AI Hedge Fund

An educational, multi-agent trading system: thirteen LLM "persona"
investors (Buffett, Munger, Cathie Wood, Duan Yongping, Druckenmiller,
…) plus a quantitative stack decide BUY / SELL / HOLD / SHORT per
ticker.

Markets: **US, Hong Kong, China A-share** (Shanghai + Shenzhen).
Output: **English / Simplified Chinese**.

<a href="https://platform.xiaomimimo.com/token-plan" target="_blank" rel="noopener">
  <img width="320" src="https://github.com/user-attachments/assets/3a264c9b-48a0-44d2-935c-e3237181668a" alt="Xiaomi MiMo v2.5 Pro" />
</a>

**Powered by [Xiaomi MiMo v2.5 Pro](https://platform.xiaomimimo.com/token-plan).**
New users get **$2** free credit with invite code **`FU5PSQ`**.

> **Disclaimer.** Educational / research only. No real trades, no
> investment advice, past performance ≠ future results.

## Architecture

```
   CLI · Issue bot
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
          Risk manager → Portfolio manager
          (vol / corr / drawdown caps; optional
          cvxpy MVO · risk parity · Black-Litt.)
                     ▼
          BUY / SELL / HOLD / SHORT  (10 bps default cost)
                     ▼
          Backtester · Validation (CPCV + PBO + Deflated Sharpe)
```

## Table of contents

- [Install](#install) · [Run](#run) · [Issue bot](#issue-bot) ·
  [Agent roster](#agent-roster) · [Quantitative modules](#quantitative-modules)
- [Markets & data sources](#markets--data-sources) ·
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
uv run python src/main.py --tickers AAPL,MSFT --model mimo-v2.5-pro \
  --analysts warren_buffett,duan_yongping --lang zhCN  # or --ollama

# 2. Backtester — re-runs the workflow per business day
uv run python src/backtester.py --tickers AAPL --model mimo-v2.5-pro \
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
| `src/backtester.py` | yes per business day |
| `python -m src.validation.cli evaluate` | no — pure CPCV/PBO |
| `from src.{signals,risk,portfolio,validation,event_study,features}` | no — quant modules are LLM-free |

## Issue bot

The [`Hedge Fund Issue Bot`](.github/workflows/issue-bot.yml) workflow
turns issues into runnable jobs: open from a template → bot replies
with Markdown → issue auto-closes. Subscribers get email notifications
via GitHub's native flow.

| Mode | Label | LLM | Output |
|---|---|---|---|
| Ticker analysis | `bot-ticker` | yes | Single-day BUY/SELL/HOLD/SHORT per ticker |
| Backtester | `bot-backtester` | yes/day | Multi-day equity curve + Sharpe + costs |
| Signal validation | `bot-validate` | no | CPCV IS/OOS Sharpe + PBO + DSR |
| Event study | `bot-event-study` | no | Market-model α/β + AR/CAR + t-stat |

**Lifecycle**: pick template → fill body free-form (LLM extracts args)
→ submit → ack within seconds → final reply 30 s – 5 min → auto-close.
Don't like it? Edit the body to retrigger.

**Failure replies are bilingual + actionable**: missing fields get an
example body; `ticker` / `backtester` over the **400-LLM-call cap**
(matched to the 60-min workflow timeout at ~9 s/call) get a full
breakdown plus a parameter-combo table that *would* fit; fundamental
signals on `bot-validate` (CPCV is daily-rolling) are redirected to
the five technical signals.

**Required repo secrets**: `AI_BASE_URL`, `AI_API_KEY`, `AI_MODEL`
(must exist in [`src/llm/api_models.json`](src/llm/api_models.json)).

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

Persona agents and quant modules **work side-by-side**: the same
`analyst_signals` payload feeds Black-Litterman views or
`src.validation` for OOS-robustness checks.

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

[`DataSourceManager`](src/data/sources/manager.py) does priority
routing, fallback, 5-minute rate-limit cooldowns, and price
cross-validation (warns on >2% disagreement).
[`src/tools/api.py`](src/tools/api.py) is the unified surface
(`get_prices`, `get_financial_metrics`, `search_line_items`,
`get_company_news`, `get_market_cap`, `get_insider_trades`).

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
