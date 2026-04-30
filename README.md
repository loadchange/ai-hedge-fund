# AI Hedge Fund

This is a proof of concept for an AI-powered hedge fund.  The goal of this project is to explore the use of AI to make trading decisions.  This project is for **educational** purposes only and is not intended for real trading or investment.

The fund fans out a workflow of investor "personas" and a few generic analysts onto each ticker, aggregates their signals through a risk manager, and finalises a decision in the portfolio manager. See [Agent roster](#agent-roster) for the full list and the keys used by the CLI / issue bot.

<img width="1042" alt="Screenshot 2025-03-22 at 6 19 07 PM" src="https://github.com/user-attachments/assets/cbae3dcf-b571-490d-b0ad-3f0f035ac0d4" />

Note: the system does not actually make any trades.

[![Twitter Follow](https://img.shields.io/twitter/follow/virattt?style=social)](https://twitter.com/virattt)

## Disclaimer

This project is for **educational and research purposes only**.

- Not intended for real trading or investment
- No investment advice or guarantees provided
- Creator assumes no liability for financial losses
- Consult a financial advisor for investment decisions
- Past performance does not indicate future results

By using this software, you agree to use it solely for learning purposes.

## Table of Contents
- [How to Install](#how-to-install)
- [How to Run](#how-to-run)
  - [⌨️ Command Line Interface](#️-command-line-interface)
  - [🖥️ Web Application](#️-web-application)
- [Agent roster](#agent-roster)
- [Supported Markets & Data Sources](#supported-markets--data-sources)
- [Internationalisation (i18n)](#internationalisation-i18n)
- [Network & Proxy Notes](#network--proxy-notes)
- [How to Contribute](#how-to-contribute)
- [Feature Requests](#feature-requests)
- [License](#license)

## How to Install

Before you can run the AI Hedge Fund, you'll need to install it and set up your API keys. These steps are common to both the full-stack web application and command line interface.

### 1. Clone the Repository

```bash
git clone https://github.com/virattt/ai-hedge-fund.git
cd ai-hedge-fund
```

### 2. Set up API keys

Create a `.env` file for your API keys:
```bash
# Create .env file for your API keys (in the root directory)
cp .env.example .env
```

Open and edit the `.env` file to add your API keys:
```bash
# For running LLMs hosted by openai (gpt-4o, gpt-4o-mini, etc.)
OPENAI_API_KEY=your-openai-api-key

# For getting financial data to power the hedge fund
FINANCIAL_DATASETS_API_KEY=your-financial-datasets-api-key
```

**Important**: You must set at least one LLM API key (e.g. `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`) for the hedge fund to work. 

## How to Run

### ⌨️ Command Line Interface

You can run the AI Hedge Fund directly via terminal. This approach offers more granular control and is useful for automation, scripting, and integration purposes.

<img width="992" alt="Screenshot 2025-01-06 at 5 50 17 PM" src="https://github.com/user-attachments/assets/e8ca04bf-9989-4a7d-a8b4-34e04666663b" />

#### Quick Start

This project uses [**uv**](https://docs.astral.sh/uv/) as its package and project manager.

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies (uv will create a `.venv/` and pin from `uv.lock`):
```bash
uv sync
```

#### Run the AI Hedge Fund
```bash
uv run python src/main.py --ticker AAPL,MSFT,NVDA
```

You can also specify a `--ollama` flag to run the AI hedge fund using local LLMs.

```bash
uv run python src/main.py --ticker AAPL,MSFT,NVDA --ollama
```

You can optionally specify the start and end dates to make decisions over a specific time period.

```bash
uv run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01
```

To run the system with Chinese output, add `--lang zhCN`:

```bash
uv run python src/main.py --ticker 600519.SS --lang zhCN
```

#### Run the Backtester
```bash
uv run python src/backtester.py --ticker AAPL,MSFT,NVDA
```

**Example Output:**
<img width="941" alt="Screenshot 2025-01-06 at 5 47 52 PM" src="https://github.com/user-attachments/assets/00e794ea-8628-44e6-9a84-8f8a31ad3b47" />


Note: The `--ollama`, `--start-date`, and `--end-date` flags work for the backtester, as well!

### 🖥️ Web Application

The new way to run the AI Hedge Fund is through our web application that provides a user-friendly interface. This is recommended for users who prefer visual interfaces over command line tools.

Please see detailed instructions on how to install and run the web application [here](https://github.com/virattt/ai-hedge-fund/tree/main/app).

<img width="1721" alt="Screenshot 2025-06-28 at 6 41 03 PM" src="https://github.com/user-attachments/assets/b95ab696-c9f4-416c-9ad1-51feb1f5374b" />


## Agent roster

Use the **key** column when listing analysts in a CLI invocation (`--analysts warren_buffett,duan_yongping,…`) or in an issue body for the bot. Use `all` to enable everything.

### Investor personas

| Display name | Key | Style |
|---|---|---|
| Aswath Damodaran | `aswath_damodaran` | The Dean of Valuation — story, numbers, disciplined DCF |
| Ben Graham | `ben_graham` | Godfather of value — hidden gems with margin of safety |
| Bill Ackman | `bill_ackman` | Activist — bold concentrated positions, pushes for change |
| Cathie Wood (木头姐) | `cathie_wood` | Queen of growth — disruptive innovation, large TAM |
| Charlie Munger | `charlie_munger` | Wonderful businesses at fair prices |
| Duan Yongping (段永平) | `duan_yongping` | 本分 + 不懂不投 — quality cash-flow businesses, strict stop-doing list |
| Michael Burry | `michael_burry` | Big Short contrarian — deep value hunter |
| Mohnish Pabrai | `mohnish_pabrai` | Dhandho — heads I win, tails I don't lose much |
| Nassim Taleb | `nassim_taleb` | Tail risk, antifragility, asymmetric payoffs |
| Peter Lynch | `peter_lynch` | Practical "ten-baggers" in everyday businesses |
| Phil Fisher | `phil_fisher` | Meticulous scuttlebutt research, long-term growth |
| Rakesh Jhunjhunwala | `rakesh_jhunjhunwala` | Big Bull of India — emerging-market growth |
| Stanley Druckenmiller | `stanley_druckenmiller` | Macro hunter for asymmetric setups |
| Warren Buffett | `warren_buffett` | Oracle of Omaha — wonderful companies at fair prices |

### Generic analysts

| Display name | Key | Style |
|---|---|---|
| Valuation Analyst | `valuation_analyst` | Intrinsic-value calculator (DCF, multiples) |
| Sentiment Analyst | `sentiment_analyst` | Market-sentiment indicators |
| News Sentiment Analyst | `news_sentiment_analyst` | Headline sentiment |
| Fundamentals Analyst | `fundamentals_analyst` | Pure financial-statement ratios |
| Technical Analyst | `technical_analyst` | Trend / mean reversion / momentum / volatility |
| Growth Analyst | `growth_analyst` | Growth and quality metrics |

### Decision layer (always on, not selectable)

- `risk_management_agent` — sets per-ticker position limits from volatility and correlation
- `portfolio_manager` — final `BUY` / `SELL` / `HOLD` / `SHORT` / `COVER` decision

## Supported Markets & Data Sources

The hedge fund routes data requests to multiple providers based on the **ticker format**:

| Market | Ticker examples | Detection rule |
|---|---|---|
| US | `AAPL`, `MSFT`, `NVDA` | default |
| Hong Kong | `9988.HK`, `0700.HK` | `.HK` suffix |
| China A-share (Shanghai) | `600519.SS`, `600519` | `.SS` suffix or 6-digit numeric starting with `6` |
| China A-share (Shenzhen) | `002594.SZ`, `000001` | `.SZ` suffix or 6-digit numeric starting with `0`/`3` |

Detection is implemented in [`src/data/sources/base.py`](src/data/sources/base.py) (`classify_ticker`).

### Data source layer

A `DataSourceManager` ([`src/data/sources/manager.py`](src/data/sources/manager.py)) orchestrates these providers with **priority routing**, **fallback**, **rate-limit cooldown** (5 min), and **price cross-validation** (warns when two sources differ by >2%):

| Source | Coverage | Notes |
|---|---|---|
| `financialdatasets` | US prices + financials + line items | Requires `FINANCIAL_DATASETS_API_KEY` |
| `yfinance` | US/HK prices + financial metrics | Free; rate-limited |
| `akshare` | US/HK/CN prices + financials | Free; uses Sina/Eastmoney upstream |
| `tencent` | US/HK/CN realtime quotes (market-cap, PE, PB) | Free; HK/CN financial metrics are enriched from Tencent realtime data |

For A-share line items, the system also has a **fallback path** in `tools/api.py` that derives missing line items (e.g. `shareholders_equity`, `total_liabilities`, `revenue`, `outstanding_shares`) from akshare's financial-indicator table when the financialdatasets API does not cover the ticker.

## Internationalisation (i18n)

Add `--lang zhCN` to any CLI invocation to switch the entire output (progress bars, agent names, signals, table headers, reasoning) to Simplified Chinese. The supported values are `en` (default) and `zhCN`. Implementation lives in [`src/i18n.py`](src/i18n.py):

- Status messages, table headers, action/signal enums, and the 13 investor-agent display names are translated via lookup dictionaries.
- The current language is also injected into every LLM system prompt (`get_lang_instruction`) so that model-generated `reasoning` is produced in the chosen language.
- A post-call sanitiser (`_sanitize_reasoning_fields` in `src/utils/llm.py`) walks the Pydantic response and converts any JSON-shaped `reasoning` value into natural-language text via `summarize_json_reasoning`, so the output table never shows raw JSON.

## Network & Proxy Notes

- **Proxy bypass**: domestic data endpoints (Sina, Tencent at `web.ifzq.gtimg.cn`) often break when a proxy is in front of them. The akshare adapter temporarily clears `HTTP_PROXY`/`HTTPS_PROXY` for these hosts via `_no_proxy_for(PROXY_BYPASS_DOMAINS)` in [`src/data/sources/akshare_src.py`](src/data/sources/akshare_src.py). Add hosts to `PROXY_BYPASS_DOMAINS` if you encounter additional ones.
- **SSL retry**: when the financialdatasets/HTTP request fails with an SSL error through a proxy, `tools/api.py:_make_api_request` retries once with the proxy disabled.
- **Anthropic via gateway**: set `ANTHROPIC_BASE_URL` in `.env` to route Claude calls through a proxy/gateway. Extended thinking is disabled by default in `src/llm/models.py` so JSON-mode parsing stays reliable.
- **OpenAI-compatible gateways** (e.g. xiaomi MiMo, OpenRouter): set `OPENAI_API_BASE` alongside `OPENAI_API_KEY`. Note that all `provider: "OpenAI"` entries in `src/llm/api_models.json` share the same env vars, so picking a model-only gateway will affect the GPT entries too.

## How to Contribute

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

**Important**: Please keep your pull requests small and focused.  This will make it easier to review and merge.

## Feature Requests

If you have a feature request, please open an [issue](https://github.com/virattt/ai-hedge-fund/issues) and make sure it is tagged with `enhancement`.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
