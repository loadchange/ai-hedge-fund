import logging
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.sources import get_data_source_manager
from src.data.sources.base import classify_ticker, get_proxy_dict
from src.data.models import (
    CompanyNews,
    FinancialMetrics,
    Price,
    LineItem,
    InsiderTrade,
)

# Global cache instance
_cache = get_cache()


def _make_api_request(url: str, headers: dict, method: str = "GET", json_data: dict = None, max_retries: int = 3) -> requests.Response | None:
    """Make an HTTP request with retry on 429, fallback on SSL through proxy.

    Returns the Response on success (any status code), or None when the
    network itself is unreachable (DNS failure, connection refused, timeout,
    etc.). Callers must treat None as "skip this data source" — agents
    should never crash the whole run because a single optional API is down.
    """
    proxies = get_proxy_dict()

    def _do_request(use_proxies):
        if method.upper() == "POST":
            return requests.post(url, headers=headers, json=json_data, proxies=use_proxies)
        return requests.get(url, headers=headers, proxies=use_proxies)

    for attempt in range(max_retries + 1):  # +1 for initial attempt
        try:
            response = _do_request(proxies)
        except requests.exceptions.SSLError as e:
            # SSL error through proxy — retry once without proxy.
            logger.debug("SSL error via proxy for %s, retrying without proxy: %s", url, e)
            try:
                response = _do_request(None)
            except requests.exceptions.RequestException as e2:
                logger.debug("HTTP request to %s failed after SSL fallback: %s", url, e2)
                return None
        except requests.exceptions.RequestException as e:
            # DNS failure / connection refused / timeout / other transport issues.
            # Don't crash the agent graph — return None and let the caller move on.
            logger.debug("HTTP request to %s failed: %s", url, e)
            return None

        if response.status_code == 429 and attempt < max_retries:
            # Linear backoff: 60s, 90s, 120s, 150s...
            delay = 60 + (30 * attempt)
            print(f"Rate limited (429). Attempt {attempt + 1}/{max_retries + 1}. Waiting {delay}s before retrying...")
            time.sleep(delay)
            continue

        return response

    return None


def get_prices(ticker: str, start_date: str, end_date: str) -> list[Price]:
    """Fetch price data from cache or multi-source manager."""
    cache_key = f"{ticker}_{start_date}_{end_date}"

    if cached_data := _cache.get_prices(cache_key):
        return [Price(**price) for price in cached_data]

    manager = get_data_source_manager()
    prices = manager.get_prices(ticker, start_date, end_date)

    if prices:
        _cache.set_prices(cache_key, [p.model_dump() for p in prices])
    return prices


def get_financial_metrics(
    ticker: str,
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[FinancialMetrics]:
    """Fetch financial metrics from cache or multi-source manager."""
    cache_key = f"{ticker}_{period}_{end_date}_{limit}"

    if cached_data := _cache.get_financial_metrics(cache_key):
        return [FinancialMetrics(**metric) for metric in cached_data]

    manager = get_data_source_manager()
    metrics = manager.get_financial_metrics(ticker, end_date, period, limit)

    if metrics:
        _cache.set_financial_metrics(cache_key, [m.model_dump() for m in metrics])
    return metrics


def _cn_line_items_fallback(
    ticker: str, end_date: str, limit: int = 10
) -> list[LineItem]:
    """Construct LineItem objects from akshare CN financial data."""
    if classify_ticker(ticker) != "cn":
        return []

    try:
        import akshare as ak
        from src.data.sources.akshare_src import (
            _AKSHARE_LOCK,
            _pct_to_decimal,
            _safe_float,
            _suppress_tqdm,
        )
        from src.data.sources.base import normalize_ticker

        code = ticker.replace(".SS", "").replace(".SZ", "").replace(".ss", "").replace(".sz", "")
        if not code.isdigit():
            code = normalize_ticker(ticker, "akshare").lstrip("shsz")

        # Lock first so the tqdm-suppression patch is applied serially across threads.
        with _AKSHARE_LOCK, _suppress_tqdm():
            df = ak.stock_financial_analysis_indicator(symbol=code, start_year="2020")
        if df.empty:
            return []

        import pandas as pd
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        end_dt = pd.to_datetime(end_date)
        df = df[df["日期"] <= end_dt].sort_values("日期", ascending=False).head(limit)

        items = []
        for _, row in df.iterrows():
            report_period = row["日期"].strftime("%Y-%m-%d") if pd.notna(row.get("日期")) else end_date
            eps = _safe_float(row.get("每股收益_调整后(元)"))
            bps = _safe_float(row.get("每股净资产_调整前(元)"))
            roe_pct = _safe_float(row.get("净资产收益率(%)"))
            total_assets = _safe_float(row.get("总资产(元)"))
            gross_profit = _safe_float(row.get("主营业务利润(元)"))
            net_income = _safe_float(row.get("扣除非经常性损益后的净利润(元)"))
            ocf_per_share = _safe_float(row.get("每股经营性现金流(元)"))
            debt_ratio = _pct_to_decimal(row.get("资产负债率(%)"))
            current_ratio = _safe_float(row.get("流动比率"))

            # Derive shareholders_equity from ROE and net_income
            shareholders_equity = None
            if roe_pct and net_income and roe_pct != 0:
                shareholders_equity = net_income / (roe_pct / 100.0)

            # Derive total_liabilities from debt_ratio and total_assets
            total_liabilities = None
            if debt_ratio and total_assets:
                total_liabilities = total_assets * debt_ratio

            # Derive outstanding_shares from EPS and net_income
            outstanding_shares = None
            if eps and net_income and eps != 0:
                outstanding_shares = abs(net_income / eps)

            # Derive revenue from net_income and net_margin
            net_margin = _pct_to_decimal(row.get("销售净利率(%)"))
            revenue = None
            if net_income and net_margin and net_margin != 0:
                revenue = abs(net_income / net_margin)

            # LineItem declares every numeric field with a None default,
            # so we only have to pass the values we have.
            items.append(
                LineItem(
                    ticker=ticker,
                    report_period=report_period,
                    period="ttm",
                    currency="CNY",
                    gross_profit=gross_profit,
                    net_income=net_income,
                    total_assets=total_assets,
                    total_liabilities=total_liabilities,
                    shareholders_equity=shareholders_equity,
                    outstanding_shares=outstanding_shares,
                    revenue=revenue,
                    earnings_per_share=eps,
                    book_value_per_share=bps,
                    operating_cash_flow_per_share=ocf_per_share,
                    source="akshare",
                )
            )
        return items
    except Exception as e:
        logger.debug("CN line items fallback error for %s: %s", ticker, e)
        return []


# yfinance row-name → our snake_case field mapping. Each value is
# (df_kind, row_label) where df_kind ∈ {"cashflow","balance_sheet","income"}.
_YF_LINE_ITEM_MAP: dict[str, tuple[str, str]] = {
    # Cashflow rows
    "capital_expenditure": ("cashflow", "Capital Expenditure"),
    "free_cash_flow": ("cashflow", "Free Cash Flow"),
    "dividends_and_other_cash_distributions": ("cashflow", "Cash Dividends Paid"),
    "issuance_or_purchase_of_equity_shares": ("cashflow", "Net Common Stock Issuance"),
    "operating_cash_flow": ("cashflow", "Operating Cash Flow"),
    # Balance-sheet rows
    "total_assets": ("balance_sheet", "Total Assets"),
    "total_liabilities": ("balance_sheet", "Total Liabilities Net Minority Interest"),
    "shareholders_equity": ("balance_sheet", "Stockholders Equity"),
    "outstanding_shares": ("balance_sheet", "Ordinary Shares Number"),
    "current_assets": ("balance_sheet", "Current Assets"),
    "current_liabilities": ("balance_sheet", "Current Liabilities"),
    "total_debt": ("balance_sheet", "Total Debt"),
    "cash_and_equivalents": ("balance_sheet", "Cash And Cash Equivalents"),
    "working_capital": ("balance_sheet", "Working Capital"),
    "goodwill_and_intangible_assets": ("balance_sheet", "Goodwill And Other Intangible Assets"),
    "intangible_assets": ("balance_sheet", "Other Intangible Assets"),
    # Income-statement rows
    "revenue": ("income", "Total Revenue"),
    "gross_profit": ("income", "Gross Profit"),
    "operating_income": ("income", "Operating Income"),
    "operating_expense": ("income", "Operating Expense"),
    "net_income": ("income", "Net Income"),
    "ebit": ("income", "EBIT"),
    "ebitda": ("income", "EBITDA"),
    "research_and_development": ("income", "Research And Development"),
    "depreciation_and_amortization": ("income", "Reconciled Depreciation"),
    "earnings_per_share": ("income", "Diluted EPS"),
}


def _us_line_items_fallback(
    ticker: str, end_date: str, period: str = "ttm", limit: int = 10
) -> list[LineItem]:
    """Construct LineItem objects from yfinance for US/HK tickers.

    Pulls quarterly (period="ttm") or annual cashflow / balance-sheet /
    income-statement DataFrames, picks columns ≤ end_date, and projects
    each fiscal date into a :class:`LineItem` using ``_YF_LINE_ITEM_MAP``.
    Margin / per-share fields are derived from the raw rows.
    """
    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        t = yf.Ticker(ticker)
        is_quarterly = period == "ttm"
        cashflow = t.quarterly_cashflow if is_quarterly else t.cashflow
        balance = t.quarterly_balance_sheet if is_quarterly else t.balance_sheet
        income = t.quarterly_financials if is_quarterly else t.financials
    except Exception as e:
        logger.debug("yfinance line items error for %s: %s", ticker, e)
        return []

    dfs = {"cashflow": cashflow, "balance_sheet": balance, "income": income}
    all_cols: set = set()
    for df in dfs.values():
        if df is not None and not df.empty:
            all_cols.update(df.columns)
    if not all_cols:
        return []

    end_ts = pd.to_datetime(end_date)
    valid_cols = sorted([c for c in all_cols if c <= end_ts], reverse=True)[:limit]

    def _safe_get(df_kind: str, row: str, col) -> float | None:
        df = dfs.get(df_kind)
        if df is None or df.empty or row not in df.index or col not in df.columns:
            return None
        try:
            v = df.at[row, col]
            if pd.isna(v):
                return None
            return float(v)
        except (KeyError, TypeError, ValueError):
            return None

    items: list[LineItem] = []
    for col in valid_cols:
        report_period = col.strftime("%Y-%m-%d") if hasattr(col, "strftime") else str(col)[:10]
        fields: dict = {
            "ticker": ticker,
            "report_period": report_period,
            "period": period,
            "currency": "USD",
            "source": "yfinance",
        }
        for our_field, (df_kind, yf_row) in _YF_LINE_ITEM_MAP.items():
            v = _safe_get(df_kind, yf_row, col)
            if v is not None:
                fields[our_field] = v

        # Derived ratios
        rev = fields.get("revenue")
        if rev:
            if (gp := fields.get("gross_profit")) is not None:
                fields["gross_margin"] = gp / rev
            if (oi := fields.get("operating_income")) is not None:
                fields["operating_margin"] = oi / rev
        se = fields.get("shareholders_equity")
        td = fields.get("total_debt")
        if se and td is not None:
            fields["debt_to_equity"] = td / se
        sh = fields.get("outstanding_shares")
        if se is not None and sh:
            fields["book_value_per_share"] = se / sh
        ocf = fields.get("operating_cash_flow")
        if ocf is not None and sh:
            fields["operating_cash_flow_per_share"] = ocf / sh
        ni = fields.get("net_income")
        invested = (td or 0) + (se or 0)
        if ni is not None and invested:
            fields["return_on_invested_capital"] = ni / invested

        items.append(LineItem(**fields))

    return items


def search_line_items(
    ticker: str,
    line_items: list[str],  # noqa: ARG001 — kept for API compatibility; yfinance returns full schema
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
) -> list[LineItem]:
    """Fetch line items from free data sources.

    CN tickers go through the akshare-derived fallback; US/HK tickers go
    through a yfinance-derived fallback. ``line_items`` is accepted for
    backwards compatibility but ignored — agents read whatever fields
    are present on the returned :class:`LineItem`.
    """
    market = classify_ticker(ticker)
    if market == "cn":
        return _cn_line_items_fallback(ticker, end_date, limit)
    return _us_line_items_fallback(ticker, end_date, period, limit)


def get_insider_trades(
    ticker: str,  # noqa: ARG001
    end_date: str,  # noqa: ARG001
    start_date: str | None = None,  # noqa: ARG001
    limit: int = 1000,  # noqa: ARG001
) -> list[InsiderTrade]:
    """Insider-trade data is not available from any free source we ship.

    Form 4 parsing from SEC EDGAR would close the gap for US tickers but
    is out of scope for this build. Returns an empty list; agents that
    weight insider activity (sentiment, peter_lynch, michael_burry,
    stanley_druckenmiller) already treat empty input as "no signal" and
    fall back to their other inputs.
    """
    return []


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
) -> list[CompanyNews]:
    """Fetch company news from yfinance (US/HK). CN: returns empty.

    yfinance's ``Ticker.news`` returns ~10–30 recent articles per ticker
    with no historical pagination, so ``start_date`` / ``end_date`` /
    ``limit`` filter the available headlines client-side rather than
    paging an API.
    """
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    market = classify_ticker(ticker)
    if market == "cn":
        return []

    try:
        import yfinance as yf
    except ImportError:
        return []

    try:
        raw = yf.Ticker(ticker).news or []
    except Exception as e:
        logger.debug("yfinance news error for %s: %s", ticker, e)
        return []

    # yfinance returns either the legacy flat dict or {id, content: {...}}.
    # Normalise both shapes into a CompanyNews.
    end_ts = pd.to_datetime(end_date)
    start_ts = pd.to_datetime(start_date) if start_date else None

    items: list[CompanyNews] = []
    for art in raw:
        content = art.get("content") if isinstance(art, dict) else None
        body = content if isinstance(content, dict) else art
        title = body.get("title") if isinstance(body, dict) else None
        if not title:
            continue
        publisher = body.get("publisher") or (body.get("provider") or {}).get("displayName") or "yfinance"
        # yfinance gives publish time as either a unix ts or an ISO string.
        pub_raw = body.get("pubDate") or body.get("providerPublishTime")
        try:
            pub_ts = pd.to_datetime(pub_raw, unit="s") if isinstance(pub_raw, (int, float)) else pd.to_datetime(pub_raw)
        except Exception:
            continue
        if pd.isna(pub_ts):
            continue
        # Strip tz so we can compare against the tz-naive start/end dates.
        try:
            pub_ts = pub_ts.tz_localize(None) if pub_ts.tz is not None else pub_ts
        except (AttributeError, TypeError):
            pass
        if pub_ts > end_ts:
            continue
        if start_ts is not None and pub_ts < start_ts:
            continue

        url = body.get("canonicalUrl", {}).get("url") if isinstance(body.get("canonicalUrl"), dict) else body.get("link")
        items.append(
            CompanyNews(
                ticker=ticker,
                title=title,
                author=body.get("author"),
                source=str(publisher),
                date=pub_ts.strftime("%Y-%m-%d"),
                url=url or "",
                sentiment=None,
            )
        )
        if len(items) >= limit:
            break

    if items:
        _cache.set_company_news(cache_key, [n.model_dump() for n in items])
    return items


def get_market_cap(
    ticker: str,
    end_date: str,  # noqa: ARG001 — kept for API compatibility, sources don't use it
) -> float | None:
    """Fetch market cap from real-time / fundamentals sources only.

    CN/HK: Tencent realtime quote is the most current source. Falls
    back to financial-metrics for any market when realtime is missing.
    """
    market = classify_ticker(ticker)

    if market in ("cn", "hk"):
        from src.data.sources.tencent_src import TencentSource
        quote = TencentSource().get_realtime_quote(ticker)
        if quote and quote.get("market_cap"):
            return quote["market_cap"]

    financial_metrics = get_financial_metrics(ticker, end_date)
    if financial_metrics and financial_metrics[0].market_cap:
        return financial_metrics[0].market_cap

    return None


def prices_to_df(prices: list[Price]) -> pd.DataFrame:
    """Convert prices to a DataFrame."""
    df = pd.DataFrame([p.model_dump() for p in prices])
    df["Date"] = pd.to_datetime(df["time"])
    df.set_index("Date", inplace=True)
    numeric_cols = ["open", "close", "high", "low", "volume"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.sort_index(inplace=True)
    return df


def get_price_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date)
    return prices_to_df(prices)
