import datetime
import logging
import os
import pandas as pd
import requests
import time

logger = logging.getLogger(__name__)

from src.data.cache import get_cache
from src.data.sources import get_data_source_manager
from src.data.sources.base import get_proxy_dict
from src.data.models import (
    CompanyNews,
    CompanyNewsResponse,
    FinancialMetrics,
    FinancialMetricsResponse,
    Price,
    PriceResponse,
    LineItem,
    LineItemResponse,
    InsiderTrade,
    InsiderTradeResponse,
    CompanyFactsResponse,
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


def get_prices(ticker: str, start_date: str, end_date: str, api_key: str = None) -> list[Price]:
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
    api_key: str = None,
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
    """Construct LineItem objects from akshare CN financial data as a fallback."""
    from src.data.sources.base import classify_ticker
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

            # Build field dict — set all agent-expected fields to None by default
            fields: dict = {
                "ticker": ticker,
                "report_period": report_period,
                "period": "ttm",
                "currency": "CNY",
                # Fields agents access (set to None if unavailable)
                "capital_expenditure": None,
                "depreciation_and_amortization": None,
                "dividends_and_other_cash_distributions": None,
                "issuance_or_purchase_of_equity_shares": None,
                "free_cash_flow": None,
                "cash_and_equivalents": None,
                "current_assets": None,
                "current_liabilities": None,
                "total_debt": None,
                "goodwill_and_intangible_assets": None,
                "intangible_assets": None,
                "operating_expense": None,
                "operating_income": None,
                "research_and_development": None,
                "debt_to_equity": None,
                "gross_margin": None,
                "operating_margin": None,
                "return_on_invested_capital": None,
                "ebit": None,
                "ebitda": None,
                "working_capital": None,
                "book_value_per_share": None,
                "earnings_per_share": None,
                "operating_cash_flow_per_share": None,
            }
            # Override with available data
            if gross_profit is not None:
                fields["gross_profit"] = gross_profit
            if net_income is not None:
                fields["net_income"] = net_income
            if total_assets is not None:
                fields["total_assets"] = total_assets
            if total_liabilities is not None:
                fields["total_liabilities"] = total_liabilities
            if shareholders_equity is not None:
                fields["shareholders_equity"] = shareholders_equity
            if outstanding_shares is not None:
                fields["outstanding_shares"] = outstanding_shares
            if revenue is not None:
                fields["revenue"] = revenue
            if eps is not None:
                fields["earnings_per_share"] = eps
            if bps is not None:
                fields["book_value_per_share"] = bps
            if ocf_per_share is not None:
                fields["operating_cash_flow_per_share"] = ocf_per_share
            item = LineItem(**fields)

            items.append(item)
        return items
    except Exception as e:
        logger.debug("CN line items fallback error for %s: %s", ticker, e)
        return []


def search_line_items(
    ticker: str,
    line_items: list[str],
    end_date: str,
    period: str = "ttm",
    limit: int = 10,
    api_key: str = None,
) -> list[LineItem]:
    """Fetch line items from API, with CN fallback via akshare."""
    # If not in cache or insufficient data, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    url = "https://api.financialdatasets.ai/financials/search/line-items"

    body = {
        "tickers": [ticker],
        "line_items": line_items,
        "end_date": end_date,
        "period": period,
        "limit": limit,
    }
    response = _make_api_request(url, headers, method="POST", json_data=body)
    if response is None or response.status_code != 200:
        # Fallback for CN stocks
        return _cn_line_items_fallback(ticker, end_date, limit)

    try:
        data = response.json()
        response_model = LineItemResponse(**data)
        search_results = response_model.search_results
    except Exception as e:
        logger.warning("Failed to parse line items response for %s: %s", ticker, e)
        return _cn_line_items_fallback(ticker, end_date, limit)
    if not search_results:
        return _cn_line_items_fallback(ticker, end_date, limit)

    # Cache the results
    return search_results[:limit]


def get_insider_trades(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[InsiderTrade]:
    """Fetch insider trades from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_insider_trades(cache_key):
        return [InsiderTrade(**trade) for trade in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_trades = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/insider-trades/?ticker={ticker}&filing_date_lte={current_end_date}"
        if start_date:
            url += f"&filing_date_gte={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response is None or response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = InsiderTradeResponse(**data)
            insider_trades = response_model.insider_trades
        except Exception as e:
            logger.warning("Failed to parse insider trades response for %s: %s", ticker, e)
            break

        if not insider_trades:
            break

        all_trades.extend(insider_trades)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(insider_trades) < limit:
            break

        # Update end_date to the oldest filing date from current batch for next iteration
        current_end_date = min(trade.filing_date for trade in insider_trades).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_trades:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_insider_trades(cache_key, [trade.model_dump() for trade in all_trades])
    return all_trades


def get_company_news(
    ticker: str,
    end_date: str,
    start_date: str | None = None,
    limit: int = 1000,
    api_key: str = None,
) -> list[CompanyNews]:
    """Fetch company news from cache or API."""
    # Create a cache key that includes all parameters to ensure exact matches
    cache_key = f"{ticker}_{start_date or 'none'}_{end_date}_{limit}"
    
    # Check cache first - simple exact match
    if cached_data := _cache.get_company_news(cache_key):
        return [CompanyNews(**news) for news in cached_data]

    # If not in cache, fetch from API
    headers = {}
    financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
    if financial_api_key:
        headers["X-API-KEY"] = financial_api_key

    all_news = []
    current_end_date = end_date

    while True:
        url = f"https://api.financialdatasets.ai/news/?ticker={ticker}&end_date={current_end_date}"
        if start_date:
            url += f"&start_date={start_date}"
        url += f"&limit={limit}"

        response = _make_api_request(url, headers)
        if response is None or response.status_code != 200:
            break

        try:
            data = response.json()
            response_model = CompanyNewsResponse(**data)
            company_news = response_model.news
        except Exception as e:
            logger.warning("Failed to parse company news response for %s: %s", ticker, e)
            break

        if not company_news:
            break

        all_news.extend(company_news)

        # Only continue pagination if we have a start_date and got a full page
        if not start_date or len(company_news) < limit:
            break

        # Update end_date to the oldest date from current batch for next iteration
        current_end_date = min(news.date for news in company_news).split("T")[0]

        # If we've reached or passed the start_date, we can stop
        if current_end_date <= start_date:
            break

    if not all_news:
        return []

    # Cache the results using the comprehensive cache key
    _cache.set_company_news(cache_key, [news.model_dump() for news in all_news])
    return all_news


def get_market_cap(
    ticker: str,
    end_date: str,
    api_key: str = None,
) -> float | None:
    """Fetch market cap from cache, financial metrics, Tencent, or company facts API."""
    from src.data.sources.base import classify_ticker

    market = classify_ticker(ticker)

    # Try Tencent real-time quote for CN/HK stocks (always available)
    if market in ("cn", "hk"):
        from src.data.sources.tencent_src import TencentSource
        quote = TencentSource().get_realtime_quote(ticker)
        if quote and quote.get("market_cap"):
            return quote["market_cap"]

    # For non-HK stocks when end_date is today, try company facts API first
    if market not in ("hk", "cn") and end_date == datetime.datetime.now().strftime("%Y-%m-%d"):
        headers = {}
        financial_api_key = api_key or os.environ.get("FINANCIAL_DATASETS_API_KEY")
        if financial_api_key:
            headers["X-API-KEY"] = financial_api_key

        url = f"https://api.financialdatasets.ai/company/facts/?ticker={ticker}"
        response = _make_api_request(url, headers)
        if response is not None and response.status_code == 200:
            data = response.json()
            response_model = CompanyFactsResponse(**data)
            return response_model.company_facts.market_cap

    # Try to get market cap from financial metrics (works for all markets)
    financial_metrics = get_financial_metrics(ticker, end_date, api_key=api_key)
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


# Update the get_price_data function to use the new functions
def get_price_data(ticker: str, start_date: str, end_date: str, api_key: str = None) -> pd.DataFrame:
    prices = get_prices(ticker, start_date, end_date, api_key=api_key)
    return prices_to_df(prices)
