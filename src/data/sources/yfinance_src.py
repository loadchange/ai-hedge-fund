from __future__ import annotations

import logging
from datetime import datetime

import yfinance as yf

from src.data.models import FinancialMetrics, Price
from .base import DataSource, get_proxy_dict

logger = logging.getLogger(__name__)

# Silence yfinance's own logger to suppress 404 / "possibly delisted" noise
_yf_logger = logging.getLogger("yfinance")
_yf_logger.setLevel(logging.CRITICAL)
_yf_logger.propagate = False


def _safe_float(val, default=None):
    """Convert to float, return default if None or invalid."""
    if val is None:
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


class YFinanceSource(DataSource):
    """Adapter for Yahoo Finance via yfinance."""

    def __init__(self):
        # Configure proxy for yfinance (uses curl_cffi internally, needs dict format)
        proxies = get_proxy_dict()
        if proxies:
            yf.config.network.proxy = proxies

    @property
    def name(self) -> str:
        return "yfinance"

    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list[Price]:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(start=start_date, end=end_date, auto_adjust=True)
            if hist.empty:
                return []

            prices = []
            for idx, row in hist.iterrows():
                time_str = idx.strftime("%Y-%m-%d") if hasattr(idx, "strftime") else str(idx)[:10]
                prices.append(
                    Price(
                        open=_safe_float(row.get("Open"), 0.0),
                        close=_safe_float(row.get("Close"), 0.0),
                        high=_safe_float(row.get("High"), 0.0),
                        low=_safe_float(row.get("Low"), 0.0),
                        volume=int(row.get("Volume", 0)),
                        time=time_str,
                        source=self.name,
                    )
                )
            return prices
        except Exception as e:
            err_msg = str(e).lower()
            if "rate" in err_msg or "429" in err_msg or "too many" in err_msg:
                logger.debug("yfinance rate limited for %s, skipping", ticker)
            else:
                logger.warning("yfinance price error for %s: %s", ticker, e)
            return []

    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list[FinancialMetrics]:
        try:
            t = yf.Ticker(ticker)
            info = t.info
            if not info or info.get("regularMarketPrice") is None:
                return []

            report_period = end_date
            market_cap = _safe_float(info.get("marketCap"))

            metric = FinancialMetrics(
                ticker=ticker,
                report_period=report_period,
                period="ttm",
                currency=info.get("currency", "USD"),
                market_cap=market_cap,
                enterprise_value=_safe_float(info.get("enterpriseValue")),
                price_to_earnings_ratio=_safe_float(info.get("trailingPE")),
                price_to_book_ratio=_safe_float(info.get("priceToBook")),
                price_to_sales_ratio=_safe_float(info.get("priceToSalesTrailing12Months")),
                enterprise_value_to_ebitda_ratio=_safe_float(info.get("enterpriseToEbitda")),
                enterprise_value_to_revenue_ratio=_safe_float(info.get("enterpriseToRevenue")),
                free_cash_flow_yield=_safe_float(info.get("freeCashflowYield")),
                peg_ratio=_safe_float(info.get("pegRatio")),
                gross_margin=_safe_float(info.get("grossMargins")),
                operating_margin=_safe_float(info.get("operatingMargins")),
                net_margin=_safe_float(info.get("profitMargins")),
                return_on_equity=_safe_float(info.get("returnOnEquity")),
                return_on_assets=_safe_float(info.get("returnOnAssets")),
                return_on_invested_capital=None,
                asset_turnover=_safe_float(info.get("assetTurnover")),
                current_ratio=_safe_float(info.get("currentRatio")),
                quick_ratio=_safe_float(info.get("quickRatio")),
                cash_ratio=None,
                debt_to_equity=_safe_float(info.get("debtToEquity")),
                debt_to_assets=None,
                interest_coverage=None,
                revenue_growth=_safe_float(info.get("revenueGrowth")),
                earnings_growth=_safe_float(info.get("earningsGrowth")),
                earnings_per_share=_safe_float(info.get("trailingEps")),
                book_value_per_share=_safe_float(info.get("bookValue")),
                source=self.name,
            )
            return [metric]
        except Exception as e:
            err_msg = str(e).lower()
            if "rate" in err_msg or "429" in err_msg or "too many" in err_msg:
                logger.debug("yfinance rate limited for %s metrics, skipping", ticker)
            else:
                logger.warning("yfinance metrics error for %s: %s", ticker, e)
            return []
