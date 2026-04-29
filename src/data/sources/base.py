from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.data.models import Price, FinancialMetrics


def get_proxy_dict() -> dict[str, str] | None:
    """Return proxies dict for requests library from env vars, or None if not configured."""
    http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    if http or https:
        proxies: dict[str, str] = {}
        if http:
            proxies["http"] = http
        if https:
            proxies["https"] = https
        return proxies
    return None


def classify_ticker(ticker: str) -> str:
    """Classify ticker by market. Returns 'cn', 'hk', or 'us'."""
    upper = ticker.upper()
    if upper.endswith(".HK"):
        return "hk"
    if upper.endswith(".SS") or upper.endswith(".SZ"):
        return "cn"
    # 6-digit pure number → A-share
    if ticker.isdigit() and len(ticker) == 6:
        return "cn"
    return "us"


def normalize_ticker(ticker: str, source: str) -> str:
    """Normalize ticker format for a given source."""
    if source in ("akshare", "tencent"):
        upper = ticker.upper()
        if upper.endswith(".HK"):
            return ticker.split(".")[0]  # 9988.HK -> 9988
        # A-share: need sh/sz prefix for akshare
        code = ticker.replace(".SS", "").replace(".SZ", "").replace(".ss", "").replace(".sz", "")
        if code.isdigit() and len(code) == 6:
            if upper.endswith(".SZ") or code.startswith(("0", "3")):
                return f"sz{code}"
            return f"sh{code}"
        return ticker
    # yfinance and financialdatasets use standard format
    return ticker


class DataSource(ABC):
    """Abstract base for all data sources."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Source identifier for logging and metadata."""
        ...

    @abstractmethod
    def get_prices(
        self, ticker: str, start_date: str, end_date: str
    ) -> list["Price"]:
        """Fetch daily OHLCV price data."""
        ...

    @abstractmethod
    def get_financial_metrics(
        self, ticker: str, end_date: str, period: str = "ttm", limit: int = 10
    ) -> list["FinancialMetrics"]:
        """Fetch financial metrics (ROE, margins, etc.)."""
        ...
