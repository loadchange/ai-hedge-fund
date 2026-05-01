from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from src.data.models import (
        CompanyFacts,
        CompanyNews,
        Earnings,
        FinancialMetrics,
        InsiderTrade,
        Price,
    )


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
    upper = ticker.upper()
    code = ticker.replace(".SS", "").replace(".SZ", "").replace(".ss", "").replace(".sz", "")

    if source in ("akshare", "tencent"):
        if upper.endswith(".HK"):
            return ticker.split(".")[0]  # 9988.HK -> 9988
        # A-share: need sh/sz prefix (no dot) for akshare/tencent
        if code.isdigit() and len(code) == 6:
            if upper.endswith(".SZ") or code.startswith(("0", "3")):
                return f"sz{code}"
            return f"sh{code}"
        return ticker

    if source == "baostock":
        # baostock A-share format uses a dot: "sh.600519" / "sz.002594".
        if code.isdigit() and len(code) == 6:
            if upper.endswith(".SZ") or code.startswith(("0", "3")):
                return f"sz.{code}"
            return f"sh.{code}"
        return ticker

    # yfinance and financialdatasets use standard format.
    return ticker


class DataSource(ABC):
    """Abstract base for all data sources used by ``DataSourceManager``.

    Multi-source orchestration contract: each adapter (financialdatasets,
    yfinance, akshare, tencent) implements this so the manager can pick
    a provider per market and fall back on failure.
    """

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


@runtime_checkable
class DataClient(Protocol):
    """Protocol for richer single-provider clients (e.g. ``FinancialDatasetsSource``).

    Distinct from ``DataSource`` (which is the multi-source adapter
    contract): ``DataClient`` exposes the full read-only API of a single
    provider — quotes, financials, news, insider trades, company facts,
    earnings — and is consumed by quantitative modules (signals,
    features, event studies) that talk to one provider at a time.

    Methods return empty lists or ``None`` on failure — never raise.
    """

    def get_prices(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> list["Price"]: ...

    def get_financial_metrics(
        self,
        ticker: str,
        end_date: str,
        period: str = "ttm",
        limit: int = 10,
    ) -> list["FinancialMetrics"]: ...

    def get_news(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list["CompanyNews"]: ...

    def get_insider_trades(
        self,
        ticker: str,
        end_date: str,
        start_date: str | None = None,
        limit: int = 1000,
    ) -> list["InsiderTrade"]: ...

    def get_company_facts(self, ticker: str) -> "CompanyFacts | None": ...

    def get_earnings(self, ticker: str) -> "Earnings | None": ...
